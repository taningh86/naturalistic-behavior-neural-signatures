"""
Change-point detection across ALL sessions (1-8), LHA and RSP.

For each session:
1. Load GRU-ODE hidden states (full session)
2. Compute flow speed, gate value, divergence, traj speed, PC1, pop FR
3. Smooth with 30s window
4. Run Max-U change-point scan (200-1200s)
5. Compare detected CP to behavioral transition time (last hesitant bout)

Output:
- Summary table (CSV): session x region x metric -> CP time, behavioral transition, delta
- Per-session change-point scan figures
- Summary figure: CP times vs behavioral transition across sessions
"""

import yaml
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import spikeinterface.extractors as se
import torch
import torch.nn as nn
from torchdiffeq import odeint
from sklearn.decomposition import PCA
from scipy import stats as sp_stats
from scipy.ndimage import uniform_filter1d
import warnings, sys

warnings.filterwarnings('ignore')

BIN_SIZE_MS = 10
FS = 30000
BIN_SAMPLES = int(BIN_SIZE_MS * FS / 1000)
D_SHARED = 32
HIDDEN_SIZE = 32
ODE_GATE_HIDDEN = 64
ODE_DT = 1.0
PRED_BINS = 10
LHA_DEPTH_MAX = 1300
RSP_DEPTH_MIN = 1300
TIME_CUTOFF = 800

SMOOTH_WINDOW_SEC = 30
SUBSAMPLE_EVERY = 100  # every 1s

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open("paths.yaml") as f:
    paths_config = yaml.safe_load(f)

SESSION_INFO = {
    1: {'state': 'Fed', 'phase': 'Exploration'},
    2: {'state': 'Fed', 'phase': 'Foraging'},
    3: {'state': 'Fed', 'phase': 'Exploration'},
    4: {'state': 'Fed', 'phase': 'Foraging'},
    5: {'state': 'Fasted', 'phase': 'Exploration'},
    6: {'state': 'Fasted', 'phase': 'Foraging'},
    7: {'state': 'Fasted', 'phase': 'Exploration'},
    8: {'state': 'Fasted', 'phase': 'Foraging'},
}

METRICS = ['flow_speed', 'gate_mean', 'divergence', 'traj_speed', 'pc1', 'pop_fr']


class GRUODEFunc(nn.Module):
    def __init__(self, hidden_size, gate_hidden=64):
        super().__init__()
        self.update_gate = nn.Sequential(
            nn.Linear(hidden_size, gate_hidden), nn.Tanh(),
            nn.Linear(gate_hidden, hidden_size), nn.Sigmoid(),
        )
        self.candidate = nn.Sequential(
            nn.Linear(hidden_size, gate_hidden), nn.Tanh(),
            nn.Linear(gate_hidden, hidden_size), nn.Tanh(),
        )

    def forward(self, t, h):
        z = self.update_gate(h)
        n = self.candidate(h)
        return (1 - z) * (n - h)


class PooledGRUODE(nn.Module):
    def __init__(self, session_neuron_counts, d_shared, hidden_size,
                 gate_hidden=64, pred_steps=10):
        super().__init__()
        self.d_shared = d_shared
        self.hidden_size = hidden_size
        self.pred_steps = pred_steps
        self.input_projections = nn.ModuleDict()
        for sn, n_neurons in session_neuron_counts.items():
            self.input_projections[str(sn)] = nn.Linear(n_neurons, d_shared)
        self.ode_func = GRUODEFunc(hidden_size, gate_hidden)
        self.obs_cell = nn.GRUCell(input_size=d_shared, hidden_size=hidden_size)
        self.fc_shared = nn.Linear(hidden_size, d_shared)
        self.output_projections = nn.ModuleDict()
        for sn, n_neurons in session_neuron_counts.items():
            self.output_projections[str(sn)] = nn.Linear(d_shared, n_neurons)
        self.register_buffer('t_span', torch.tensor([0.0, ODE_DT]))


def load_model(region, state):
    condition = "fed" if state == "Fed" else "fasted"
    model_path = Path("data") / f"gru_ode_10ms_poisson_{region}_{condition}_model.pt"
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    model = PooledGRUODE(
        checkpoint['neuron_counts'],
        checkpoint['config']['d_shared'],
        checkpoint['config']['hidden_size'],
        checkpoint['config'].get('gate_hidden', ODE_GATE_HIDDEN),
        checkpoint['config'].get('pred_bins', PRED_BINS),
    ).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def evaluate_flow(ode_func, points):
    h = torch.tensor(points, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        return ode_func(0.0, h).cpu().numpy()


def compute_local_divergence(ode_func, points, eps=0.01):
    divs = np.zeros(len(points))
    for d in range(HIDDEN_SIZE):
        perturb = np.zeros(HIDDEN_SIZE)
        perturb[d] = eps
        f_plus = evaluate_flow(ode_func, points + perturb)
        f_minus = evaluate_flow(ode_func, points - perturb)
        divs += (f_plus[:, d] - f_minus[:, d]) / (2 * eps)
    return divs


def max_u_changepoint(values, times, min_before=20, min_after=20):
    """Find time that maximizes Mann-Whitney U separation between before/after."""
    n = len(values)
    best_stat = 0.5
    best_idx = n // 2
    best_pval = 1.0
    scan_stats = []
    scan_times = []

    for split in range(min_before, n - min_after):
        before = values[:split]
        after = values[split:]
        try:
            stat, pval = sp_stats.mannwhitneyu(before, after, alternative='two-sided')
            norm_stat = stat / (len(before) * len(after))
        except:
            norm_stat = 0.5
            pval = 1.0
        scan_stats.append(norm_stat)
        scan_times.append(times[split])
        if abs(norm_stat - 0.5) > abs(best_stat - 0.5):
            best_stat = norm_stat
            best_idx = split
            best_pval = pval

    return times[best_idx], best_stat, best_pval, np.array(scan_times), np.array(scan_stats)


def process_session(sess_num, exc_df):
    """Process one session: compute metrics, find CPs, return results dict."""
    info = SESSION_INFO[sess_num]
    state = info['state']
    phase = info['phase']

    print(f"\n{'='*70}")
    print(f"  Session {sess_num} ({state}, {phase})")
    print(f"{'='*70}")
    sys.stdout.flush()

    # --- Excursion classification ---
    s_df = exc_df[exc_df["session"] == sess_num].copy()
    not_pot = s_df["farthest_zone"] != "Pot"
    all_hes = s_df[(s_df["feeding_bins"] == 0) & (s_df["digging_bins"] == 0) &
                   not_pot & (s_df["reversals"] >= 1) & (s_df["duration"] >= 2.0)]
    hesitant = all_hes[all_hes["start_time"] < TIME_CUTOFF]
    task = s_df[(s_df["feeding_bins"] > 0) | (s_df["digging_bins"] > 0)]
    non_hes_non_task = s_df[~s_df.index.isin(all_hes.index) & ~s_df.index.isin(task.index)]
    committed = non_hes_non_task[non_hes_non_task["reached_arena"] == True]

    n_hes = len(hesitant)
    n_com = len(committed)
    n_task = len(task)
    behav_transition = hesitant["end_time"].max() if n_hes > 0 else np.nan

    print(f"  Hesitant(<{TIME_CUTOFF}s): {n_hes}, Committed: {n_com}, Task: {n_task}")
    if n_hes > 0:
        print(f"  Behavioral transition (last hesitant end): {behav_transition:.1f}s")
    else:
        print(f"  No early hesitant bouts — no behavioral transition defined")
    sys.stdout.flush()

    # --- Load spike data ---
    sp = paths_config['single_probe']['coordinates_1']['mouse01']['sessions']
    sorted_path = Path(sp[f'session_{sess_num}']['sorted'])
    sorting = se.read_kilosort(sorted_path)
    ci = pd.read_csv(sorted_path / "cluster_info.tsv", sep="\t")
    lc = "group" if "group" in ci.columns and ci["group"].eq("good").any() else "KSLabel"
    good = ci[ci[lc] == "good"]
    lha_ids = good[good["depth"] < LHA_DEPTH_MAX]["cluster_id"].values
    rsp_ids = good[good["depth"] >= RSP_DEPTH_MIN]["cluster_id"].values
    print(f"  Good units: {len(lha_ids)} LHA, {len(rsp_ids)} RSP")

    # --- Pop FR at 100ms ---
    FR_BIN = 100
    FR_SAMP = int(FR_BIN * FS / 1000)

    def get_fr(uids):
        amin = np.inf; amax = 0; sts = {}
        for u in uids:
            st = sorting.get_unit_spike_train(u); sts[u] = st
            if len(st) > 0: amin = min(amin, np.min(st)); amax = max(amax, np.max(st))
        nb = int((amax - amin) / FR_SAMP) + 1
        d = np.zeros((nb, len(uids)), dtype=np.float32)
        for i, u in enumerate(uids):
            st = sts[u]
            if len(st) > 0:
                b = ((st - amin) // FR_SAMP).astype(int)
                b = b[(b >= 0) & (b < nb)]
                np.add.at(d[:, i], b, 1)
        return d / (FR_BIN / 1000.0), amin

    lha_fr, lha_fr_amin = get_fr(lha_ids)
    rsp_fr, rsp_fr_amin = get_fr(rsp_ids)

    session_results = []

    for region, unit_ids, fr_data, fr_amin in [
            ("LHA", lha_ids, lha_fr, lha_fr_amin),
            ("RSP", rsp_ids, rsp_fr, rsp_fr_amin)]:

        print(f"\n  --- {region} ---")
        sys.stdout.flush()

        h_file = f"gru_ode_10ms_hidden_{region.lower()}_s{sess_num}.npy"
        h_states = np.load(f"data/{h_file}")

        model = load_model(region.lower(), state)
        ode_func = model.ode_func
        pca = PCA(n_components=3)
        pca.fit(h_states)

        # Time alignment
        allmin = np.inf
        for u in unit_ids:
            st = sorting.get_unit_spike_train(u)
            if len(st) > 0:
                allmin = min(allmin, np.min(st))
        offset = allmin / FS

        n_bins = len(h_states)
        h_times = offset + np.arange(n_bins) * 0.01

        MAX_TIME = 1500
        mask_h = h_times <= MAX_TIME
        h_states_clip = h_states[mask_h]
        h_times_clip = h_times[mask_h]
        n_clip = len(h_states_clip)

        # FR
        fr_offset = fr_amin / FS
        fr_times = fr_offset + np.arange(len(fr_data)) * 0.1
        fr_mask = fr_times <= MAX_TIME
        fr_data_clip = fr_data[fr_mask]
        fr_times_clip = fr_times[fr_mask]
        pop_fr = fr_data_clip.mean(axis=1)

        # Subsample for expensive metrics
        sub_idx = np.arange(0, n_clip, SUBSAMPLE_EVERY)
        sub_points = h_states_clip[sub_idx]
        sub_times = h_times_clip[sub_idx]

        # Flow speed
        dhdt = evaluate_flow(ode_func, sub_points)
        flow_speed = np.linalg.norm(dhdt, axis=1)

        # Gate
        h_tensor = torch.tensor(sub_points, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            z_gate = ode_func.update_gate(h_tensor).cpu().numpy()
        gate_mean = z_gate.mean(axis=1)

        # Divergence
        divergence = compute_local_divergence(ode_func, sub_points)

        # Traj speed
        traj_speed_full = np.linalg.norm(np.diff(h_states_clip, axis=0), axis=1)
        traj_speed_sub = np.array([
            traj_speed_full[max(0, i-SUBSAMPLE_EVERY//2):min(len(traj_speed_full), i+SUBSAMPLE_EVERY//2)].mean()
            for i in sub_idx[:-1]
        ])
        traj_speed_sub = np.append(traj_speed_sub, traj_speed_sub[-1])

        # PC1
        h_pcs_clip = pca.transform(h_states_clip)
        pc1_sub = h_pcs_clip[sub_idx, 0]

        # Smooth
        smooth_n = SMOOTH_WINDOW_SEC
        flow_speed_sm = uniform_filter1d(flow_speed, smooth_n)
        gate_mean_sm = uniform_filter1d(gate_mean, smooth_n)
        divergence_sm = uniform_filter1d(divergence, smooth_n)
        traj_speed_sm = uniform_filter1d(traj_speed_sub, smooth_n)
        pc1_sm = uniform_filter1d(pc1_sub, smooth_n)
        pop_fr_sm = uniform_filter1d(pop_fr, 300)

        # Change-point scan
        metric_data = {
            'flow_speed': (flow_speed_sm, sub_times),
            'gate_mean': (gate_mean_sm, sub_times),
            'divergence': (divergence_sm, sub_times),
            'traj_speed': (traj_speed_sm, sub_times),
            'pc1': (pc1_sm, sub_times),
            'pop_fr': (pop_fr_sm, fr_times_clip),
        }

        for mname in METRICS:
            mvals, mtimes = metric_data[mname]
            tmask = (mtimes >= 200) & (mtimes <= 1200)
            vals = mvals[tmask]
            ts = mtimes[tmask]
            if len(vals) < 50:
                print(f"    {mname}: too few points, skipping")
                continue
            cp_time, cp_stat, cp_pval, _, _ = max_u_changepoint(
                vals, ts, min_before=20, min_after=20)
            delta = cp_time - behav_transition if not np.isnan(behav_transition) else np.nan
            session_results.append({
                'session': sess_num, 'state': state, 'phase': phase,
                'region': region, 'metric': mname,
                'cp_time': cp_time, 'u_norm': cp_stat, 'cp_pval': cp_pval,
                'behav_transition': behav_transition,
                'delta_s': delta,
                'n_hesitant': n_hes,
            })
            dsign = f"{delta:+.0f}s" if not np.isnan(delta) else "N/A"
            print(f"    {mname:<15}: CP={cp_time:.0f}s  U={cp_stat:.3f}  "
                  f"p={cp_pval:.1e}  delta={dsign}")

        sys.stdout.flush()

    return session_results


def main():
    print("=" * 70)
    print("  CHANGE-POINT DETECTION — ALL SESSIONS (1-8)")
    print("  LHA and RSP, 6 metrics each")
    print("=" * 70)
    sys.stdout.flush()

    exc_df = pd.read_csv("data/excursion_features_all_sessions.csv")
    all_results = []

    for sess_num in range(1, 9):
        results = process_session(sess_num, exc_df)
        all_results.extend(results)

    # Save CSV
    results_df = pd.DataFrame(all_results)
    results_df.to_csv("data/hesitant_transition_changepoints_all.csv", index=False)
    print(f"\nSaved: data/hesitant_transition_changepoints_all.csv ({len(results_df)} rows)")

    # =============================================================
    # SUMMARY TABLE
    # =============================================================
    print("\n" + "=" * 70)
    print("  SUMMARY: Change-point times vs behavioral transition")
    print("=" * 70)

    for region in ["LHA", "RSP"]:
        rdf = results_df[results_df["region"] == region]
        print(f"\n  {region}:")
        print(f"  {'Session':<10} {'State':<8} {'BehavT':<8} "
              + "  ".join(f"{m[:8]:<10}" for m in METRICS))
        print(f"  {'-'*90}")
        for sess in range(1, 9):
            sdf = rdf[rdf["session"] == sess]
            if len(sdf) == 0:
                continue
            bt = sdf.iloc[0]["behav_transition"]
            bt_str = f"{bt:.0f}s" if not np.isnan(bt) else "N/A"
            state = sdf.iloc[0]["state"]
            vals = []
            for m in METRICS:
                mrow = sdf[sdf["metric"] == m]
                if len(mrow) > 0:
                    cp = mrow.iloc[0]["cp_time"]
                    delta = mrow.iloc[0]["delta_s"]
                    if not np.isnan(delta):
                        vals.append(f"{cp:.0f}({delta:+.0f})")
                    else:
                        vals.append(f"{cp:.0f}")
                else:
                    vals.append("---")
            print(f"  S{sess:<9} {state:<8} {bt_str:<8} "
                  + "  ".join(f"{v:<10}" for v in vals))

    # =============================================================
    # FIGURE: CP times vs behavioral transition for all sessions
    # =============================================================
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))
    fig.suptitle('Change-Point Times vs Behavioral Transition (All Sessions)\n'
                 'Each dot = one metric\'s detected CP; red line = last hesitant bout',
                 fontsize=14, fontweight='bold')

    metric_colors = {
        'flow_speed': '#9C27B0', 'gate_mean': '#795548', 'divergence': '#009688',
        'traj_speed': '#FF9800', 'pc1': '#1A237E', 'pop_fr': '#212121'
    }
    metric_markers = {
        'flow_speed': 'o', 'gate_mean': 's', 'divergence': 'D',
        'traj_speed': '^', 'pc1': 'v', 'pop_fr': 'P'
    }

    for ri, region in enumerate(["LHA", "RSP"]):
        ax = axes[ri]
        rdf = results_df[results_df["region"] == region]

        sessions_with_hes = []
        for sess in range(1, 9):
            sdf = rdf[rdf["session"] == sess]
            if len(sdf) == 0:
                continue
            bt = sdf.iloc[0]["behav_transition"]
            n_hes = sdf.iloc[0]["n_hesitant"]

            # Plot behavioral transition
            if not np.isnan(bt):
                ax.axhline(bt, xmin=(sess-0.7)/8.6, xmax=(sess+0.3)/8.6,
                           color='red', linewidth=2, alpha=0.7)
                sessions_with_hes.append(sess)

            # Plot each metric's CP
            for mi, m in enumerate(METRICS):
                mrow = sdf[sdf["metric"] == m]
                if len(mrow) > 0:
                    cp = mrow.iloc[0]["cp_time"]
                    jitter = (mi - 2.5) * 0.06
                    ax.scatter(sess + jitter, cp, color=metric_colors[m],
                               marker=metric_markers[m], s=80, zorder=5,
                               edgecolors='black', linewidths=0.5)

            # Annotate n_hesitant
            ax.text(sess, 1220, f'n={n_hes}', ha='center', fontsize=8, color='gray')

        ax.set_ylabel('Time (s)')
        ax.set_title(f'{region}', fontsize=13, fontweight='bold')
        ax.set_xticks(range(1, 9))
        ax.set_xticklabels([f'S{s}\n{SESSION_INFO[s]["state"][:3]}\n{SESSION_INFO[s]["phase"][:3]}'
                            for s in range(1, 9)], fontsize=9)
        ax.set_ylim(150, 1250)
        ax.axhspan(700, 850, color='red', alpha=0.05)  # typical transition zone

        if ri == 0:
            handles = []
            for m in METRICS:
                handles.append(ax.scatter([], [], color=metric_colors[m],
                               marker=metric_markers[m], s=60, label=m,
                               edgecolors='black', linewidths=0.5))
            handles.append(plt.Line2D([0], [0], color='red', linewidth=2,
                                       label='Behav. transition'))
            ax.legend(handles=handles, loc='upper right', fontsize=8, ncol=2)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig('figures/hesitant_transition_cp_all_sessions.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: figures/hesitant_transition_cp_all_sessions.png")

    # =============================================================
    # FIGURE: Delta (CP - behavioral transition) distribution
    # =============================================================
    valid = results_df.dropna(subset=["delta_s"])
    if len(valid) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('Distance Between Neural Change-Point and Behavioral Transition\n'
                     'Delta = CP_time - Behavioral_transition (negative = CP before transition)',
                     fontsize=13, fontweight='bold')

        for ri, region in enumerate(["LHA", "RSP"]):
            ax = axes[ri]
            rdf = valid[valid["region"] == region]

            for m in METRICS:
                mdf = rdf[rdf["metric"] == m]
                if len(mdf) > 0:
                    deltas = mdf["delta_s"].values
                    sessions = mdf["session"].values
                    ax.scatter(sessions, deltas, color=metric_colors[m],
                               marker=metric_markers[m], s=80,
                               edgecolors='black', linewidths=0.5, label=m)

            ax.axhline(0, color='red', linewidth=1.5, linestyle='--', alpha=0.7,
                       label='Perfect alignment')
            ax.axhspan(-100, 100, color='green', alpha=0.05)
            ax.set_xlabel('Session')
            ax.set_ylabel('Delta (s) = CP - Behavioral transition')
            ax.set_title(f'{region}', fontsize=12, fontweight='bold')
            ax.set_xticks(range(1, 9))

            # Annotate median delta per metric
            text_lines = []
            for m in METRICS:
                mdf = rdf[rdf["metric"] == m]
                if len(mdf) > 0:
                    med = np.median(mdf["delta_s"])
                    text_lines.append(f'{m[:10]:10s} med={med:+.0f}s')
            ax.text(0.02, 0.02, '\n'.join(text_lines), transform=ax.transAxes,
                    fontsize=7, fontfamily='monospace', va='bottom',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            if ri == 0:
                ax.legend(fontsize=7, ncol=2, loc='upper right')

        plt.tight_layout(rect=[0, 0, 1, 0.90])
        plt.savefig('figures/hesitant_transition_delta_all_sessions.png',
                    dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: figures/hesitant_transition_delta_all_sessions.png")

    print("\n" + "=" * 70)
    print("  DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
