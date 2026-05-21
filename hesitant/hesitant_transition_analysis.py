"""
Transition analysis: continuous neural state across Session 1.

Goal: detect whether LHA/RSP neural dynamics shift around the time
the animal transitions from hesitant exploration to committed exploration (~750s).

Approach:
1. Compute continuous GRU-ODE flow properties + population FR across the full session
2. Overlay excursion types (hesitant, committed, task-engaged)
3. Sliding-window smoothing to see slow trends
4. Change-point detection: find the time that maximizes Mann-Whitney U between
   the before/after distributions (max-U scan)
5. Compare pre-transition (0-750s) vs post-transition (750-1500s) distributions

All metrics are time-resolved — no averaging over bouts.
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
TIME_CUTOFF = 800  # seconds — hesitant bouts before this

# Sliding window for smoothing (in seconds)
SMOOTH_WINDOW_SEC = 30  # 30s window
# Subsampling: compute expensive metrics every N bins
SUBSAMPLE_EVERY = 100  # every 1s (100 x 10ms bins)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open("paths.yaml") as f:
    paths_config = yaml.safe_load(f)


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


def load_model(region):
    model_path = Path("data") / f"gru_ode_10ms_poisson_{region}_fed_model.pt"
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
    """Find time that maximizes Mann-Whitney U between before/after.
    Returns (best_time, best_stat, best_pval, stats_at_each_time).
    """
    n = len(values)
    best_stat = 0.5  # 0.5 = no difference; deviation from 0.5 = stronger split
    best_idx = n // 2
    best_pval = 1.0
    scan_stats = []
    scan_times = []

    for split in range(min_before, n - min_after):
        before = values[:split]
        after = values[split:]
        try:
            stat, pval = sp_stats.mannwhitneyu(before, after, alternative='two-sided')
            # Normalize stat to 0-1 range: U / (n1 * n2)
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


def main():
    print("=" * 70)
    print("  TRANSITION ANALYSIS: Continuous Neural State Across Session 1")
    print("  Tracking hesitant -> committed exploration switch")
    print("=" * 70)
    sys.stdout.flush()

    # --- Load excursion data ---
    exc_df = pd.read_csv("data/excursion_features_all_sessions.csv")
    s1 = exc_df[exc_df["session"] == 1].copy()

    # Classify excursions
    not_pot = s1["farthest_zone"] != "Pot"
    all_hes = s1[(s1["feeding_bins"] == 0) & (s1["digging_bins"] == 0) &
                 not_pot & (s1["reversals"] >= 1) & (s1["duration"] >= 2.0)]
    hesitant = all_hes[all_hes["start_time"] < TIME_CUTOFF]
    task = s1[(s1["feeding_bins"] > 0) | (s1["digging_bins"] > 0)]
    non_hes_non_task = s1[~s1.index.isin(all_hes.index) & ~s1.index.isin(task.index)]
    committed = non_hes_non_task[non_hes_non_task["reached_arena"] == True]

    print(f"  Excursions: {len(hesitant)} early hesitant, "
          f"{len(committed)} committed, {len(task)} task-engaged")
    last_hes_time = hesitant["end_time"].max() if len(hesitant) > 0 else 0
    first_com_time = committed["start_time"].min() if len(committed) > 0 else 0
    print(f"  Last hesitant ends: {last_hes_time:.1f}s")
    print(f"  First committed starts: {first_com_time:.1f}s")
    print(f"  Transition gap: {first_com_time - last_hes_time:.1f}s")
    sys.stdout.flush()

    # --- Load spike data ---
    sp = paths_config['single_probe']['coordinates_1']['mouse01']['sessions']
    sorted_path = Path(sp['session_1']['sorted'])
    sorting = se.read_kilosort(sorted_path)
    ci = pd.read_csv(sorted_path / "cluster_info.tsv", sep="\t")
    lc = "group" if "group" in ci.columns and ci["group"].eq("good").any() else "KSLabel"
    good = ci[ci[lc] == "good"]
    lha_ids = good[good["depth"] < LHA_DEPTH_MAX]["cluster_id"].values
    rsp_ids = good[good["depth"] >= RSP_DEPTH_MIN]["cluster_id"].values
    print(f"  Good units: {len(lha_ids)} LHA, {len(rsp_ids)} RSP")

    # --- Compute population FR at 100ms ---
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

    print("  Loading firing rates...")
    sys.stdout.flush()
    lha_fr, lha_fr_amin = get_fr(lha_ids)
    rsp_fr, rsp_fr_amin = get_fr(rsp_ids)

    # --- Analyze each region ---
    for region, unit_ids, h_file, fr_data, fr_amin in [
            ("LHA", lha_ids, "gru_ode_10ms_hidden_lha_s1.npy", lha_fr, lha_fr_amin),
            ("RSP", rsp_ids, "gru_ode_10ms_hidden_rsp_s1.npy", rsp_fr, rsp_fr_amin)]:

        print(f"\n{'='*70}")
        print(f"  {region} — Continuous State Tracking")
        print(f"{'='*70}")
        sys.stdout.flush()

        # Time alignment
        allmin = np.inf
        for u in unit_ids:
            st = sorting.get_unit_spike_train(u)
            if len(st) > 0:
                allmin = min(allmin, np.min(st))
        offset = allmin / FS

        h_states = np.load(f"data/{h_file}")
        model = load_model(region.lower())
        ode_func = model.ode_func
        pca = PCA(n_components=3)
        pca.fit(h_states)
        h_pcs = pca.transform(h_states)

        # Time axis for hidden states (10ms bins)
        n_bins = len(h_states)
        h_times = offset + np.arange(n_bins) * 0.01  # in seconds

        # Limit to first 1500s for cleaner plots
        MAX_TIME = 1500
        mask_h = h_times <= MAX_TIME
        h_states_clip = h_states[mask_h]
        h_pcs_clip = h_pcs[mask_h]
        h_times_clip = h_times[mask_h]
        n_clip = len(h_states_clip)

        # FR time axis (100ms bins)
        fr_offset = fr_amin / FS
        fr_times = fr_offset + np.arange(len(fr_data)) * 0.1
        fr_mask = fr_times <= MAX_TIME
        fr_data_clip = fr_data[fr_mask]
        fr_times_clip = fr_times[fr_mask]
        pop_fr = fr_data_clip.mean(axis=1)  # mean over neurons

        print(f"  Hidden states: {n_clip} bins ({h_times_clip[0]:.1f}-{h_times_clip[-1]:.1f}s)")
        print(f"  Computing flow properties at {n_clip // SUBSAMPLE_EVERY} time points...")
        sys.stdout.flush()

        # --- Compute subsampled flow properties ---
        sub_idx = np.arange(0, n_clip, SUBSAMPLE_EVERY)
        sub_points = h_states_clip[sub_idx]
        sub_times = h_times_clip[sub_idx]

        # Flow speed
        dhdt = evaluate_flow(ode_func, sub_points)
        flow_speed = np.linalg.norm(dhdt, axis=1)

        # Gate value
        h_tensor = torch.tensor(sub_points, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            z_gate = ode_func.update_gate(h_tensor).cpu().numpy()
        gate_mean = z_gate.mean(axis=1)

        # Local divergence
        print(f"  Computing divergence ({len(sub_idx)} points)...")
        sys.stdout.flush()
        divergence = compute_local_divergence(ode_func, sub_points)

        # Trajectory speed (from hidden states, subsampled)
        traj_speed_full = np.linalg.norm(np.diff(h_states_clip, axis=0), axis=1)
        # Average over SUBSAMPLE_EVERY bins
        traj_speed_sub = np.array([
            traj_speed_full[max(0, i-SUBSAMPLE_EVERY//2):min(len(traj_speed_full), i+SUBSAMPLE_EVERY//2)].mean()
            for i in sub_idx[:-1]
        ])
        traj_speed_sub = np.append(traj_speed_sub, traj_speed_sub[-1])

        # PC1, PC2
        pc1_sub = h_pcs_clip[sub_idx, 0]
        pc2_sub = h_pcs_clip[sub_idx, 1]

        # Smooth all metrics (30s window = 30 points at 1s subsampling)
        smooth_n = SMOOTH_WINDOW_SEC  # number of subsampled points
        flow_speed_sm = uniform_filter1d(flow_speed, smooth_n)
        gate_mean_sm = uniform_filter1d(gate_mean, smooth_n)
        divergence_sm = uniform_filter1d(divergence, smooth_n)
        traj_speed_sm = uniform_filter1d(traj_speed_sub, smooth_n)
        pc1_sm = uniform_filter1d(pc1_sub, smooth_n)
        pc2_sm = uniform_filter1d(pc2_sub, smooth_n)

        # Smooth FR (30s = 300 bins at 100ms)
        pop_fr_sm = uniform_filter1d(pop_fr, 300)

        # =============================================================
        # CHANGE-POINT DETECTION
        # =============================================================
        print(f"  Running change-point detection...")
        sys.stdout.flush()

        cp_results = {}
        for mname, mvals, mtimes in [
                ('flow_speed', flow_speed_sm, sub_times),
                ('gate_mean', gate_mean_sm, sub_times),
                ('divergence', divergence_sm, sub_times),
                ('traj_speed', traj_speed_sm, sub_times),
                ('pc1', pc1_sm, sub_times),
                ('pop_fr', pop_fr_sm, fr_times_clip)]:
            # Only scan in 200-1200s range
            tmask = (mtimes >= 200) & (mtimes <= 1200)
            vals = mvals[tmask]
            ts = mtimes[tmask]
            if len(vals) < 50:
                continue
            cp_time, cp_stat, cp_pval, scan_t, scan_s = max_u_changepoint(
                vals, ts, min_before=20, min_after=20)
            cp_results[mname] = {
                'time': cp_time, 'stat': cp_stat, 'pval': cp_pval,
                'scan_times': scan_t, 'scan_stats': scan_s
            }
            print(f"    {mname:<15}: change-point at {cp_time:.0f}s "
                  f"(U_norm={cp_stat:.3f}, p={cp_pval:.2e})")

        # =============================================================
        # PRE vs POST transition comparison
        # =============================================================
        transition_time = last_hes_time  # ~751s
        pre_mask = (sub_times >= 200) & (sub_times < transition_time)
        post_mask = (sub_times >= transition_time) & (sub_times < transition_time + 750)

        print(f"\n  Pre-transition (200-{transition_time:.0f}s) vs Post-transition "
              f"({transition_time:.0f}-{transition_time+750:.0f}s):")
        print(f"  {'Metric':<15} {'Pre mean':<12} {'Post mean':<12} {'Change%':<10} {'p':<10} {'sig'}")
        print(f"  {'-'*65}")

        for mname, mvals in [('flow_speed', flow_speed_sm), ('gate_mean', gate_mean_sm),
                              ('divergence', divergence_sm), ('traj_speed', traj_speed_sm),
                              ('pc1', pc1_sm)]:
            pre = mvals[pre_mask]
            post = mvals[post_mask]
            if len(pre) < 5 or len(post) < 5:
                continue
            stat, pval = sp_stats.mannwhitneyu(pre, post, alternative='two-sided')
            pm = np.mean(pre)
            qm = np.mean(post)
            chg = ((qm - pm) / abs(pm) * 100) if abs(pm) > 1e-8 else 0
            sig = '*' if pval < 0.05 else 'ns'
            print(f"  {mname:<15} {pm:<12.4f} {qm:<12.4f} {chg:<+10.1f} {pval:<10.4f} {sig}")

        # Same for FR (different time base)
        fr_pre_mask = (fr_times_clip >= 200) & (fr_times_clip < transition_time)
        fr_post_mask = (fr_times_clip >= transition_time) & (fr_times_clip < transition_time + 750)
        pre_fr = pop_fr_sm[fr_pre_mask]
        post_fr = pop_fr_sm[fr_post_mask]
        if len(pre_fr) >= 5 and len(post_fr) >= 5:
            stat, pval = sp_stats.mannwhitneyu(pre_fr, post_fr, alternative='two-sided')
            pm = np.mean(pre_fr)
            qm = np.mean(post_fr)
            chg = ((qm - pm) / abs(pm) * 100) if abs(pm) > 1e-8 else 0
            sig = '*' if pval < 0.05 else 'ns'
            print(f"  {'pop_fr':<15} {pm:<12.4f} {qm:<12.4f} {chg:<+10.1f} {pval:<10.4f} {sig}")

        # =============================================================
        # FIGURE 1: Continuous time series with excursion overlays
        # =============================================================
        fig, axes = plt.subplots(6, 1, figsize=(20, 18), sharex=True)
        fig.suptitle(f'{region} — Continuous Neural State Across Session 1\n'
                     f'30s smoothing | Transition at ~{transition_time:.0f}s '
                     f'(last hesitant → first committed)',
                     fontsize=14, fontweight='bold')

        # Colors for excursion shading
        exc_colors = {'hesitant': '#E53935', 'committed': '#1E88E5', 'task': '#43A047'}

        def shade_excursions(ax):
            for _, row in hesitant.iterrows():
                ax.axvspan(row['start_time'], row['end_time'],
                           color=exc_colors['hesitant'], alpha=0.15)
            for _, row in committed.iterrows():
                if row['start_time'] <= MAX_TIME:
                    ax.axvspan(row['start_time'], row['end_time'],
                               color=exc_colors['committed'], alpha=0.15)
            for _, row in task.iterrows():
                if row['start_time'] <= MAX_TIME:
                    ax.axvspan(row['start_time'], row['end_time'],
                               color=exc_colors['task'], alpha=0.15)

        plot_data = [
            (sub_times, flow_speed_sm, 'Flow speed |dh/dt|', 'purple'),
            (sub_times, traj_speed_sm, 'Trajectory speed', 'darkorange'),
            (sub_times, divergence_sm, 'Local divergence', 'teal'),
            (sub_times, gate_mean_sm, 'Gate value (z)', 'brown'),
            (sub_times, pc1_sm, 'PC1 position', 'navy'),
            (fr_times_clip, pop_fr_sm, f'Population FR ({len(unit_ids)} units, Hz)', 'black'),
        ]

        for i, (t, vals, ylabel, color) in enumerate(plot_data):
            ax = axes[i]
            shade_excursions(ax)
            ax.plot(t, vals, color=color, linewidth=0.8, alpha=0.9)

            # Also plot raw (unsmoothed) as faint background
            if i < 5:
                raw_vals = [flow_speed, traj_speed_sub, divergence, gate_mean, pc1_sub][i]
                ax.plot(sub_times, raw_vals, color=color, linewidth=0.2, alpha=0.15)
            else:
                ax.plot(fr_times_clip, pop_fr, color=color, linewidth=0.1, alpha=0.1)

            # Transition line
            ax.axvline(transition_time, color='red', linewidth=2, linestyle='--',
                       alpha=0.7, label=f'Transition ({transition_time:.0f}s)')

            # Mark change-point if found
            mkey = ['flow_speed', 'traj_speed', 'divergence', 'gate_mean', 'pc1', 'pop_fr'][i]
            if mkey in cp_results:
                cp_t = cp_results[mkey]['time']
                ax.axvline(cp_t, color='gold', linewidth=2, linestyle=':',
                           alpha=0.8, label=f'Change-point ({cp_t:.0f}s)')

            ax.set_ylabel(ylabel, fontsize=10)
            if ylabel.startswith('Local div') or ylabel.startswith('PC1'):
                ax.axhline(0, color='gray', linewidth=0.3, linestyle='--')
            if i == 0:
                ax.legend(loc='upper right', fontsize=8)

        axes[-1].set_xlabel('Time in session (s)', fontsize=12)
        axes[-1].set_xlim(0, MAX_TIME)

        # Add legend for excursion types
        legend_patches = [
            mpatches.Patch(color=exc_colors['hesitant'], alpha=0.3, label='Hesitant'),
            mpatches.Patch(color=exc_colors['committed'], alpha=0.3, label='Committed'),
            mpatches.Patch(color=exc_colors['task'], alpha=0.3, label='Task-engaged'),
        ]
        axes[0].legend(handles=legend_patches + axes[0].get_legend_handles_labels()[0][:2],
                        loc='upper right', fontsize=8, ncol=2)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(f'figures/hesitant_transition_{region.lower()}.png',
                    dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: figures/hesitant_transition_{region.lower()}.png")

        # =============================================================
        # FIGURE 2: Change-point scan (U-statistic landscape)
        # =============================================================
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'{region} — Change-Point Detection (Max-U Scan)\n'
                     f'U_norm = 0.5 means no difference; deviation from 0.5 = stronger split',
                     fontsize=13, fontweight='bold')

        for i, mkey in enumerate(['flow_speed', 'traj_speed', 'divergence',
                                   'gate_mean', 'pc1', 'pop_fr']):
            ax = axes[i // 3, i % 3]
            if mkey in cp_results:
                cp = cp_results[mkey]
                ax.plot(cp['scan_times'], cp['scan_stats'], linewidth=1.5, color='steelblue')
                ax.axhline(0.5, color='gray', linewidth=0.5, linestyle='--')
                ax.axvline(cp['time'], color='gold', linewidth=2, linestyle=':',
                           label=f"CP={cp['time']:.0f}s (p={cp['pval']:.1e})")
                ax.axvline(transition_time, color='red', linewidth=1.5, linestyle='--',
                           label=f'Behavioral ({transition_time:.0f}s)', alpha=0.7)
                ax.legend(fontsize=8)
            ax.set_title(mkey.replace('_', ' ').title())
            ax.set_xlabel('Split time (s)')
            ax.set_ylabel('Normalized U')

        plt.tight_layout(rect=[0, 0, 1, 0.93])
        plt.savefig(f'figures/hesitant_transition_changepoint_{region.lower()}.png',
                    dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: figures/hesitant_transition_changepoint_{region.lower()}.png")

        # =============================================================
        # FIGURE 3: Latent trajectory colored by time (PC1 vs PC2)
        # =============================================================
        fig, axes = plt.subplots(1, 3, figsize=(21, 7))
        fig.suptitle(f'{region} — Latent Trajectory (PC1 vs PC2)\n'
                     f'Color = time in session | Excursion periods marked',
                     fontsize=13, fontweight='bold')

        # Full trajectory colored by time
        ax = axes[0]
        sc = ax.scatter(h_pcs_clip[::50, 0], h_pcs_clip[::50, 1],
                        c=h_times_clip[::50], cmap='viridis', s=1, alpha=0.5)
        plt.colorbar(sc, ax=ax, label='Time (s)')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_title('Full trajectory')

        # Pre-transition only
        ax = axes[1]
        pre_t_mask = h_times_clip <= transition_time
        ax.scatter(h_pcs_clip[pre_t_mask][::20, 0], h_pcs_clip[pre_t_mask][::20, 1],
                   c=h_times_clip[pre_t_mask][::20], cmap='viridis', s=2, alpha=0.4)
        # Overlay hesitant bouts
        for _, row in hesitant.iterrows():
            bmask = (h_times_clip >= row['start_time']) & (h_times_clip <= row['end_time'])
            if bmask.sum() > 0:
                ax.plot(h_pcs_clip[bmask, 0], h_pcs_clip[bmask, 1],
                        color='red', linewidth=1.5, alpha=0.7)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_title(f'Pre-transition (0-{transition_time:.0f}s)\nRed = hesitant bouts')

        # Post-transition only
        ax = axes[2]
        post_t_mask = (h_times_clip > transition_time) & (h_times_clip <= MAX_TIME)
        ax.scatter(h_pcs_clip[post_t_mask][::20, 0], h_pcs_clip[post_t_mask][::20, 1],
                   c=h_times_clip[post_t_mask][::20], cmap='viridis', s=2, alpha=0.4)
        # Overlay committed bouts
        for _, row in committed.iterrows():
            if row['start_time'] > transition_time and row['start_time'] <= MAX_TIME:
                bmask = (h_times_clip >= row['start_time']) & (h_times_clip <= row['end_time'])
                if bmask.sum() > 0:
                    ax.plot(h_pcs_clip[bmask, 0], h_pcs_clip[bmask, 1],
                            color='blue', linewidth=1.5, alpha=0.7)
        # Overlay task bouts
        for _, row in task.iterrows():
            if row['start_time'] > transition_time and row['start_time'] <= MAX_TIME:
                bmask = (h_times_clip >= row['start_time']) & (h_times_clip <= row['end_time'])
                if bmask.sum() > 0:
                    ax.plot(h_pcs_clip[bmask, 0], h_pcs_clip[bmask, 1],
                            color='green', linewidth=1.5, alpha=0.7)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_title(f'Post-transition ({transition_time:.0f}-{MAX_TIME}s)\nBlue=committed, Green=task')

        plt.tight_layout(rect=[0, 0, 1, 0.92])
        plt.savefig(f'figures/hesitant_transition_latent_{region.lower()}.png',
                    dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: figures/hesitant_transition_latent_{region.lower()}.png")

        sys.stdout.flush()

    print("\n" + "=" * 70)
    print("  DONE — Transition analysis complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
