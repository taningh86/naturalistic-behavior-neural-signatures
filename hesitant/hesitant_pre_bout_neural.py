"""
Pre-bout neural signatures: 5 seconds immediately before each excursion.
Compare pre-hesitant vs pre-committed vs pre-task-engaged.
Session 1 only. Early hesitant (<800s).

The animal is in Home during these 5s windows, so any difference
cannot be explained by location/locomotion — it reflects internal state.
"""

import yaml
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import spikeinterface.extractors as se
import torch
import torch.nn as nn
from torchdiffeq import odeint
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats as sp_stats
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
PRE_WINDOW = 5.0  # seconds before bout start
N_RESAMPLE = 50   # resample pre-window to 50 points

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


def find_fixed_point(ode_func, hidden_states):
    np.random.seed(42)
    indices = np.random.choice(len(hidden_states), size=min(500, len(hidden_states)),
                                replace=False)
    starts = hidden_states[indices]
    h = torch.tensor(starts, dtype=torch.float32, device=DEVICE).requires_grad_(True)
    optimizer = torch.optim.Adam([h], lr=0.01)
    for _ in range(5000):
        optimizer.zero_grad()
        with torch.enable_grad():
            dhdt = ode_func(0, h)
            loss = (dhdt ** 2).sum(dim=1).sum()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        dhdt_final = ode_func(0, h)
        final_speeds = (dhdt_final ** 2).sum(dim=1).cpu().numpy()
    h_np = h.detach().cpu().numpy()
    mask = final_speeds < 1e-6
    if mask.sum() == 0:
        return hidden_states.mean(axis=0)
    return h_np[mask].mean(axis=0)


def resample(profile, n_out=N_RESAMPLE):
    n_in = len(profile)
    if n_in < 2:
        return np.full(n_out, profile[0] if n_in > 0 else np.nan)
    return np.interp(np.linspace(0, 1, n_out), np.linspace(0, 1, n_in), profile)


def compute_pre_bout_dynamics(ode_func, traj, dom_fp):
    """Time-resolved dynamics for a pre-bout window."""
    n = len(traj)
    dhdt = evaluate_flow(ode_func, traj)
    flow_speed = np.linalg.norm(dhdt, axis=1)

    traj_vel = np.diff(traj, axis=0)
    traj_speed = np.linalg.norm(traj_vel, axis=1)
    traj_speed = np.append(traj_speed, traj_speed[-1])

    alignment = np.zeros(n)
    for t in range(n - 1):
        fn = np.linalg.norm(dhdt[t])
        vn = np.linalg.norm(traj_vel[t])
        if fn > 1e-8 and vn > 1e-8:
            alignment[t] = np.dot(dhdt[t], traj_vel[t]) / (fn * vn)
    alignment[-1] = alignment[-2] if n > 1 else 0

    dist_to_fp = np.linalg.norm(traj - dom_fp, axis=1)
    ddist = np.diff(dist_to_fp)
    ddist = np.append(ddist, ddist[-1])

    div_indices = np.arange(0, n, max(1, n // 10))
    div_vals = compute_local_divergence(ode_func, traj[div_indices])
    divergence = np.interp(np.arange(n), div_indices, div_vals)

    h_tensor = torch.tensor(traj, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        z_gate = ode_func.update_gate(h_tensor).cpu().numpy()
    mean_gate = z_gate.mean(axis=1)

    return {
        'flow_speed': flow_speed,
        'traj_speed': traj_speed,
        'alignment': alignment,
        'dist_to_fp': dist_to_fp,
        'ddist_fp': ddist,
        'divergence': divergence,
        'gate_mean': mean_gate,
    }


def main():
    print("=" * 70)
    print(f"  Pre-Bout Neural Signatures ({PRE_WINDOW}s before excursion start)")
    print(f"  Session 1 (Fed, Exploration)")
    print("=" * 70)
    sys.stdout.flush()

    # --- Define groups ---
    exc_df = pd.read_csv("data/excursion_features_all_sessions.csv")
    s1 = exc_df[exc_df["session"] == 1].copy()
    not_pot = s1["farthest_zone"] != "Pot"
    all_hes = s1[(s1["feeding_bins"] == 0) & (s1["digging_bins"] == 0) &
                 not_pot & (s1["reversals"] >= 1) & (s1["duration"] >= 2.0)]
    hesitant = all_hes[all_hes["start_time"] < TIME_CUTOFF]
    task = s1[(s1["feeding_bins"] > 0) | (s1["digging_bins"] > 0)]
    non_hes_non_task = s1[~s1.index.isin(all_hes.index) & ~s1.index.isin(task.index)]
    hes_dur_min = hesitant["duration"].min()
    hes_dur_max = hesitant["duration"].max()
    committed = non_hes_non_task[
        (non_hes_non_task["reached_arena"] == True) &
        (non_hes_non_task["duration"] >= hes_dur_min) &
        (non_hes_non_task["duration"] <= hes_dur_max)
    ]

    groups = {"Pre-hesitant": hesitant, "Pre-committed": committed, "Pre-task": task}
    group_names = ["Pre-hesitant", "Pre-committed", "Pre-task"]
    colors = {"Pre-hesitant": "#E53935", "Pre-committed": "#1E88E5", "Pre-task": "#43A047"}

    for gn, gdf in groups.items():
        n = len(gdf)
        print(f"  {gn}: {n} bouts")
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

    # --- Also load raw firing rates at 100ms for population FR ---
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

    metrics = ['flow_speed', 'traj_speed', 'alignment', 'dist_to_fp',
               'ddist_fp', 'divergence', 'gate_mean']

    for region, unit_ids, h_file, fr_data, fr_amin in [
            ("LHA", lha_ids, "gru_ode_10ms_hidden_lha_s1.npy", lha_fr, lha_fr_amin),
            ("RSP", rsp_ids, "gru_ode_10ms_hidden_rsp_s1.npy", rsp_fr, rsp_fr_amin)]:

        print(f"\n{'='*70}")
        print(f"  {region}")
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
        dom_fp = find_fixed_point(ode_func, h_states)

        # --- Extract pre-bout windows ---
        group_profiles = {gn: {m: [] for m in metrics} for gn in group_names}
        group_fr_profiles = {gn: [] for gn in group_names}
        group_latent_profiles = {gn: [] for gn in group_names}  # raw 32D
        group_pc_profiles = {gn: [] for gn in group_names}
        group_times = {gn: [] for gn in group_names}

        fr_offset = fr_amin / FS

        for gn, gdf in groups.items():
            count = 0
            for _, row in gdf.iterrows():
                bout_start = row["start_time"]
                pre_start = bout_start - PRE_WINDOW
                if pre_start < 1.0:  # need at least 1s from session start
                    continue

                # GRU-ODE hidden states for pre-window
                gs = max(0, int((pre_start - offset) / 0.01))
                ge = max(0, int((bout_start - offset) / 0.01))
                if ge - gs < 20:
                    continue

                traj = h_states[gs:ge, :]
                dyn = compute_pre_bout_dynamics(ode_func, traj, dom_fp)
                for m in metrics:
                    group_profiles[gn][m].append(resample(dyn[m]))

                # PC trajectory
                group_pc_profiles[gn].append(resample(pca.transform(traj)[:, 0]))
                group_latent_profiles[gn].append(traj.mean(axis=0))  # mean 32D vector

                # Firing rate for pre-window (100ms bins)
                fr_bs = max(0, int((pre_start - fr_offset) / 0.1))
                fr_be = max(0, int((bout_start - fr_offset) / 0.1))
                if fr_be - fr_bs >= 2:
                    fr_slice = fr_data[fr_bs:fr_be, :].mean(axis=1)  # pop mean per bin
                    group_fr_profiles[gn].append(resample(fr_slice))

                group_times[gn].append(bout_start)
                count += 1

            print(f"  {gn}: {count} valid pre-windows")

        # Convert
        for gn in group_names:
            for m in metrics:
                arr = group_profiles[gn][m]
                group_profiles[gn][m] = np.array(arr) if len(arr) > 0 else np.array([]).reshape(0, N_RESAMPLE)
            group_fr_profiles[gn] = np.array(group_fr_profiles[gn]) if len(group_fr_profiles[gn]) > 0 else np.array([]).reshape(0, N_RESAMPLE)
            group_latent_profiles[gn] = np.array(group_latent_profiles[gn]) if len(group_latent_profiles[gn]) > 0 else np.array([]).reshape(0, HIDDEN_SIZE)
            group_pc_profiles[gn] = np.array(group_pc_profiles[gn]) if len(group_pc_profiles[gn]) > 0 else np.array([]).reshape(0, N_RESAMPLE)
            group_times[gn] = np.array(group_times[gn])

        # =============================================================
        # FIGURE: Time-resolved pre-bout dynamics
        # =============================================================
        norm_time = np.linspace(-PRE_WINDOW, 0, N_RESAMPLE)  # -5s to 0s

        fig, axes = plt.subplots(2, 5, figsize=(25, 10))
        fig.suptitle(f'Pre-Bout Neural State ({PRE_WINDOW}s Before Excursion Start) — {region}\n'
                     f'Session 1: Animal is in Home during these windows\n'
                     f'x-axis: time relative to excursion start (0 = leave Home)',
                     fontsize=13, fontweight='bold')

        plot_items = [
            ('flow_speed', 'Flow speed |dh/dt|'),
            ('traj_speed', 'Trajectory speed'),
            ('alignment', 'Flow-traj alignment'),
            ('dist_to_fp', 'Distance to FP'),
            ('divergence', 'Local divergence'),
            ('gate_mean', 'Gate value (z)'),
            ('ddist_fp', 'dDist/dt to FP'),
            ('fr', 'Population FR (Hz)'),
            ('pc1', 'PC1 position'),
            ('stats', 'Statistics'),
        ]

        for idx, (key, title) in enumerate(plot_items):
            row = idx // 5
            col = idx % 5
            ax = axes[row, col]

            if key == 'stats':
                # Statistical summary
                text_lines = [f'{region} — Pre-bout Stats\n']
                for m in metrics:
                    hp = group_profiles["Pre-hesitant"][m]
                    cp = group_profiles["Pre-committed"][m]
                    if len(hp) >= 2 and len(cp) >= 2:
                        # Test at -4s, -2.5s, -0.5s (near start)
                        for ti, tl in [(5, '-4.0s'), (25, '-2.5s'), (45, '-0.5s')]:
                            hv = hp[:, ti]
                            cv = cp[:, ti]
                            stat, pval = sp_stats.mannwhitneyu(hv, cv, alternative='two-sided')
                            sig = '*' if pval < 0.05 else ''
                            if pval < 0.1:
                                text_lines.append(
                                    f'{m[:12]:12s} {tl}: '
                                    f'H={np.median(hv):.3f} C={np.median(cv):.3f} '
                                    f'p={pval:.3f}{sig}')
                ax.text(0.02, 0.95, '\n'.join(text_lines), transform=ax.transAxes,
                        va='top', fontsize=7, fontfamily='monospace')
                ax.axis('off')
                continue

            if key == 'fr':
                # Firing rate
                for gn in group_names:
                    profiles = group_fr_profiles[gn]
                    if len(profiles) == 0:
                        continue
                    mean_p = np.mean(profiles, axis=0)
                    sem_p = np.std(profiles, axis=0) / np.sqrt(len(profiles))
                    ax.plot(norm_time, mean_p, color=colors[gn], linewidth=2,
                            label=f'{gn} ({len(profiles)})')
                    ax.fill_between(norm_time, mean_p - sem_p, mean_p + sem_p,
                                    color=colors[gn], alpha=0.15)
                ax.axvline(0, color='black', linewidth=0.5, linestyle='--', alpha=0.5)
                ax.set_xlabel('Time to bout start (s)')
                ax.set_ylabel('Pop FR (Hz)')
                ax.set_title(title)
                ax.legend(fontsize=7)
                continue

            if key == 'pc1':
                for gn in group_names:
                    profiles = group_pc_profiles[gn]
                    if len(profiles) == 0:
                        continue
                    mean_p = np.mean(profiles, axis=0)
                    sem_p = np.std(profiles, axis=0) / np.sqrt(len(profiles))
                    ax.plot(norm_time, mean_p, color=colors[gn], linewidth=2,
                            label=f'{gn} ({len(profiles)})')
                    ax.fill_between(norm_time, mean_p - sem_p, mean_p + sem_p,
                                    color=colors[gn], alpha=0.15)
                ax.axvline(0, color='black', linewidth=0.5, linestyle='--', alpha=0.5)
                ax.set_xlabel('Time to bout start (s)')
                ax.set_ylabel('PC1')
                ax.set_title(title)
                continue

            # GRU-ODE metrics
            for gn in group_names:
                profiles = group_profiles[gn][key]
                if len(profiles) == 0:
                    continue
                mean_p = np.mean(profiles, axis=0)
                sem_p = np.std(profiles, axis=0) / np.sqrt(len(profiles))
                ax.plot(norm_time, mean_p, color=colors[gn], linewidth=2,
                        label=f'{gn} ({len(profiles)})')
                ax.fill_between(norm_time, mean_p - sem_p, mean_p + sem_p,
                                color=colors[gn], alpha=0.15)
                # Show individual traces
                for prof in profiles[::max(1, len(profiles)//3)]:
                    ax.plot(norm_time, prof, color=colors[gn], alpha=0.06, linewidth=0.5)

            ax.axvline(0, color='black', linewidth=0.5, linestyle='--', alpha=0.5)
            if key in ['alignment', 'ddist_fp', 'divergence']:
                ax.axhline(0, color='gray', linewidth=0.3, linestyle='--')
            ax.set_xlabel('Time to bout start (s)')
            ax.set_title(title)
            if idx == 0:
                ax.legend(fontsize=7)

        plt.tight_layout(rect=[0, 0, 1, 0.94])
        plt.savefig(f'figures/hesitant_pre_bout_{region.lower()}.png',
                    dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: figures/hesitant_pre_bout_{region.lower()}.png")

        # =============================================================
        # Print full stats table
        # =============================================================
        print(f"\n  Pre-bout statistics (Pre-hesitant vs Pre-committed):")
        print(f"  {'Metric':<20} {'Time':<8} {'Hes med':<10} {'Com med':<10} {'p':<10} {'sig'}")
        print(f"  {'-'*68}")

        for m in metrics + ['fr', 'pc1']:
            if m == 'fr':
                hp = group_fr_profiles["Pre-hesitant"]
                cp = group_fr_profiles["Pre-committed"]
            elif m == 'pc1':
                hp = group_pc_profiles["Pre-hesitant"]
                cp = group_pc_profiles["Pre-committed"]
            else:
                hp = group_profiles["Pre-hesitant"][m]
                cp = group_profiles["Pre-committed"][m]

            if len(hp) < 2 or len(cp) < 2:
                continue

            for ti, tl in [(5, '-4.0s'), (15, '-3.0s'), (25, '-2.5s'),
                           (35, '-1.5s'), (45, '-0.5s')]:
                hv = hp[:, ti]
                cv = cp[:, ti]
                stat, pval = sp_stats.mannwhitneyu(hv, cv, alternative='two-sided')
                sig = '*' if pval < 0.05 else 'ns'
                print(f"  {m:<20} {tl:<8} {np.median(hv):<10.4f} {np.median(cv):<10.4f} "
                      f"{pval:<10.4f} {sig}")

        # =============================================================
        # Pre-bout latent centroid comparison
        # =============================================================
        print(f"\n  Pre-bout mean latent state (32D centroid):")
        hp_lat = group_latent_profiles["Pre-hesitant"]
        cp_lat = group_latent_profiles["Pre-committed"]
        if len(hp_lat) >= 2 and len(cp_lat) >= 2:
            # Cosine similarity within and between
            h_cos = cosine_similarity(hp_lat)
            c_cos = cosine_similarity(cp_lat)
            x_cos = cosine_similarity(hp_lat, cp_lat)
            h_ut = h_cos[np.triu_indices(len(hp_lat), k=1)]
            c_ut = c_cos[np.triu_indices(len(cp_lat), k=1)]
            print(f"    Within pre-hes cosine:  {np.mean(h_ut):.3f} +/- {np.std(h_ut):.3f}")
            print(f"    Within pre-com cosine:  {np.mean(c_ut):.3f} +/- {np.std(c_ut):.3f}")
            print(f"    Cross hes-com cosine:   {np.mean(x_cos):.3f} +/- {np.std(x_cos):.3f}")

            # Euclidean distance between group centroids
            h_cent = hp_lat.mean(axis=0)
            c_cent = cp_lat.mean(axis=0)
            dist = np.linalg.norm(h_cent - c_cent)
            h_spread = np.mean(np.linalg.norm(hp_lat - h_cent, axis=1))
            c_spread = np.mean(np.linalg.norm(cp_lat - c_cent, axis=1))
            print(f"    Centroid distance:      {dist:.4f}")
            print(f"    Pre-hes spread:         {h_spread:.4f}")
            print(f"    Pre-com spread:         {c_spread:.4f}")
            print(f"    Separation ratio:       {dist / ((h_spread + c_spread) / 2):.3f}")

        sys.stdout.flush()

    print("\nDone!")


if __name__ == "__main__":
    main()
