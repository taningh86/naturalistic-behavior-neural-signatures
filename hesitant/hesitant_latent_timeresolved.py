"""
Time-resolved GRU-ODE latent dynamics along trajectories.
Hesitant vs Short-committed vs Task-engaged, Session 1.

For each bout: compute flow speed, direction, curvature, distance to FP,
local divergence at EVERY time point. Then compare temporal profiles.
"""

import yaml
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import spikeinterface.extractors as se
import torch
import torch.nn as nn
from torchdiffeq import odeint
from sklearn.decomposition import PCA
from scipy import stats as sp_stats
from scipy.ndimage import gaussian_filter1d
import warnings, sys

warnings.filterwarnings('ignore')

BIN_SIZE_MS = 10
FS = 30000
D_SHARED = 32
HIDDEN_SIZE = 32
ODE_GATE_HIDDEN = 64
ODE_DT = 1.0
PRED_BINS = 10
LHA_DEPTH_MAX = 1300
RSP_DEPTH_MIN = 1300
TIME_CUTOFF = 750
N_RESAMPLE = 50  # resample all bouts to 50 normalized time points

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
    """Evaluate dh/dt at points. Returns numpy."""
    h = torch.tensor(points, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        dhdt = ode_func(0.0, h).cpu().numpy()
    return dhdt


def compute_local_divergence(ode_func, points, eps=0.01):
    """Estimate divergence (trace of Jacobian) at points via finite differences.
    Positive = expanding flow, Negative = contracting flow."""
    divs = np.zeros(len(points))
    for d in range(HIDDEN_SIZE):
        perturb = np.zeros(HIDDEN_SIZE)
        perturb[d] = eps
        f_plus = evaluate_flow(ode_func, points + perturb)
        f_minus = evaluate_flow(ode_func, points - perturb)
        divs += (f_plus[:, d] - f_minus[:, d]) / (2 * eps)
    return divs


def compute_trajectory_dynamics(ode_func, traj, dom_fp, pca):
    """Compute time-resolved dynamics along a trajectory.
    Returns dict of arrays, one value per time point."""
    n = len(traj)

    # Flow field at each point
    dhdt = evaluate_flow(ode_func, traj)
    flow_speed = np.linalg.norm(dhdt, axis=1)

    # Actual trajectory velocity
    traj_vel = np.diff(traj, axis=0)
    traj_speed = np.linalg.norm(traj_vel, axis=1)
    # Pad to same length
    traj_speed = np.append(traj_speed, traj_speed[-1])

    # Flow-trajectory alignment at each point
    alignment = np.zeros(n)
    for t in range(n - 1):
        fn = np.linalg.norm(dhdt[t])
        vn = np.linalg.norm(traj_vel[t])
        if fn > 1e-8 and vn > 1e-8:
            alignment[t] = np.dot(dhdt[t], traj_vel[t]) / (fn * vn)
    alignment[-1] = alignment[-2] if n > 1 else 0

    # Distance to fixed point
    dist_to_fp = np.linalg.norm(traj - dom_fp, axis=1)

    # Change in distance to FP (positive = moving away, negative = approaching)
    ddist = np.diff(dist_to_fp)
    ddist = np.append(ddist, ddist[-1])

    # Curvature (angle change between consecutive velocities)
    curvature = np.zeros(n)
    for t in range(len(traj_vel) - 1):
        n1 = np.linalg.norm(traj_vel[t])
        n2 = np.linalg.norm(traj_vel[t + 1])
        if n1 > 1e-8 and n2 > 1e-8:
            cos_a = np.clip(np.dot(traj_vel[t], traj_vel[t+1]) / (n1 * n2), -1, 1)
            curvature[t + 1] = np.arccos(cos_a)

    # PC projections
    traj_pca = pca.transform(traj)
    dhdt_pca = dhdt @ pca.components_.T

    # Local divergence (subsample for speed — every 10th point)
    div_indices = np.arange(0, n, 10)
    div_vals = compute_local_divergence(ode_func, traj[div_indices])
    # Interpolate to full resolution
    divergence = np.interp(np.arange(n), div_indices, div_vals)

    # Gate activity: z values (how "frozen" are the dynamics)
    h_tensor = torch.tensor(traj, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        z_gate = ode_func.update_gate(h_tensor).cpu().numpy()
    mean_gate = z_gate.mean(axis=1)  # mean across dimensions

    return {
        'flow_speed': flow_speed,
        'traj_speed': traj_speed,
        'alignment': alignment,
        'dist_to_fp': dist_to_fp,
        'ddist_fp': ddist,
        'curvature': curvature,
        'divergence': divergence,
        'gate_mean': mean_gate,
        'pc1': traj_pca[:, 0],
        'pc2': traj_pca[:, 1],
        'flow_pc1': dhdt_pca[:, 0],
        'flow_pc2': dhdt_pca[:, 1],
    }


def resample_profile(profile, n_out=N_RESAMPLE):
    """Resample a time series to n_out points (normalized time 0-1)."""
    n_in = len(profile)
    if n_in < 2:
        return np.full(n_out, profile[0] if n_in > 0 else np.nan)
    x_in = np.linspace(0, 1, n_in)
    x_out = np.linspace(0, 1, n_out)
    return np.interp(x_out, x_in, profile)


def find_fixed_points(ode_func, hidden_states, n_starts=500, max_steps=5000,
                      speed_thresh=1e-6):
    indices = np.random.choice(len(hidden_states), size=min(n_starts, len(hidden_states)),
                                replace=False)
    starts = hidden_states[indices]
    h = torch.tensor(starts, dtype=torch.float32, device=DEVICE).requires_grad_(True)
    optimizer = torch.optim.Adam([h], lr=0.01)
    for step in range(max_steps):
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
    mask = final_speeds < speed_thresh
    if mask.sum() == 0:
        # Use session mean as fallback
        return hidden_states.mean(axis=0)
    # Return centroid of converged points
    return h_np[mask].mean(axis=0)


def main():
    print("=" * 70)
    print("  Time-Resolved GRU-ODE Dynamics Along Trajectories")
    print("  Session 1 (Fed, Exploration)")
    print("=" * 70)
    sys.stdout.flush()

    # --- Define groups ---
    exc_df = pd.read_csv("data/excursion_features_all_sessions.csv")
    s1 = exc_df[exc_df["session"] == 1].copy()
    not_pot = s1["farthest_zone"] != "Pot"
    all_hes = s1[(s1["feeding_bins"] == 0) & (s1["digging_bins"] == 0) &
                 not_pot & (s1["reversals"] >= 1) & (s1["duration"] >= 2.0)]
    hesitant = all_hes[all_hes["start_time"] < TIME_CUTOFF].copy()
    task = s1[(s1["feeding_bins"] > 0) | (s1["digging_bins"] > 0)].copy()
    non_hes_non_task = s1[~s1.index.isin(all_hes.index) & ~s1.index.isin(task.index)]
    hes_dur_min = hesitant["duration"].min()
    hes_dur_max = hesitant["duration"].max()
    committed = non_hes_non_task[
        (non_hes_non_task["reached_arena"] == True) &
        (non_hes_non_task["duration"] >= hes_dur_min) &
        (non_hes_non_task["duration"] <= hes_dur_max)
    ].copy()

    groups = {"Hesitant": hesitant, "Short-committed": committed, "Task-engaged": task}
    group_names = ["Hesitant", "Short-committed", "Task-engaged"]
    colors = {"Hesitant": "#E53935", "Short-committed": "#1E88E5", "Task-engaged": "#43A047"}

    for gn, gdf in groups.items():
        n = len(gdf)
        if n > 0:
            print(f"  {gn}: {n} bouts, dur={gdf['duration'].median():.1f}s")
        else:
            print(f"  {gn}: 0 bouts")
    sys.stdout.flush()

    # --- Load spike data for alignment ---
    sp = paths_config['single_probe']['coordinates_1']['mouse01']['sessions']
    sorted_path = Path(sp['session_1']['sorted'])
    sorting = se.read_kilosort(sorted_path)
    ci = pd.read_csv(sorted_path / "cluster_info.tsv", sep="\t")
    lc = "group" if "group" in ci.columns and ci["group"].eq("good").any() else "KSLabel"
    good = ci[ci[lc] == "good"]
    lha_ids = good[good["depth"] < LHA_DEPTH_MAX]["cluster_id"].values
    rsp_ids = good[good["depth"] >= RSP_DEPTH_MIN]["cluster_id"].values

    # --- Process each region ---
    for region, unit_ids, h_file in [
            ("LHA", lha_ids, "gru_ode_10ms_hidden_lha_s1.npy"),
            ("RSP", rsp_ids, "gru_ode_10ms_hidden_rsp_s1.npy")]:

        print(f"\n{'='*70}")
        print(f"  {region}")
        print(f"{'='*70}")
        sys.stdout.flush()

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

        # Find dominant fixed point
        print("  Finding fixed point...")
        sys.stdout.flush()
        np.random.seed(42)
        dom_fp = find_fixed_points(ode_func, h_states)
        fp_norm = np.linalg.norm(dom_fp)
        print(f"    ||FP||={fp_norm:.3f}")

        # --- Extract trajectories and compute time-resolved dynamics ---
        print("  Computing time-resolved dynamics...")
        sys.stdout.flush()

        metrics = ['flow_speed', 'traj_speed', 'alignment', 'dist_to_fp',
                   'ddist_fp', 'curvature', 'divergence', 'gate_mean',
                   'pc1', 'pc2', 'flow_pc1', 'flow_pc2']

        # Store resampled profiles per group
        group_profiles = {gn: {m: [] for m in metrics} for gn in group_names}
        # Store individual bout trajectories for plotting
        group_trajs_pca = {gn: [] for gn in group_names}
        group_flow_along = {gn: [] for gn in group_names}

        for gn, gdf in groups.items():
            count = 0
            for _, row in gdf.iterrows():
                gs = max(0, int((row["start_time"] - offset) / 0.01))
                ge = min(h_states.shape[0], int((row["end_time"] - offset) / 0.01))
                if ge - gs < 30:  # at least 300ms
                    continue

                traj = h_states[gs:ge, :]
                dyn = compute_trajectory_dynamics(ode_func, traj, dom_fp, pca)

                for m in metrics:
                    group_profiles[gn][m].append(resample_profile(dyn[m]))

                group_trajs_pca[gn].append(pca.transform(traj))
                group_flow_along[gn].append(
                    np.column_stack([dyn['flow_pc1'], dyn['flow_pc2']]))
                count += 1

            print(f"    {gn}: {count} bouts processed")
        sys.stdout.flush()

        # Convert to arrays
        for gn in group_names:
            for m in metrics:
                if len(group_profiles[gn][m]) > 0:
                    group_profiles[gn][m] = np.array(group_profiles[gn][m])
                else:
                    group_profiles[gn][m] = np.array([]).reshape(0, N_RESAMPLE)

        # =============================================================
        # FIGURE: Time-resolved dynamics
        # =============================================================
        norm_time = np.linspace(0, 100, N_RESAMPLE)  # 0-100% of bout

        fig, axes = plt.subplots(3, 4, figsize=(22, 15))
        fig.suptitle(f'Time-Resolved GRU-ODE Dynamics — {region}\n'
                     f'Session 1: Hesitant (<{TIME_CUTOFF}s) vs Short-committed vs Task-engaged\n'
                     f'X-axis: normalized bout time (0%=start, 100%=end)',
                     fontsize=13, fontweight='bold')

        plot_metrics = [
            ('flow_speed', 'Flow speed |dh/dt|', 'ODE-prescribed speed'),
            ('traj_speed', 'Trajectory speed |dh|', 'Actual movement speed'),
            ('alignment', 'Flow-trajectory alignment', 'cos(dh/dt, actual dh)\n+1=following flow, -1=against'),
            ('dist_to_fp', 'Distance to fixed point', 'Distance to dominant FP'),
            ('ddist_fp', 'dDist/dt to FP', '+moving away, -approaching FP'),
            ('curvature', 'Trajectory curvature', 'Angle change (rad)\n0=straight, pi=reversal'),
            ('divergence', 'Local divergence', 'tr(Jacobian)\n+expanding, -contracting'),
            ('gate_mean', 'Mean gate value (z)', 'z->1: frozen, z->0: active'),
            ('pc1', 'PC1 position', 'Position in latent PC1'),
            ('pc2', 'PC2 position', 'Position in latent PC2'),
            ('flow_pc1', 'Flow in PC1 direction', 'dh/dt projected to PC1'),
            ('flow_pc2', 'Flow in PC2 direction', 'dh/dt projected to PC2'),
        ]

        for idx, (metric, title, ylabel) in enumerate(plot_metrics):
            row = idx // 4
            col = idx % 4
            ax = axes[row, col]

            for gn in group_names:
                profiles = group_profiles[gn][metric]
                if len(profiles) == 0:
                    continue
                mean_prof = np.mean(profiles, axis=0)
                sem_prof = np.std(profiles, axis=0) / np.sqrt(len(profiles))

                ax.plot(norm_time, mean_prof, color=colors[gn], linewidth=2,
                        label=f'{gn} (n={len(profiles)})')
                ax.fill_between(norm_time, mean_prof - sem_prof, mean_prof + sem_prof,
                                color=colors[gn], alpha=0.15)

                # Also show individual traces faintly
                for prof in profiles[::max(1, len(profiles)//5)]:  # show ~5 examples
                    ax.plot(norm_time, prof, color=colors[gn], alpha=0.08, linewidth=0.5)

            ax.set_xlabel('Bout time (%)')
            ax.set_ylabel(ylabel, fontsize=8)
            ax.set_title(title, fontsize=10)
            if idx == 0:
                ax.legend(fontsize=7, loc='best')

            # Add reference lines
            if metric == 'alignment':
                ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
            elif metric == 'ddist_fp':
                ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
            elif metric == 'divergence':
                ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')

        plt.tight_layout(rect=[0, 0, 1, 0.94])
        plt.savefig(f'figures/hesitant_dynamics_timeresolved_{region.lower()}.png',
                    dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: figures/hesitant_dynamics_timeresolved_{region.lower()}.png")

        # =============================================================
        # FIGURE 2: Individual trajectory examples with flow arrows
        # =============================================================
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f'{region}: Example Trajectories with Local Flow Arrows\n'
                     f'Arrows = ODE flow direction at trajectory location',
                     fontsize=13, fontweight='bold')

        for gi, gn in enumerate(group_names):
            ax = axes[gi]
            trajs = group_trajs_pca[gn]
            flows = group_flow_along[gn]

            # Plot up to 8 example trajectories
            n_show = min(8, len(trajs))
            cmap = plt.cm.viridis

            for ti in range(n_show):
                traj_pc = trajs[ti]
                flow_pc = flows[ti]
                n_pts = len(traj_pc)

                # Color by time within bout
                time_colors = cmap(np.linspace(0, 1, n_pts))

                # Trajectory path
                ax.plot(traj_pc[:, 0], traj_pc[:, 1], color=colors[gn],
                        alpha=0.3, linewidth=0.8)

                # Flow arrows at subsampled points
                step = max(1, n_pts // 15)
                for t in range(0, n_pts, step):
                    ax.annotate('', xy=(traj_pc[t, 0] + flow_pc[t, 0] * 0.3,
                                        traj_pc[t, 1] + flow_pc[t, 1] * 0.3),
                                xytext=(traj_pc[t, 0], traj_pc[t, 1]),
                                arrowprops=dict(arrowstyle='->', color=time_colors[t],
                                               lw=0.8, mutation_scale=8))

                # Mark start and end
                ax.scatter(traj_pc[0, 0], traj_pc[0, 1], c='green', s=30,
                          zorder=5, edgecolors='black', linewidths=0.5, marker='o')
                ax.scatter(traj_pc[-1, 0], traj_pc[-1, 1], c='red', s=30,
                          zorder=5, edgecolors='black', linewidths=0.5, marker='s')

            # Mark fixed point
            fp_pca = pca.transform(dom_fp.reshape(1, -1))
            ax.scatter(fp_pca[0, 0], fp_pca[0, 1], marker='x', s=150,
                      c='black', linewidths=3, zorder=10)

            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
            ax.set_title(f'{gn} (n={n_show} shown)\ngreen=start, red=end')
            ax.set_xlim(-3.5, 3.5)
            ax.set_ylim(-3.5, 3.5)

        plt.tight_layout()
        plt.savefig(f'figures/hesitant_trajectory_examples_{region.lower()}.png',
                    dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: figures/hesitant_trajectory_examples_{region.lower()}.png")

        # =============================================================
        # Print key time-resolved stats
        # =============================================================
        print(f"\n  Time-resolved profile comparisons (pointwise t-test at 25%, 50%, 75%):")
        key_metrics = ['flow_speed', 'alignment', 'dist_to_fp', 'divergence', 'gate_mean']
        time_points = [12, 25, 37]  # indices for 25%, 50%, 75%
        time_labels = ['25%', '50%', '75%']

        for m in key_metrics:
            h_prof = group_profiles["Hesitant"][m]
            c_prof = group_profiles["Short-committed"][m]
            if len(h_prof) < 2 or len(c_prof) < 2:
                continue
            print(f"\n    {m}:")
            for ti, tl in zip(time_points, time_labels):
                hv = h_prof[:, ti]
                cv = c_prof[:, ti]
                stat, pval = sp_stats.mannwhitneyu(hv, cv, alternative='two-sided')
                sig = '*' if pval < 0.05 else 'ns'
                print(f"      t={tl}: Hes={np.median(hv):.4f}, Com={np.median(cv):.4f}, "
                      f"p={pval:.4f} {sig}")

        sys.stdout.flush()

    print("\nDone!")


if __name__ == "__main__":
    main()
