"""
GRU-ODE latent dynamics comparison: hesitant vs short-committed vs task-engaged.
Session 1 (Fed, Exploration).

Compares flow field properties, distance to fixed points, trajectory speed,
and attractor proximity across three excursion types.

Groups:
  A) Hesitant (early <750s): no feed/dig, farthest != Pot, reversals >= 1, dur >= 2s
  B) Short committed: reaches arena, no feed/dig, duration in hesitant range
  C) Task-engaged: feeding or digging
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
from scipy.ndimage import gaussian_filter
import warnings, sys

warnings.filterwarnings('ignore')

# Config — must match 10ms Poisson GRU-ODE training
BIN_SIZE_MS = 10
FS = 30000
BIN_SAMPLES = int(BIN_SIZE_MS * FS / 1000)
D_SHARED = 32
HIDDEN_SIZE = 32
ODE_GATE_HIDDEN = 64
ODE_SOLVER = 'rk4'
ODE_STEP_SIZE = 1.0
ODE_DT = 1.0
PRED_BINS = 10
LHA_DEPTH_MAX = 1300
RSP_DEPTH_MIN = 1300
TIME_CUTOFF = 750  # early hesitant cutoff

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open("paths.yaml") as f:
    paths_config = yaml.safe_load(f)


# =============================================================================
# MODEL CLASSES (must match training)
# =============================================================================
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


# =============================================================================
# HELPERS
# =============================================================================

def load_model(region):
    """Load 10ms Poisson GRU-ODE model for a region."""
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


def get_allmin(sorting, unit_ids):
    """Get earliest spike time for time alignment."""
    amin = np.inf
    for u in unit_ids:
        st = sorting.get_unit_spike_train(u)
        if len(st) > 0:
            amin = min(amin, np.min(st))
    return amin


def time_to_gru_bin(t_sec, allmin):
    """Convert behavior time to GRU-ODE bin index."""
    offset = allmin / FS
    return int((t_sec - offset) / 0.01)


def evaluate_flow_at_points(ode_func, points):
    """Evaluate dh/dt at given hidden state points. Returns numpy array."""
    h = torch.tensor(points, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        dhdt = ode_func(0.0, h).cpu().numpy()
    return dhdt


def find_fixed_points(ode_func, hidden_states, n_starts=500, lr=0.01,
                      max_steps=5000, speed_thresh=1e-6):
    """Find fixed points by minimizing ||f(h)||^2."""
    indices = np.random.choice(len(hidden_states), size=min(n_starts, len(hidden_states)),
                                replace=False)
    starts = hidden_states[indices]
    h = torch.tensor(starts, dtype=torch.float32, device=DEVICE).requires_grad_(True)
    optimizer = torch.optim.Adam([h], lr=lr)

    for step in range(max_steps):
        optimizer.zero_grad()
        with torch.enable_grad():
            dhdt = ode_func(0, h)
            speed_sq = (dhdt ** 2).sum(dim=1)
            loss = speed_sq.sum()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        dhdt_final = ode_func(0, h)
        final_speeds = (dhdt_final ** 2).sum(dim=1).cpu().numpy()

    h_np = h.detach().cpu().numpy()
    mask = final_speeds < speed_thresh
    return h_np[mask], final_speeds[mask]


def cluster_fps(fps, eps=0.5):
    """Cluster nearby fixed points, return centroids."""
    if len(fps) == 0:
        return np.array([]).reshape(0, HIDDEN_SIZE)
    from sklearn.cluster import DBSCAN
    clustering = DBSCAN(eps=eps, min_samples=1).fit(fps)
    centroids = []
    for label in sorted(set(clustering.labels_)):
        mask = clustering.labels_ == label
        centroids.append(fps[mask].mean(axis=0))
    return np.array(centroids)


def compute_jacobian(ode_func, fp):
    """Compute Jacobian at a fixed point."""
    h = torch.tensor(fp, dtype=torch.float32, device=DEVICE).unsqueeze(0).requires_grad_(True)
    dhdt = ode_func(0, h)
    jac = torch.zeros(HIDDEN_SIZE, HIDDEN_SIZE, device=DEVICE)
    for i in range(HIDDEN_SIZE):
        if h.grad is not None:
            h.grad.zero_()
        dhdt[0, i].backward(retain_graph=True)
        jac[i] = h.grad[0].clone()
    return jac.cpu().numpy()


def analyze_stability(jac):
    """Classify fixed point stability from Jacobian eigenvalues."""
    evals = np.linalg.eigvals(jac)
    real_parts = evals.real
    max_real = np.max(real_parts)
    n_pos = np.sum(real_parts > 1e-6)
    n_neg = np.sum(real_parts < -1e-6)
    if max_real < -1e-6:
        stability = "stable"
    elif n_pos > 0 and n_neg > 0:
        stability = "saddle"
    elif max_real > 1e-6:
        stability = "unstable"
    else:
        stability = "marginal"
    return stability, evals


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("  GRU-ODE Latent Dynamics: Hesitant vs Committed vs Task-engaged")
    print("  Session 1 (Fed, Exploration)")
    print("=" * 70)
    sys.stdout.flush()

    # --- Define excursion groups ---
    exc_df = pd.read_csv("data/excursion_features_all_sessions.csv")
    s1 = exc_df[exc_df["session"] == 1].copy()

    not_pot = s1["farthest_zone"] != "Pot"
    all_hes = s1[(s1["feeding_bins"] == 0) & (s1["digging_bins"] == 0) &
                 not_pot & (s1["reversals"] >= 1) & (s1["duration"] >= 2.0)]

    # A) Early hesitant
    hesitant = all_hes[all_hes["start_time"] < TIME_CUTOFF].copy()

    # C) Task-engaged
    task = s1[(s1["feeding_bins"] > 0) | (s1["digging_bins"] > 0)].copy()

    # B) Short committed: not hesitant, not task, reaches arena,
    #    duration in hesitant range (>= hes min, <= hes max)
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

    print(f"\n  Groups:")
    for gn, gdf in groups.items():
        if len(gdf) > 0:
            print(f"    {gn}: {len(gdf)} bouts, dur={gdf['duration'].median():.1f}s "
                  f"({gdf['duration'].min():.1f}-{gdf['duration'].max():.1f}), "
                  f"t={gdf['start_time'].min():.0f}-{gdf['start_time'].max():.0f}s")
        else:
            print(f"    {gn}: 0 bouts")
    sys.stdout.flush()

    # --- Load spike data for time alignment ---
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

        allmin = get_allmin(sorting, unit_ids)
        h_states = np.load(f"data/{h_file}")
        model = load_model(region.lower())
        ode_func = model.ode_func

        # --- Find fixed points ---
        print(f"  Finding fixed points...")
        sys.stdout.flush()
        np.random.seed(42)
        fps_raw, fps_speeds = find_fixed_points(ode_func, h_states,
                                                  n_starts=500, max_steps=5000)
        fps = cluster_fps(fps_raw, eps=0.5)
        print(f"    Found {len(fps_raw)} raw -> {len(fps)} unique fixed points")

        # Stability analysis
        for fi, fp in enumerate(fps):
            jac = compute_jacobian(ode_func, fp)
            stab, evals = analyze_stability(jac)
            max_real = np.max(evals.real)
            print(f"    FP{fi}: ||fp||={np.linalg.norm(fp):.3f}, "
                  f"stability={stab}, max_eig_real={max_real:.2e}")

        # Dominant fixed point (largest cluster or first)
        dom_fp = fps[0] if len(fps) > 0 else h_states.mean(axis=0)

        # --- Fit PCA ---
        pca = PCA(n_components=3)
        pca.fit(h_states)

        # --- Extract trajectories and compute per-bout dynamics ---
        print(f"\n  Computing trajectory dynamics...")
        sys.stdout.flush()

        all_results = []
        trajs_by_group = {}

        for gn, gdf in groups.items():
            trajs = []
            for _, row in gdf.iterrows():
                gs = time_to_gru_bin(row["start_time"], allmin)
                ge = time_to_gru_bin(row["end_time"], allmin)
                gs = max(0, gs)
                ge = min(h_states.shape[0], ge)
                if ge - gs < 20:  # at least 200ms
                    continue

                traj = h_states[gs:ge, :]
                trajs.append(traj)

                # --- Flow field metrics along trajectory ---
                dhdt = evaluate_flow_at_points(ode_func, traj)
                flow_speed = np.linalg.norm(dhdt, axis=1)

                # Actual trajectory velocity
                traj_vel = np.diff(traj, axis=0)
                traj_speed = np.linalg.norm(traj_vel, axis=1)

                # Alignment: cosine between flow direction and actual movement
                alignments = []
                for t in range(len(traj_vel)):
                    f_norm = np.linalg.norm(dhdt[t])
                    v_norm = np.linalg.norm(traj_vel[t])
                    if f_norm > 1e-8 and v_norm > 1e-8:
                        cos_align = np.dot(dhdt[t], traj_vel[t]) / (f_norm * v_norm)
                        alignments.append(cos_align)
                alignment = np.mean(alignments) if alignments else np.nan

                # Distance to dominant fixed point
                dist_to_fp = np.linalg.norm(traj - dom_fp, axis=1)

                # Time in slow-flow region (||dh/dt|| < median session flow speed)
                session_flow = evaluate_flow_at_points(ode_func,
                    h_states[::100])  # subsample for speed
                median_flow = np.median(np.linalg.norm(session_flow, axis=1))
                pct_slow = np.mean(flow_speed < median_flow) * 100

                # Trajectory curvature (angle change between consecutive velocity vectors)
                if len(traj_vel) > 1:
                    angles = []
                    for t in range(len(traj_vel) - 1):
                        n1 = np.linalg.norm(traj_vel[t])
                        n2 = np.linalg.norm(traj_vel[t + 1])
                        if n1 > 1e-8 and n2 > 1e-8:
                            cos_a = np.clip(np.dot(traj_vel[t], traj_vel[t+1]) / (n1*n2), -1, 1)
                            angles.append(np.arccos(cos_a))
                    mean_curvature = np.mean(angles) if angles else np.nan
                else:
                    mean_curvature = np.nan

                all_results.append({
                    'group': gn,
                    'start_time': row['start_time'],
                    'duration': row['duration'],
                    'flow_speed_mean': np.mean(flow_speed),
                    'flow_speed_std': np.std(flow_speed),
                    'traj_speed_mean': np.mean(traj_speed),
                    'alignment': alignment,
                    'dist_to_fp_mean': np.mean(dist_to_fp),
                    'dist_to_fp_min': np.min(dist_to_fp),
                    'dist_to_fp_start': dist_to_fp[0],
                    'dist_to_fp_end': dist_to_fp[-1],
                    'pct_slow_flow': pct_slow,
                    'mean_curvature': mean_curvature,
                })

            trajs_by_group[gn] = trajs

        results_df = pd.DataFrame(all_results)

        # --- Print summary ---
        print(f"\n  {'Metric':<25} ", end="")
        for gn in group_names:
            print(f"{gn:<20}", end="")
        print()
        print(f"  {'-'*85}")

        metrics = ['flow_speed_mean', 'traj_speed_mean', 'alignment',
                   'dist_to_fp_mean', 'dist_to_fp_min', 'pct_slow_flow', 'mean_curvature']
        for m in metrics:
            print(f"  {m:<25} ", end="")
            for gn in group_names:
                vals = results_df[results_df['group'] == gn][m].dropna()
                if len(vals) > 0:
                    print(f"{vals.median():>8.4f} (n={len(vals):<3})", end="  ")
                else:
                    print(f"{'---':>8} {'':>8}", end="  ")
            print()

        # --- Statistical tests ---
        print(f"\n  Mann-Whitney U tests (Hesitant vs Short-committed):")
        hes_r = results_df[results_df['group'] == 'Hesitant']
        com_r = results_df[results_df['group'] == 'Short-committed']
        for m in metrics:
            v1 = hes_r[m].dropna().values
            v2 = com_r[m].dropna().values
            if len(v1) > 1 and len(v2) > 1:
                stat, pval = sp_stats.mannwhitneyu(v1, v2, alternative='two-sided')
                sig = '*' if pval < 0.05 else 'ns'
                print(f"    {m:<25}: med={np.median(v1):.4f} vs {np.median(v2):.4f}, "
                      f"p={pval:.4f} {sig}")
        sys.stdout.flush()

        # =============================================================
        # FIGURE
        # =============================================================
        fig, axes = plt.subplots(2, 4, figsize=(22, 11))
        fig.suptitle(f'GRU-ODE Latent Dynamics — {region}\n'
                     f'Session 1: Hesitant (<{TIME_CUTOFF}s) vs Short-committed vs Task-engaged',
                     fontsize=13, fontweight='bold')

        # --- Row 1: Flow field + trajectories ---

        # Panel 1: Flow field with trajectories
        ax = axes[0, 0]
        # Evaluate flow field on grid
        pc_range = 3.5
        grid = np.linspace(-pc_range, pc_range, 30)
        PC1g, PC2g = np.meshgrid(grid, grid)
        pts_pca = np.zeros((len(grid)**2, pca.n_components_))
        pts_pca[:, 0] = PC1g.ravel()
        pts_pca[:, 1] = PC2g.ravel()
        pts_full = pca.inverse_transform(pts_pca)
        dhdt_grid = evaluate_flow_at_points(ode_func, pts_full)
        dhdt_pca = dhdt_grid @ pca.components_.T
        U = dhdt_pca[:, 0].reshape(len(grid), len(grid))
        V = dhdt_pca[:, 1].reshape(len(grid), len(grid))
        speed_grid = np.sqrt(U**2 + V**2)

        ax.streamplot(grid, grid, U, V, color=speed_grid, cmap='Greys',
                      linewidth=0.6, arrowsize=0.8, density=1.2)
        for gn in group_names:
            for traj in trajs_by_group.get(gn, []):
                proj = pca.transform(traj)
                ax.plot(proj[:, 0], proj[:, 1], color=colors[gn], alpha=0.3, linewidth=0.5)
        if len(fps) > 0:
            fp_pca = pca.transform(fps)
            ax.scatter(fp_pca[:, 0], fp_pca[:, 1], marker='x', s=100,
                      c='black', linewidths=2, zorder=10)
        ax.set_xlim(-pc_range, pc_range)
        ax.set_ylim(-pc_range, pc_range)
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
        ax.set_title('Flow field + trajectories')
        for gn in group_names:
            ax.plot([], [], color=colors[gn], linewidth=2, label=gn)
        ax.legend(fontsize=7, loc='lower right')

        # Panel 2: Flow speed heatmap + slow regions
        ax = axes[0, 1]
        im = ax.pcolormesh(PC1g, PC2g, speed_grid, cmap='magma_r', shading='auto')
        plt.colorbar(im, ax=ax, label='|dh/dt|', shrink=0.8)
        slow_thresh = np.percentile(speed_grid, 15)
        ax.contour(PC1g, PC2g, speed_grid, levels=[slow_thresh],
                   colors='cyan', linewidths=1.5, linestyles='--')
        if len(fps) > 0:
            ax.scatter(fp_pca[:, 0], fp_pca[:, 1], marker='x', s=100,
                      c='white', linewidths=2, zorder=10)
        # Mark bout centroids
        for gn in group_names:
            for traj in trajs_by_group.get(gn, []):
                cent = pca.transform(traj.mean(axis=0).reshape(1, -1))
                ax.scatter(cent[0, 0], cent[0, 1], c=colors[gn], s=15,
                          alpha=0.6, edgecolors='white', linewidths=0.3)
        ax.set_xlim(-pc_range, pc_range)
        ax.set_ylim(-pc_range, pc_range)
        ax.set_xlabel(f'PC1')
        ax.set_ylabel(f'PC2')
        ax.set_title('Flow speed + slow regions (cyan)')

        # Panel 3: Dwell time heatmaps per group
        ax = axes[0, 2]
        for gn in group_names:
            all_pts = []
            for traj in trajs_by_group.get(gn, []):
                proj = pca.transform(traj)
                all_pts.append(proj[:, :2])
            if len(all_pts) > 0:
                all_pts = np.concatenate(all_pts, axis=0)
                # Compute 2D density
                h_map, xe, ye = np.histogram2d(all_pts[:, 0], all_pts[:, 1],
                                                bins=40, range=[[-pc_range, pc_range],
                                                                [-pc_range, pc_range]])
                h_map = gaussian_filter(h_map.T, sigma=1.5)
                # Normalize
                h_map = h_map / h_map.max() if h_map.max() > 0 else h_map
                # Contour at 50% and 80% density
                ax.contour(np.linspace(-pc_range, pc_range, 40),
                          np.linspace(-pc_range, pc_range, 40),
                          h_map, levels=[0.3, 0.6], colors=[colors[gn]],
                          linewidths=[1, 2], alpha=0.8)
        if len(fps) > 0:
            ax.scatter(fp_pca[:, 0], fp_pca[:, 1], marker='x', s=100,
                      c='black', linewidths=2, zorder=10)
        ax.set_xlim(-pc_range, pc_range)
        ax.set_ylim(-pc_range, pc_range)
        ax.set_xlabel(f'PC1')
        ax.set_ylabel(f'PC2')
        ax.set_title('Dwell density contours (30%, 60%)')
        for gn in group_names:
            ax.plot([], [], color=colors[gn], linewidth=2, label=gn)
        ax.legend(fontsize=7)

        # Panel 4: Autonomous trajectory simulation from bout starts
        ax = axes[0, 3]
        # For each group, take a few bout start points and simulate autonomous ODE
        n_sim = min(5, min(len(trajs_by_group.get(gn, [])) for gn in group_names
                          if gn in trajs_by_group))
        for gn in group_names:
            trajs = trajs_by_group.get(gn, [])
            for ti in range(min(n_sim, len(trajs))):
                start_h = trajs[ti][0]
                # Simulate autonomous ODE from start point
                h_t = torch.tensor(start_h, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                sim_traj = [h_t.cpu().numpy().flatten()]
                with torch.no_grad():
                    for _ in range(200):  # 200 steps
                        dhdt_t = ode_func(0, h_t)
                        h_t = h_t + dhdt_t * ODE_DT
                        sim_traj.append(h_t.cpu().numpy().flatten())
                sim_traj = np.array(sim_traj)
                sim_pca = pca.transform(sim_traj)
                ax.plot(sim_pca[:, 0], sim_pca[:, 1], color=colors[gn],
                       alpha=0.4, linewidth=0.8)
                ax.scatter(sim_pca[0, 0], sim_pca[0, 1], color=colors[gn],
                          s=20, zorder=4, edgecolors='black', linewidths=0.3)
        if len(fps) > 0:
            ax.scatter(fp_pca[:, 0], fp_pca[:, 1], marker='x', s=100,
                      c='black', linewidths=2, zorder=10)
        ax.set_xlim(-pc_range, pc_range)
        ax.set_ylim(-pc_range, pc_range)
        ax.set_xlabel(f'PC1')
        ax.set_ylabel(f'PC2')
        ax.set_title('Autonomous ODE from bout starts')

        # --- Row 2: Quantitative comparisons ---

        # Panel 5: Flow speed boxplot
        ax = axes[1, 0]
        plot_data = []
        plot_labels = []
        for gn in group_names:
            vals = results_df[results_df['group'] == gn]['flow_speed_mean'].dropna()
            if len(vals) > 0:
                plot_data.append(vals.values)
                plot_labels.append(gn)
        if plot_data:
            bp = ax.boxplot(plot_data, tick_labels=plot_labels, patch_artist=True,
                           showfliers=True, flierprops={'markersize': 3})
            for patch, lbl in zip(bp['boxes'], plot_labels):
                patch.set_facecolor(colors[lbl]); patch.set_alpha(0.6)
            for i, (d, lbl) in enumerate(zip(plot_data, plot_labels)):
                x = np.random.normal(i+1, 0.04, len(d))
                ax.scatter(x, d, c=colors[lbl], alpha=0.4, s=10, zorder=3)
        ax.set_ylabel('Mean |dh/dt|')
        ax.set_title('Flow speed during bouts')

        # Panel 6: Distance to fixed point
        ax = axes[1, 1]
        plot_data = []
        plot_labels = []
        for gn in group_names:
            vals = results_df[results_df['group'] == gn]['dist_to_fp_mean'].dropna()
            if len(vals) > 0:
                plot_data.append(vals.values)
                plot_labels.append(gn)
        if plot_data:
            bp = ax.boxplot(plot_data, tick_labels=plot_labels, patch_artist=True,
                           showfliers=True, flierprops={'markersize': 3})
            for patch, lbl in zip(bp['boxes'], plot_labels):
                patch.set_facecolor(colors[lbl]); patch.set_alpha(0.6)
            for i, (d, lbl) in enumerate(zip(plot_data, plot_labels)):
                x = np.random.normal(i+1, 0.04, len(d))
                ax.scatter(x, d, c=colors[lbl], alpha=0.4, s=10, zorder=3)
        ax.set_ylabel('Mean dist to FP')
        ax.set_title('Distance to dominant fixed point')

        # Panel 7: Flow-trajectory alignment
        ax = axes[1, 2]
        plot_data = []
        plot_labels = []
        for gn in group_names:
            vals = results_df[results_df['group'] == gn]['alignment'].dropna()
            if len(vals) > 0:
                plot_data.append(vals.values)
                plot_labels.append(gn)
        if plot_data:
            bp = ax.boxplot(plot_data, tick_labels=plot_labels, patch_artist=True,
                           showfliers=True, flierprops={'markersize': 3})
            for patch, lbl in zip(bp['boxes'], plot_labels):
                patch.set_facecolor(colors[lbl]); patch.set_alpha(0.6)
            for i, (d, lbl) in enumerate(zip(plot_data, plot_labels)):
                x = np.random.normal(i+1, 0.04, len(d))
                ax.scatter(x, d, c=colors[lbl], alpha=0.4, s=10, zorder=3)
        ax.set_ylabel('cos(flow, velocity)')
        ax.set_title('Flow-trajectory alignment')

        # Panel 8: % time in slow flow regions
        ax = axes[1, 3]
        plot_data = []
        plot_labels = []
        for gn in group_names:
            vals = results_df[results_df['group'] == gn]['pct_slow_flow'].dropna()
            if len(vals) > 0:
                plot_data.append(vals.values)
                plot_labels.append(gn)
        if plot_data:
            bp = ax.boxplot(plot_data, tick_labels=plot_labels, patch_artist=True,
                           showfliers=True, flierprops={'markersize': 3})
            for patch, lbl in zip(bp['boxes'], plot_labels):
                patch.set_facecolor(colors[lbl]); patch.set_alpha(0.6)
            for i, (d, lbl) in enumerate(zip(plot_data, plot_labels)):
                x = np.random.normal(i+1, 0.04, len(d))
                ax.scatter(x, d, c=colors[lbl], alpha=0.4, s=10, zorder=3)
        ax.set_ylabel('% time in slow flow')
        ax.set_title('Time near attractors')

        plt.tight_layout()
        plt.savefig(f'figures/hesitant_latent_dynamics_{region.lower()}.png',
                    dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n  Saved: figures/hesitant_latent_dynamics_{region.lower()}.png")
        sys.stdout.flush()

    # Save all results
    results_df.to_csv('data/hesitant_latent_dynamics_results.csv', index=False)
    print(f"\nSaved: data/hesitant_latent_dynamics_results.csv")
    print("\nDone!")


if __name__ == "__main__":
    main()
