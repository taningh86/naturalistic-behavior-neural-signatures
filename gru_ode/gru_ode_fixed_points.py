"""
Fixed Point Analysis of GRU-ODE with Tau -- LHA and RSP Fed
============================================================
Finds fixed points (attractors) where dh/dt = 0 in the learned ODE dynamics.

Method:
  1. Load trained With-tau models for LHA Fed and RSP Fed
  2. Extract hidden states from data as realistic starting points
  3. Optimize ||f(h)||^2 -> 0 via gradient descent to find fixed points
  4. Cluster nearby fixed points to get unique attractors
  5. Compute Jacobian eigenvalues to classify stability
  6. Visualize in PCA space
"""

import yaml
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import spikeinterface.extractors as se
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchdiffeq import odeint
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from scipy import stats as sp_stats
import warnings
import time

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

BIN_SIZE_MS = 10
PRED_WINDOW_MS = 100
PRED_BINS = PRED_WINDOW_MS // BIN_SIZE_MS
FS = 30000
BIN_SAMPLES = int(BIN_SIZE_MS * FS / 1000)

SEQ_LEN = 50
STRIDE = 50
D_SHARED = 32
HIDDEN_SIZE = 32
ODE_GATE_HIDDEN = 64
TRAIN_FRAC = 0.8

ODE_SOLVER = 'rk4'
ODE_STEP_SIZE = 1.0
ODE_DT = 1.0

LHA_DEPTH_MAX = 1300
RSP_DEPTH_MIN = 1300

# Fixed point search parameters
N_STARTS = 500        # Number of initial conditions to search from
FP_LR = 0.01          # Learning rate for fixed point optimization
FP_STEPS = 10000      # Max optimization steps
FP_TOL = 1e-10        # Convergence threshold for ||f(h)||^2
CLUSTER_EPS = 3.0     # DBSCAN eps for clustering nearby fixed points
FP_SPEED_THRESH = 1e-6  # Threshold for ||f(h)||^2 to consider "found"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open("paths.yaml") as f:
    paths_config = yaml.safe_load(f)

SESSION_INFO = {
    1: {'state': 'Fed', 'phase': 'Exploration'},
    2: {'state': 'Fed', 'phase': 'Foraging'},
    3: {'state': 'Fed', 'phase': 'Exploration'},
    4: {'state': 'Fed', 'phase': 'Foraging'},
}


# =============================================================================
# MODEL DEFINITION (must match training script)
# =============================================================================

class GRUODEFuncTau(nn.Module):
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
        self.log_tau = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, t, h):
        z = self.update_gate(h)
        n = self.candidate(h)
        tau = torch.exp(self.log_tau)
        return (1.0 / tau) * (1 - z) * (n - h)


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

        self.ode_func = GRUODEFuncTau(hidden_size, gate_hidden)
        self.obs_cell = nn.GRUCell(input_size=d_shared, hidden_size=hidden_size)
        self.fc_shared = nn.Linear(hidden_size, d_shared)

        self.output_projections = nn.ModuleDict()
        for sn, n_neurons in session_neuron_counts.items():
            self.output_projections[str(sn)] = nn.Linear(d_shared, n_neurons)

        self.register_buffer('t_span', torch.tensor([0.0, ODE_DT]))

    def _ode_evolve(self, h):
        h_evolved = odeint(
            self.ode_func, h, self.t_span,
            method=ODE_SOLVER, options={'step_size': ODE_STEP_SIZE},
        )
        return h_evolved[-1]

    def extract_hidden_states(self, x, session_num):
        sn_key = str(session_num)
        batch_size = x.shape[0]
        with torch.no_grad():
            h = torch.zeros(batch_size, self.hidden_size, device=x.device)
            hidden_seq = []
            for k in range(x.shape[1]):
                h = self._ode_evolve(h)
                x_proj = self.input_projections[sn_key](x[:, k, :])
                h = self.obs_cell(x_proj, h)
                hidden_seq.append(h.unsqueeze(1))
            return torch.cat(hidden_seq, dim=1)


# =============================================================================
# DATA LOADING
# =============================================================================

def get_good_units_by_region(sorted_path_obj):
    ci = sorted_path_obj / "cluster_info.tsv"
    if not ci.exists():
        return np.array([]), np.array([])
    df = pd.read_csv(ci, sep='\t')
    if 'depth' not in df.columns:
        return np.array([]), np.array([])
    label_col = None
    if 'group' in df.columns and df['group'].eq('good').any():
        label_col = 'group'
    elif 'KSLabel' in df.columns:
        label_col = 'KSLabel'
    if label_col is None:
        return np.array([]), np.array([])
    good = df[df[label_col] == 'good']
    lha_ids = good[good['depth'] < LHA_DEPTH_MAX]['cluster_id'].values
    rsp_ids = good[good['depth'] >= RSP_DEPTH_MIN]['cluster_id'].values
    return lha_ids, rsp_ids


def bin_spike_trains(sorting, unit_ids):
    spike_trains = {}
    all_min, all_max = np.inf, 0
    for uid in unit_ids:
        st = sorting.get_unit_spike_train(uid)
        spike_trains[uid] = st
        if len(st) > 0:
            all_min = min(all_min, np.min(st))
            all_max = max(all_max, np.max(st))
    n_bins = int((all_max - all_min) / BIN_SAMPLES) + 1
    data = np.zeros((n_bins, len(unit_ids)), dtype=np.float32)
    for i, uid in enumerate(unit_ids):
        st = spike_trains[uid]
        if len(st) > 0:
            b = ((st - all_min) // BIN_SAMPLES).astype(int)
            b = b[(b >= 0) & (b < n_bins)]
            np.add.at(data[:, i], b, 1)
    raw_data = data.copy()
    means = data.mean(axis=0, keepdims=True)
    stds = data.std(axis=0, keepdims=True)
    stds[stds < 1e-8] = 1.0
    zscore_data = (data - means) / stds
    return zscore_data, raw_data, n_bins


def load_fed_sessions():
    sessions_data = {}
    sp = paths_config['single_probe']['coordinates_1']['mouse01']['sessions']
    for sess_num, info in SESSION_INFO.items():
        key = f"session_{sess_num}"
        sc = sp[key]
        sorted_path = Path(sc['sorted'])
        if not sorted_path.exists():
            continue
        sorting = se.read_kilosort(sorted_path)
        lha_ids, rsp_ids = get_good_units_by_region(sorted_path)
        if len(lha_ids) < 3 or len(rsp_ids) < 3:
            continue
        lha_zscore, lha_raw, lha_bins = bin_spike_trains(sorting, lha_ids)
        rsp_zscore, rsp_raw, rsp_bins = bin_spike_trains(sorting, rsp_ids)
        sessions_data[sess_num] = {
            'lha': {'zscore': lha_zscore, 'raw': lha_raw,
                    'n_neurons': len(lha_ids), 'n_bins': lha_bins},
            'rsp': {'zscore': rsp_zscore, 'raw': rsp_raw,
                    'n_neurons': len(rsp_ids), 'n_bins': rsp_bins},
            'state': info['state'], 'phase': info['phase'],
        }
        print(f"  Session {sess_num}: LHA={len(lha_ids)}, RSP={len(rsp_ids)} neurons")
    return sessions_data


# =============================================================================
# EXTRACT HIDDEN STATES
# =============================================================================

def extract_hidden_states(model, sessions_data, session_nums, region, device):
    all_hidden = []
    for sn in session_nums:
        zscore = sessions_data[sn][region]['zscore']
        T = len(zscore)
        seqs = []
        for i in range(0, T - SEQ_LEN - PRED_BINS, STRIDE):
            seqs.append(zscore[i:i + SEQ_LEN])
        seqs_t = torch.tensor(np.array(seqs), dtype=torch.float32).to(device)
        model.eval()
        chunk_size = 128
        for start in range(0, len(seqs_t), chunk_size):
            chunk = seqs_t[start:start + chunk_size]
            h = model.extract_hidden_states(chunk, sn)
            all_hidden.append(h[:, -1, :].cpu().numpy())
    return np.concatenate(all_hidden, axis=0)


# =============================================================================
# FIXED POINT SEARCH
# =============================================================================

def find_fixed_points(ode_func, hidden_states, n_starts, lr, max_steps, tol,
                      speed_thresh, device):
    """
    Find fixed points by minimizing ||f(h)||^2 from sampled starting points.
    BATCHED: all starting points optimized simultaneously for GPU efficiency.
    """
    n_total = len(hidden_states)
    indices = np.random.choice(n_total, size=min(n_starts, n_total), replace=False)
    starts = hidden_states[indices]

    # All starting points as a single batch tensor
    h = torch.tensor(starts, dtype=torch.float32, device=device).requires_grad_(True)

    optimizer = torch.optim.Adam([h], lr=lr)
    ode_func.eval()

    for step in range(max_steps):
        optimizer.zero_grad()
        with torch.enable_grad():
            dhdt = ode_func(0, h)  # (N, hidden_size)
            speed_sq = (dhdt ** 2).sum(dim=1)  # (N,)
            loss = speed_sq.sum()
        loss.backward()
        optimizer.step()

        if (step + 1) % 1000 == 0:
            min_speed = speed_sq.min().item()
            mean_speed = speed_sq.mean().item()
            n_converged = (speed_sq < speed_thresh).sum().item()
            print(f"    Step {step+1}: min_speed={min_speed:.2e}, "
                  f"mean_speed={mean_speed:.2e}, converged={n_converged}/{len(starts)}")

        # Stop early if all have converged
        if speed_sq.max().item() < tol:
            print(f"    All converged at step {step+1}")
            break

    # Collect those that converged
    with torch.no_grad():
        dhdt_final = ode_func(0, h)
        final_speeds = (dhdt_final ** 2).sum(dim=1).cpu().numpy()

    h_np = h.detach().cpu().numpy()
    mask = final_speeds < speed_thresh
    found_fps = h_np[mask]
    found_speeds = final_speeds[mask]

    print(f"    Final: {mask.sum()}/{len(starts)} converged (speed < {speed_thresh})")

    if len(found_fps) == 0:
        return np.array([]).reshape(0, HIDDEN_SIZE), np.array([])

    return found_fps, found_speeds


def cluster_fixed_points(fps, eps=0.5):
    """Cluster nearby fixed points using DBSCAN."""
    if len(fps) == 0:
        return np.array([]).reshape(0, HIDDEN_SIZE), np.array([])

    clustering = DBSCAN(eps=eps, min_samples=1).fit(fps)
    labels = clustering.labels_
    unique_labels = set(labels)

    unique_fps = []
    cluster_sizes = []
    for label in sorted(unique_labels):
        mask = labels == label
        centroid = fps[mask].mean(axis=0)
        unique_fps.append(centroid)
        cluster_sizes.append(mask.sum())

    return np.array(unique_fps), np.array(cluster_sizes)


# =============================================================================
# JACOBIAN AND STABILITY ANALYSIS
# =============================================================================

def compute_jacobian(ode_func, fp, device):
    """Compute Jacobian df/dh at a fixed point."""
    h = torch.tensor(fp, dtype=torch.float32, device=device).unsqueeze(0).requires_grad_(True)
    dhdt = ode_func(0, h)

    jacobian = torch.zeros(HIDDEN_SIZE, HIDDEN_SIZE, device=device)
    for i in range(HIDDEN_SIZE):
        if h.grad is not None:
            h.grad.zero_()
        dhdt[0, i].backward(retain_graph=True)
        jacobian[i] = h.grad[0].clone()

    return jacobian.cpu().numpy()


def analyze_stability(jacobian):
    """Analyze stability from Jacobian eigenvalues."""
    eigenvalues = np.linalg.eigvals(jacobian)
    real_parts = eigenvalues.real
    imag_parts = eigenvalues.imag

    max_real = np.max(real_parts)
    n_positive = np.sum(real_parts > 0)
    n_negative = np.sum(real_parts < 0)
    n_oscillatory = np.sum(np.abs(imag_parts) > 1e-6)

    if max_real < -1e-6:
        stability = "stable"
    elif max_real > 1e-6:
        if n_positive == HIDDEN_SIZE:
            stability = "unstable"
        else:
            stability = "saddle"
    else:
        stability = "marginal"

    # Dominant timescale from slowest decaying eigenvalue
    negative_reals = real_parts[real_parts < 0]
    if len(negative_reals) > 0:
        slowest_decay = np.max(negative_reals)  # Least negative = slowest
        dominant_timescale = -1.0 / slowest_decay if slowest_decay != 0 else np.inf
    else:
        dominant_timescale = np.inf

    return {
        'eigenvalues': eigenvalues,
        'max_real': max_real,
        'n_positive': n_positive,
        'n_negative': n_negative,
        'n_oscillatory': n_oscillatory,
        'stability': stability,
        'dominant_timescale': dominant_timescale,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print(f"Device: {DEVICE}")
    print(f"Fixed Point Analysis -- GRU-ODE with tau, Fed State")
    print(f"Searching from {N_STARTS} starting points per region")
    print()

    # --- Load data ---
    print("Loading Fed session data...")
    sessions_data = load_fed_sessions()
    session_nums = sorted(sessions_data.keys())
    print(f"Loaded {len(session_nums)} sessions\n")

    all_fp_results = []

    for region in ['lha', 'rsp']:
        print(f"\n{'='*60}")
        print(f"  {region.upper()} Fed -- Fixed Point Analysis")
        print(f"{'='*60}")

        # Load model
        model_path = Path("data") / f"gru_ode_10ms_tau_{region}_fed_model.pt"
        checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
        neuron_counts = checkpoint['neuron_counts']

        model = PooledGRUODE(neuron_counts, D_SHARED, HIDDEN_SIZE, ODE_GATE_HIDDEN,
                             PRED_BINS).to(DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        # Print tau values
        tau_vals = torch.exp(model.ode_func.log_tau).detach().cpu().numpy()
        print(f"  Tau: mean={tau_vals.mean():.4f}, range=[{tau_vals.min():.4f}, {tau_vals.max():.4f}]")

        # Extract hidden states as starting points
        print(f"  Extracting hidden states...")
        hidden_states = extract_hidden_states(model, sessions_data, session_nums, region, DEVICE)
        print(f"  Got {len(hidden_states)} hidden states")
        print(f"  Hidden state stats: mean={hidden_states.mean():.4f}, "
              f"std={hidden_states.std():.4f}, "
              f"range=[{hidden_states.min():.4f}, {hidden_states.max():.4f}]")

        # Search for fixed points
        print(f"\n  Searching for fixed points...")
        t0 = time.time()
        fps, speeds = find_fixed_points(
            model.ode_func, hidden_states, N_STARTS, FP_LR, FP_STEPS, FP_TOL,
            FP_SPEED_THRESH, DEVICE
        )
        elapsed = time.time() - t0
        print(f"  Found {len(fps)} fixed points in {elapsed:.1f}s")

        if len(fps) == 0:
            print(f"  No fixed points found! Trying with relaxed threshold...")
            fps, speeds = find_fixed_points(
                model.ode_func, hidden_states, N_STARTS, FP_LR, FP_STEPS, FP_TOL,
                1e-4, DEVICE  # Relaxed threshold
            )
            print(f"  Found {len(fps)} with relaxed threshold")

        if len(fps) == 0:
            print(f"  Still no fixed points. Skipping {region.upper()}.")
            continue

        # Cluster nearby fixed points
        unique_fps, cluster_sizes = cluster_fixed_points(fps, CLUSTER_EPS)
        print(f"  Clustered into {len(unique_fps)} unique fixed points")
        for i, (fp, sz) in enumerate(zip(unique_fps, cluster_sizes)):
            print(f"    FP {i+1}: cluster size={sz}, norm={np.linalg.norm(fp):.4f}")

        # Stability analysis
        print(f"\n  Stability analysis (Jacobian eigenvalues):")
        fp_analyses = []
        for i, fp in enumerate(unique_fps):
            jacobian = compute_jacobian(model.ode_func, fp, DEVICE)
            analysis = analyze_stability(jacobian)
            fp_analyses.append(analysis)

            eig_real = analysis['eigenvalues'].real
            print(f"    FP {i+1}: {analysis['stability']}, "
                  f"max_real_eig={analysis['max_real']:.6f}, "
                  f"n_pos={analysis['n_positive']}, n_neg={analysis['n_negative']}, "
                  f"n_osc={analysis['n_oscillatory']}, "
                  f"dominant_tau={analysis['dominant_timescale']:.2f}")

            all_fp_results.append({
                'region': region.upper(),
                'fp_id': i + 1,
                'cluster_size': cluster_sizes[i],
                'norm': np.linalg.norm(fp),
                'stability': analysis['stability'],
                'max_real_eig': analysis['max_real'],
                'n_positive_eig': analysis['n_positive'],
                'n_negative_eig': analysis['n_negative'],
                'n_oscillatory': analysis['n_oscillatory'],
                'dominant_timescale': analysis['dominant_timescale'],
            })

        # --- Visualization ---
        print(f"\n  Generating {region.upper()} figures...")

        # PCA on hidden states
        pca = PCA(n_components=3).fit(hidden_states)
        h_pca = pca.transform(hidden_states)
        fp_pca = pca.transform(unique_fps)

        stability_colors = {'stable': '#2E7D32', 'unstable': '#C62828',
                           'saddle': '#E65100', 'marginal': '#F9A825'}
        stability_markers = {'stable': '*', 'unstable': 'X',
                            'saddle': 'D', 'marginal': 's'}

        # Count stability types
        stability_counts = {}
        for a in fp_analyses:
            s = a['stability']
            stability_counts[s] = stability_counts.get(s, 0) + 1

        # =====================================================================
        # FIGURE: 2x2 panel — trajectories, directions, speed, eigenvalues
        # =====================================================================
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        counts_str = ", ".join(f"{k}: {v}" for k, v in sorted(stability_counts.items()))
        fig.suptitle(f"{region.upper()} Fed -- Fixed Point Dynamics\n"
                     f"({len(unique_fps)} fixed points: {counts_str})",
                     fontsize=14, fontweight='bold')

        # --- Panel A: Trajectories approaching fixed points ---
        ax = axes[0, 0]

        # Simulate trajectories from random starting points toward FPs
        n_traj = 30
        traj_steps = 200
        traj_idx = np.random.choice(len(hidden_states), size=n_traj, replace=False)
        traj_starts = hidden_states[traj_idx]

        model.ode_func.eval()
        trajectories_pca = []
        with torch.no_grad():
            for start in traj_starts:
                h_t = torch.tensor(start, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                traj = [h_t.cpu().numpy().flatten()]
                for _ in range(traj_steps):
                    dhdt = model.ode_func(0, h_t)
                    h_t = h_t + dhdt * ODE_DT  # Euler step
                    traj.append(h_t.cpu().numpy().flatten())
                traj_np = np.array(traj)
                traj_pca_pts = pca.transform(traj_np)
                trajectories_pca.append(traj_pca_pts)

        # Plot hidden states as background
        n_plot = min(3000, len(h_pca))
        plot_idx = np.random.choice(len(h_pca), n_plot, replace=False)
        ax.scatter(h_pca[plot_idx, 0], h_pca[plot_idx, 1],
                   c='lightgray', s=1, alpha=0.2, rasterized=True)

        # Plot trajectories colored by time
        cmap_traj = plt.cm.viridis
        for traj_pca_pts in trajectories_pca:
            n_pts = len(traj_pca_pts)
            colors = cmap_traj(np.linspace(0, 1, n_pts))
            for t_step in range(n_pts - 1):
                ax.plot(traj_pca_pts[t_step:t_step+2, 0],
                        traj_pca_pts[t_step:t_step+2, 1],
                        color=colors[t_step], linewidth=0.8, alpha=0.7)
            # Start marker
            ax.scatter(traj_pca_pts[0, 0], traj_pca_pts[0, 1],
                       c='blue', s=15, zorder=5, alpha=0.5)

        # Plot fixed points
        plotted_types = set()
        for k, (fp_p, analysis) in enumerate(zip(fp_pca, fp_analyses)):
            stype = analysis['stability']
            color = stability_colors.get(stype, 'gray')
            marker = stability_markers.get(stype, 'o')
            sz = max(150, min(400, cluster_sizes[k] * 30))
            label = f"{stype} (n={stability_counts[stype]})" if stype not in plotted_types else None
            ax.scatter(fp_p[0], fp_p[1], c=color, s=sz, marker=marker,
                      edgecolors='black', linewidths=1.5, zorder=10, label=label)
            plotted_types.add(stype)

        ax.set_xlabel('PC1', fontsize=11)
        ax.set_ylabel('PC2', fontsize=11)
        ax.set_title('A) Trajectories (blue=start, yellow=end)', fontsize=12)
        ax.legend(fontsize=10, loc='best')

        # --- Panel B: Stable/unstable directions at dominant FP ---
        ax = axes[0, 1]

        # Find dominant FP (largest cluster)
        dom_idx = np.argmax(cluster_sizes)
        dom_fp = unique_fps[dom_idx]
        dom_analysis = fp_analyses[dom_idx]
        dom_fp_pca = fp_pca[dom_idx]

        # Get Jacobian eigenvectors in full space, project to PCA
        dom_jac = compute_jacobian(model.ode_func, dom_fp, DEVICE)
        eig_vals, eig_vecs = np.linalg.eig(dom_jac)
        eig_vecs_real = eig_vecs.real

        # Sort by real part of eigenvalue
        sort_order = np.argsort(eig_vals.real)
        eig_vals_sorted = eig_vals[sort_order]
        eig_vecs_sorted = eig_vecs_real[:, sort_order]

        # Plot hidden states
        ax.scatter(h_pca[plot_idx, 0], h_pca[plot_idx, 1],
                   c='lightgray', s=1, alpha=0.2, rasterized=True)

        # Plot dominant FP
        dom_color = stability_colors.get(dom_analysis['stability'], 'gray')
        dom_marker = stability_markers.get(dom_analysis['stability'], 'o')
        ax.scatter(dom_fp_pca[0], dom_fp_pca[1], c=dom_color, s=300,
                   marker=dom_marker, edgecolors='black', linewidths=2, zorder=10)

        # Draw eigenvector arrows in PCA space
        arrow_scale = 2.0  # Scale for visibility
        n_stable_shown = 0
        n_unstable_shown = 0
        for ev_idx in range(len(eig_vals_sorted)):
            ev_real = eig_vals_sorted[ev_idx].real
            evec = eig_vecs_sorted[:, ev_idx]
            # Project eigenvector to PCA space
            evec_pca = pca.transform(evec.reshape(1, -1)).flatten()
            evec_dir = evec_pca[:2]
            evec_len = np.linalg.norm(evec_dir)
            if evec_len < 1e-8:
                continue
            evec_dir = evec_dir / evec_len

            if ev_real > 1e-6:
                # Unstable direction -- red, thick
                ax.annotate('', xy=(dom_fp_pca[0] + evec_dir[0]*arrow_scale,
                                    dom_fp_pca[1] + evec_dir[1]*arrow_scale),
                            xytext=(dom_fp_pca[0], dom_fp_pca[1]),
                            arrowprops=dict(arrowstyle='->', color='red',
                                          lw=3, mutation_scale=20))
                ax.annotate('', xy=(dom_fp_pca[0] - evec_dir[0]*arrow_scale,
                                    dom_fp_pca[1] - evec_dir[1]*arrow_scale),
                            xytext=(dom_fp_pca[0], dom_fp_pca[1]),
                            arrowprops=dict(arrowstyle='->', color='red',
                                          lw=3, mutation_scale=20))
                n_unstable_shown += 1
            elif ev_real < -1e-6 and n_stable_shown < 3:
                # Stable directions -- green, thin (show a few of the strongest)
                ax.annotate('', xy=(dom_fp_pca[0], dom_fp_pca[1]),
                            xytext=(dom_fp_pca[0] + evec_dir[0]*arrow_scale,
                                    dom_fp_pca[1] + evec_dir[1]*arrow_scale),
                            arrowprops=dict(arrowstyle='->', color='#2E7D32',
                                          lw=1.5, mutation_scale=15))
                ax.annotate('', xy=(dom_fp_pca[0], dom_fp_pca[1]),
                            xytext=(dom_fp_pca[0] - evec_dir[0]*arrow_scale,
                                    dom_fp_pca[1] - evec_dir[1]*arrow_scale),
                            arrowprops=dict(arrowstyle='->', color='#2E7D32',
                                          lw=1.5, mutation_scale=15))
                n_stable_shown += 1

        # Manual legend for arrows
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='#2E7D32', lw=2,
                   label=f'Stable dirs ({dom_analysis["n_negative"]} dims)'),
            Line2D([0], [0], color='red', lw=3,
                   label=f'Unstable dirs ({dom_analysis["n_positive"]} dims)'),
        ]
        if dom_analysis['n_positive'] > 0:
            ax.legend(handles=legend_elements, fontsize=10, loc='best')
        else:
            ax.legend(handles=[legend_elements[0]], fontsize=10, loc='best')

        ax.set_xlabel('PC1', fontsize=11)
        ax.set_ylabel('PC2', fontsize=11)
        ax.set_title(f'B) Dominant FP: {dom_analysis["stability"]} '
                     f'(cluster n={cluster_sizes[dom_idx]})', fontsize=12)

        # --- Panel C: Speed vs distance from dominant FP ---
        ax = axes[1, 0]

        # Compute speed (||dh/dt||) at each hidden state
        all_speeds = []
        all_dists = []
        chunk_sz = 512
        model.ode_func.eval()
        with torch.no_grad():
            for c_start in range(0, len(hidden_states), chunk_sz):
                chunk = torch.tensor(
                    hidden_states[c_start:c_start+chunk_sz],
                    dtype=torch.float32, device=DEVICE
                )
                dhdt = model.ode_func(0, chunk)
                speed = torch.sqrt((dhdt**2).sum(dim=1)).cpu().numpy()
                dist = np.sqrt(((hidden_states[c_start:c_start+chunk_sz] - dom_fp)**2).sum(axis=1))
                all_speeds.append(speed)
                all_dists.append(dist)

        all_speeds = np.concatenate(all_speeds)
        all_dists = np.concatenate(all_dists)

        # Bin by distance and compute mean speed per bin
        n_dist_bins = 30
        dist_bins = np.linspace(0, np.percentile(all_dists, 99), n_dist_bins + 1)
        bin_centers = 0.5 * (dist_bins[:-1] + dist_bins[1:])
        mean_speeds = []
        std_speeds = []
        for b in range(n_dist_bins):
            mask = (all_dists >= dist_bins[b]) & (all_dists < dist_bins[b+1])
            if mask.sum() > 0:
                mean_speeds.append(all_speeds[mask].mean())
                std_speeds.append(all_speeds[mask].std())
            else:
                mean_speeds.append(np.nan)
                std_speeds.append(np.nan)
        mean_speeds = np.array(mean_speeds)
        std_speeds = np.array(std_speeds)

        # Scatter (subsampled) + binned mean
        sub_idx = np.random.choice(len(all_speeds), min(2000, len(all_speeds)), replace=False)
        ax.scatter(all_dists[sub_idx], all_speeds[sub_idx],
                   c='lightblue', s=3, alpha=0.3, rasterized=True)
        ax.plot(bin_centers, mean_speeds, 'o-', color='navy', linewidth=2,
                markersize=4, label='Mean speed')
        ax.fill_between(bin_centers, mean_speeds - std_speeds,
                        mean_speeds + std_speeds, color='navy', alpha=0.15)

        # Mark FP distance = 0
        ax.axvline(x=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_xlabel('Distance from dominant FP', fontsize=11)
        ax.set_ylabel('Speed ||dh/dt||', fontsize=11)
        ax.set_title('C) Speed vs Distance from FP', fontsize=12)
        ax.legend(fontsize=10)

        # Also compute speed along trajectories for a few
        # Show that near saddle: speed dips then rises (escape)
        # vs near stable: speed monotonically decreases
        ax_inset = ax.inset_axes([0.55, 0.55, 0.42, 0.4])
        for t_i, traj_pca_pts in enumerate(trajectories_pca[:8]):
            # Recompute speed along this trajectory
            n_pts = len(traj_pca_pts)
            traj_full = pca.inverse_transform(traj_pca_pts)
            traj_t = torch.tensor(traj_full, dtype=torch.float32, device=DEVICE)
            with torch.no_grad():
                dhdt = model.ode_func(0, traj_t)
                speeds = torch.sqrt((dhdt**2).sum(dim=1)).cpu().numpy()
            time_axis = np.arange(n_pts)
            ax_inset.plot(time_axis, speeds, linewidth=0.8, alpha=0.6)
        ax_inset.set_xlabel('ODE step', fontsize=8)
        ax_inset.set_ylabel('Speed', fontsize=8)
        ax_inset.set_title('Speed along trajectories', fontsize=8)
        ax_inset.tick_params(labelsize=7)

        # --- Panel D: Eigenvalue spectrum ---
        ax = axes[1, 1]
        top_indices = np.argsort(cluster_sizes)[::-1][:5]
        for rank, idx in enumerate(top_indices):
            analysis = fp_analyses[idx]
            stype = analysis['stability']
            eig_real_sorted = np.sort(analysis['eigenvalues'].real)
            color = stability_colors.get(stype, 'gray')
            ax.plot(range(HIDDEN_SIZE), eig_real_sorted, 'o-', markersize=4,
                    color=color, linewidth=1.5,
                    label=f"FP{idx+1} ({stype}, n={cluster_sizes[idx]})", alpha=0.8)
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax.set_xlabel('Eigenvalue index (sorted)', fontsize=11)
        ax.set_ylabel('Real part of eigenvalue', fontsize=11)
        ax.set_title('D) Eigenvalue Spectrum', fontsize=12)
        ax.legend(fontsize=9)

        # Shade positive region
        ax.axhspan(0, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 0.01,
                    color='red', alpha=0.05)
        ax.axhspan(ax.get_ylim()[0], 0, color='green', alpha=0.05)

        plt.tight_layout()
        fig.savefig(Path("figures") / f"gru_ode_fixed_points_{region}_fed.png",
                    dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: figures/gru_ode_fixed_points_{region}_fed.png")

    # Save results
    if all_fp_results:
        fp_df = pd.DataFrame(all_fp_results)
        fp_df.to_csv(Path("data") / "gru_ode_fixed_points_fed.csv", index=False)
        print(f"\nResults saved: data/gru_ode_fixed_points_fed.csv")

        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        for _, row in fp_df.iterrows():
            print(f"  {row['region']} FP{row['fp_id']}: {row['stability']}, "
                  f"cluster_size={row['cluster_size']}, "
                  f"max_real_eig={row['max_real_eig']:.6f}, "
                  f"dominant_tau={row['dominant_timescale']:.2f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
