"""
GRU-ODE Manifold Analysis
=========================
Four analyses of the learned neural manifold:
  A) Local dimensionality map -- intrinsic dim varies across the manifold
  B) State density / dwell time -- where the system spends most time
  C) Fed vs fasted manifold comparison -- Procrustes + principal angles
  D) Flow on manifold -- vector field + fixed points

Produces one 2x2 figure per region (LHA, RSP).
"""

import yaml
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm
import spikeinterface.extractors as se
import torch
import torch.nn as nn
from torchdiffeq import odeint
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from scipy.spatial import procrustes
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

ODE_SOLVER = 'rk4'
ODE_STEP_SIZE = 1.0
ODE_DT = 1.0

LHA_DEPTH_MAX = 1300
RSP_DEPTH_MIN = 1300

# Local dimensionality
K_NEIGHBORS = 50

# Fixed point search
N_STARTS = 500
FP_LR = 0.01
FP_STEPS = 10000
FP_SPEED_THRESH = 1e-6
CLUSTER_EPS = 3.0

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


# =============================================================================
# MODEL DEFINITION
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
        return odeint(self.ode_func, h, self.t_span,
                      method=ODE_SOLVER, options={'step_size': ODE_STEP_SIZE})[-1]

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


def load_sessions(session_nums):
    sessions_data = {}
    sp = paths_config['single_probe']['coordinates_1']['mouse01']['sessions']
    for sess_num in session_nums:
        info = SESSION_INFO[sess_num]
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


def extract_hidden(model, sessions_data, session_nums, region, device):
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


def load_model(region, state, device):
    model_path = Path("data") / f"gru_ode_10ms_tau_{region}_{state.lower()}_model.pt"
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    neuron_counts = checkpoint['neuron_counts']
    model = PooledGRUODE(neuron_counts, D_SHARED, HIDDEN_SIZE, ODE_GATE_HIDDEN,
                         PRED_BINS).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    tau_vals = torch.exp(model.ode_func.log_tau).detach().cpu().numpy()
    return model, tau_vals


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def compute_local_dimensionality(hidden_states, k=K_NEIGHBORS):
    """Compute local participation ratio in k-nearest-neighbor neighborhoods."""
    nn_model = NearestNeighbors(n_neighbors=k, algorithm='auto')
    nn_model.fit(hidden_states)
    _, indices = nn_model.kneighbors(hidden_states)

    local_pr = np.zeros(len(hidden_states))
    for i in range(len(hidden_states)):
        neighbors = hidden_states[indices[i]]
        neighbors_centered = neighbors - neighbors.mean(axis=0)
        cov = np.cov(neighbors_centered.T)
        eigvals = np.linalg.eigvalsh(cov)
        eigvals = np.maximum(eigvals, 0)
        total = eigvals.sum()
        if total > 1e-12:
            local_pr[i] = total**2 / (eigvals**2).sum()
        else:
            local_pr[i] = 1.0
    return local_pr


def compute_flow_field(ode_func, pca, pc1_range, pc2_range, device):
    """Compute dh/dt projected onto PC1-PC2 grid."""
    grid_pc1, grid_pc2 = np.meshgrid(pc1_range, pc2_range)
    flow_u = np.zeros_like(grid_pc1)
    flow_v = np.zeros_like(grid_pc2)
    speed = np.zeros_like(grid_pc1)

    ode_func.eval()
    with torch.no_grad():
        for i in range(len(pc1_range)):
            for j in range(len(pc2_range)):
                pc_coords = np.zeros(3)
                pc_coords[0] = grid_pc1[j, i]
                pc_coords[1] = grid_pc2[j, i]
                h_recon = pca.inverse_transform(pc_coords)
                h_t = torch.tensor(h_recon, dtype=torch.float32,
                                   device=device).unsqueeze(0)
                dhdt = ode_func(0, h_t).cpu().numpy().flatten()
                dhdt_pca = pca.transform(dhdt.reshape(1, -1)).flatten()
                flow_u[j, i] = dhdt_pca[0]
                flow_v[j, i] = dhdt_pca[1]
                speed[j, i] = np.sqrt(dhdt_pca[0]**2 + dhdt_pca[1]**2)

    return grid_pc1, grid_pc2, flow_u, flow_v, speed


def find_fixed_points_batched(ode_func, hidden_states, device):
    """Find fixed points via batched optimization."""
    n_total = len(hidden_states)
    indices = np.random.choice(n_total, size=min(N_STARTS, n_total), replace=False)
    starts = hidden_states[indices]
    h = torch.tensor(starts, dtype=torch.float32, device=device).requires_grad_(True)
    optimizer = torch.optim.Adam([h], lr=FP_LR)
    ode_func.eval()

    for step in range(FP_STEPS):
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
    mask = final_speeds < FP_SPEED_THRESH
    if mask.sum() == 0:
        return np.array([]).reshape(0, HIDDEN_SIZE), np.array([])

    fps = h_np[mask]
    clustering = DBSCAN(eps=CLUSTER_EPS, min_samples=1).fit(fps)
    labels = clustering.labels_
    unique_fps = []
    cluster_sizes = []
    for lab in sorted(set(labels)):
        m = labels == lab
        unique_fps.append(fps[m].mean(axis=0))
        cluster_sizes.append(m.sum())
    return np.array(unique_fps), np.array(cluster_sizes)


def compute_principal_angles(pca_fed, pca_fasted, n_components=10):
    """Compute principal angles between fed and fasted subspaces."""
    U_fed = pca_fed.components_[:n_components].T      # (32, k)
    U_fas = pca_fasted.components_[:n_components].T    # (32, k)
    M = U_fed.T @ U_fas
    _, sigmas, _ = np.linalg.svd(M)
    sigmas = np.clip(sigmas, 0, 1)
    angles_deg = np.degrees(np.arccos(sigmas))
    return angles_deg, sigmas


# =============================================================================
# MAIN
# =============================================================================

def main():
    print(f"Device: {DEVICE}")
    print("GRU-ODE Manifold Analysis -- Fed vs Fasted\n")

    # Load all sessions
    print("Loading sessions...")
    fed_data = load_sessions([1, 2, 3, 4])
    fasted_data = load_sessions([5, 6, 7, 8])
    print()

    for region in ['lha', 'rsp']:
        print(f"\n{'='*60}")
        print(f"  {region.upper()} -- Manifold Analysis")
        print(f"{'='*60}")

        # Load models
        model_fed, tau_fed = load_model(region, 'fed', DEVICE)
        model_fasted, tau_fasted = load_model(region, 'fasted', DEVICE)
        print(f"  Fed tau: mean={tau_fed.mean():.3f}, Fasted tau: mean={tau_fasted.mean():.3f}")

        # Extract hidden states
        print("  Extracting hidden states...")
        t0 = time.time()
        h_fed = extract_hidden(model_fed, fed_data, [1,2,3,4], region, DEVICE)
        h_fasted = extract_hidden(model_fasted, fasted_data, [5,6,7,8], region, DEVICE)
        print(f"  Fed: {len(h_fed)} states, Fasted: {len(h_fasted)} states "
              f"({time.time()-t0:.1f}s)")

        # PCA for each condition
        pca_fed = PCA(n_components=min(10, HIDDEN_SIZE)).fit(h_fed)
        pca_fasted = PCA(n_components=min(10, HIDDEN_SIZE)).fit(h_fasted)
        h_fed_pca = pca_fed.transform(h_fed)
        h_fasted_pca = pca_fasted.transform(h_fasted)

        # =================================================================
        # Analysis A: Local dimensionality
        # =================================================================
        print("  Computing local dimensionality...")
        t0 = time.time()
        # Subsample for speed (local dim is expensive)
        n_sub = min(3000, len(h_fed), len(h_fasted))
        idx_fed = np.random.choice(len(h_fed), n_sub, replace=False)
        idx_fasted = np.random.choice(len(h_fasted), n_sub, replace=False)
        local_pr_fed = compute_local_dimensionality(h_fed[idx_fed])
        local_pr_fasted = compute_local_dimensionality(h_fasted[idx_fasted])
        print(f"  Local dim -- Fed: mean={local_pr_fed.mean():.2f}, "
              f"Fasted: mean={local_pr_fasted.mean():.2f} ({time.time()-t0:.1f}s)")

        # =================================================================
        # Analysis B: Density (KDE)
        # =================================================================
        print("  Computing density...")
        # Fed density in fed PCA space
        kde_fed = sp_stats.gaussian_kde(h_fed_pca[:, :2].T)
        # Fasted density in fasted PCA space
        kde_fasted = sp_stats.gaussian_kde(h_fasted_pca[:, :2].T)

        # =================================================================
        # Analysis C: Manifold comparison
        # =================================================================
        print("  Computing principal angles...")
        angles, cosines = compute_principal_angles(pca_fed, pca_fasted, n_components=10)
        print(f"  Principal angles (deg): {np.round(angles[:5], 1)}")

        # Procrustes alignment of fasted onto fed (in PCA space)
        # Use top 3 PCs, subsample to same size
        n_proc = min(2000, len(h_fed_pca), len(h_fasted_pca))
        fed_proc = h_fed_pca[np.random.choice(len(h_fed_pca), n_proc, replace=False), :3]
        fas_proc = h_fasted_pca[np.random.choice(len(h_fasted_pca), n_proc, replace=False), :3]
        # Normalize for Procrustes
        _, fas_aligned, disparity = procrustes(fed_proc, fas_proc)
        print(f"  Procrustes disparity: {disparity:.4f}")

        # =================================================================
        # Analysis D: Flow field + fixed points
        # =================================================================
        print("  Computing flow fields...")
        # Fed flow field
        pc1_r = np.linspace(h_fed_pca[:, 0].min()-0.5, h_fed_pca[:, 0].max()+0.5, 18)
        pc2_r = np.linspace(h_fed_pca[:, 1].min()-0.5, h_fed_pca[:, 1].max()+0.5, 18)
        pca3_fed = PCA(n_components=3).fit(h_fed)
        grid1, grid2, fu, fv, fspeed = compute_flow_field(
            model_fed.ode_func, pca3_fed, pc1_r, pc2_r, DEVICE)

        # Fasted flow field
        pc1_r_fas = np.linspace(h_fasted_pca[:, 0].min()-0.5, h_fasted_pca[:, 0].max()+0.5, 18)
        pc2_r_fas = np.linspace(h_fasted_pca[:, 1].min()-0.5, h_fasted_pca[:, 1].max()+0.5, 18)
        pca3_fasted = PCA(n_components=3).fit(h_fasted)
        grid1f, grid2f, fu_f, fv_f, fspeed_f = compute_flow_field(
            model_fasted.ode_func, pca3_fasted, pc1_r_fas, pc2_r_fas, DEVICE)

        # Fixed points
        print("  Finding fixed points...")
        fps_fed, fp_sizes_fed = find_fixed_points_batched(
            model_fed.ode_func, h_fed, DEVICE)
        fps_fasted, fp_sizes_fasted = find_fixed_points_batched(
            model_fasted.ode_func, h_fasted, DEVICE)
        print(f"  Fed: {len(fps_fed)} FPs, Fasted: {len(fps_fasted)} FPs")

        # Project FPs to PCA
        if len(fps_fed) > 0:
            fps_fed_pca = pca3_fed.transform(fps_fed)
        if len(fps_fasted) > 0:
            fps_fasted_pca = pca3_fasted.transform(fps_fasted)

        # =================================================================
        # FIGURE
        # =================================================================
        print(f"  Generating {region.upper()} figure...")

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle(f"{region.upper()} -- Neural Manifold Analysis\n"
                     f"Fed (sessions 1-4) vs Fasted (sessions 5-8)",
                     fontsize=14, fontweight='bold')

        # --- Panel A: Local Dimensionality ---
        ax = axes[0, 0]
        fed_pca_sub = pca_fed.transform(h_fed[idx_fed])
        fas_pca_sub = pca_fasted.transform(h_fasted[idx_fasted])

        # Fed scatter colored by local PR
        vmin = min(local_pr_fed.min(), local_pr_fasted.min())
        vmax = max(local_pr_fed.max(), local_pr_fasted.max())
        norm = Normalize(vmin=vmin, vmax=vmax)

        sc = ax.scatter(fed_pca_sub[:, 0], fed_pca_sub[:, 1],
                        c=local_pr_fed, cmap='viridis', norm=norm,
                        s=5, alpha=0.5, rasterized=True)
        plt.colorbar(sc, ax=ax, label='Local PR', shrink=0.8)

        ax.set_xlabel('PC1', fontsize=11)
        ax.set_ylabel('PC2', fontsize=11)
        ax.set_title('A) Local Dimensionality (Fed)', fontsize=12)

        # Inset: histogram comparing fed vs fasted local dim
        ax_inset = ax.inset_axes([0.55, 0.6, 0.42, 0.35])
        ax_inset.hist(local_pr_fed, bins=25, alpha=0.6, color='#1565C0',
                      label=f'Fed ({local_pr_fed.mean():.1f})', density=True)
        ax_inset.hist(local_pr_fasted, bins=25, alpha=0.6, color='#C62828',
                      label=f'Fasted ({local_pr_fasted.mean():.1f})', density=True)
        ax_inset.set_xlabel('Local PR', fontsize=8)
        ax_inset.set_ylabel('Density', fontsize=8)
        ax_inset.legend(fontsize=7)
        ax_inset.tick_params(labelsize=7)
        # Mann-Whitney on local dims
        u_stat, p_local = sp_stats.mannwhitneyu(local_pr_fed, local_pr_fasted,
                                                 alternative='two-sided')
        ax_inset.set_title(f'p={p_local:.2e}', fontsize=8)

        # --- Panel B: State Density ---
        ax = axes[0, 1]

        # Fed density contours
        xmin_f, xmax_f = h_fed_pca[:, 0].min()-0.5, h_fed_pca[:, 0].max()+0.5
        ymin_f, ymax_f = h_fed_pca[:, 1].min()-0.5, h_fed_pca[:, 1].max()+0.5
        xx_f, yy_f = np.mgrid[xmin_f:xmax_f:80j, ymin_f:ymax_f:80j]
        positions_f = np.vstack([xx_f.ravel(), yy_f.ravel()])
        density_fed = kde_fed(positions_f).reshape(xx_f.shape)

        # Plot fed as filled contours
        ax.contourf(xx_f, yy_f, density_fed, levels=8, cmap='Blues', alpha=0.6)
        ax.contour(xx_f, yy_f, density_fed, levels=5, colors='#1565C0',
                   linewidths=0.8, alpha=0.8)

        # Fasted density in separate inset (different PCA space)
        ax_inset2 = ax.inset_axes([0.55, 0.55, 0.42, 0.42])
        xmin_fs, xmax_fs = h_fasted_pca[:, 0].min()-0.5, h_fasted_pca[:, 0].max()+0.5
        ymin_fs, ymax_fs = h_fasted_pca[:, 1].min()-0.5, h_fasted_pca[:, 1].max()+0.5
        xx_fs, yy_fs = np.mgrid[xmin_fs:xmax_fs:80j, ymin_fs:ymax_fs:80j]
        positions_fs = np.vstack([xx_fs.ravel(), yy_fs.ravel()])
        density_fasted = kde_fasted(positions_fs).reshape(xx_fs.shape)
        ax_inset2.contourf(xx_fs, yy_fs, density_fasted, levels=8, cmap='Reds', alpha=0.6)
        ax_inset2.contour(xx_fs, yy_fs, density_fasted, levels=5, colors='#C62828',
                          linewidths=0.5, alpha=0.8)
        ax_inset2.set_xlabel('PC1', fontsize=7)
        ax_inset2.set_ylabel('PC2', fontsize=7)
        ax_inset2.set_title('Fasted', fontsize=8, color='#C62828')
        ax_inset2.tick_params(labelsize=6)

        # Compute density statistics
        fed_density_at_pts = kde_fed(h_fed_pca[:, :2].T)
        fas_density_at_pts = kde_fasted(h_fasted_pca[:, :2].T)
        fed_entropy = -np.mean(np.log(fed_density_at_pts + 1e-10))
        fas_entropy = -np.mean(np.log(fas_density_at_pts + 1e-10))

        ax.set_xlabel('PC1 (Fed PCA)', fontsize=11)
        ax.set_ylabel('PC2 (Fed PCA)', fontsize=11)
        ax.set_title(f'B) Dwell Time Density\n'
                     f'Entropy: Fed={fed_entropy:.2f}, Fasted={fas_entropy:.2f}',
                     fontsize=12)

        # Add blue "Fed" label
        ax.text(0.05, 0.95, 'Fed', transform=ax.transAxes, fontsize=12,
                color='#1565C0', fontweight='bold', verticalalignment='top')

        # --- Panel C: Manifold Comparison ---
        ax = axes[1, 0]

        # Bar plot of cosines of principal angles
        n_show = 10
        colors_bar = ['#2E7D32' if c > 0.8 else '#E65100' if c > 0.5 else '#C62828'
                      for c in cosines[:n_show]]
        bars = ax.bar(range(n_show), cosines[:n_show], color=colors_bar,
                      edgecolor='black', linewidth=0.5, alpha=0.8)
        ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=0.5)
        ax.axhline(y=0.5, color='red', linestyle=':', linewidth=0.5, alpha=0.5)
        ax.set_xlabel('Principal Angle Index', fontsize=11)
        ax.set_ylabel('cos(angle) -- subspace alignment', fontsize=11)
        ax.set_title(f'C) Fed vs Fasted Subspace Alignment\n'
                     f'Procrustes disparity={disparity:.3f}', fontsize=12)
        ax.set_ylim(0, 1.1)
        ax.set_xticks(range(n_show))
        ax.set_xticklabels([f'PC{i+1}' for i in range(n_show)], fontsize=9)

        # Add text interpretation
        n_aligned = (cosines[:n_show] > 0.8).sum()
        ax.text(0.95, 0.95, f'{n_aligned}/{n_show} PCs aligned\n(cos > 0.8)',
                transform=ax.transAxes, fontsize=10, ha='right', va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        # Inset: Procrustes-aligned overlay
        ax_proc = ax.inset_axes([0.55, 0.05, 0.42, 0.45])
        # fed_proc was normalized by procrustes, recompute for display
        fed_norm = (fed_proc - fed_proc.mean(axis=0)) / fed_proc.std()
        ax_proc.scatter(fed_norm[:, 0], fed_norm[:, 1], c='#1565C0', s=2,
                        alpha=0.2, label='Fed', rasterized=True)
        ax_proc.scatter(fas_aligned[:, 0], fas_aligned[:, 1], c='#C62828', s=2,
                        alpha=0.2, label='Fasted', rasterized=True)
        ax_proc.legend(fontsize=7, markerscale=3)
        ax_proc.set_title('Procrustes overlay', fontsize=8)
        ax_proc.tick_params(labelsize=6)

        # --- Panel D: Flow on Manifold ---
        ax = axes[1, 1]

        # Fed flow field
        h_fed_pca3 = pca3_fed.transform(h_fed)
        n_bg = min(3000, len(h_fed_pca3))
        bg_idx = np.random.choice(len(h_fed_pca3), n_bg, replace=False)
        ax.scatter(h_fed_pca3[bg_idx, 0], h_fed_pca3[bg_idx, 1],
                   c='lightblue', s=1, alpha=0.2, rasterized=True, label='Fed states')

        # Normalize flow for visibility
        max_mag = np.percentile(fspeed, 95)
        if max_mag > 0:
            fu_n = fu / max_mag * 0.8
            fv_n = fv / max_mag * 0.8
        else:
            fu_n, fv_n = fu, fv

        ax.quiver(grid1, grid2, fu_n, fv_n, fspeed, cmap='coolwarm',
                  alpha=0.7, scale=25, zorder=3)

        # Fixed points
        stability_colors = {'stable': '#2E7D32', 'saddle': '#E65100'}
        stability_markers = {'stable': '*', 'saddle': 'D'}
        if len(fps_fed) > 0:
            for k, fp in enumerate(fps_fed):
                fp_p = pca3_fed.transform(fp.reshape(1, -1)).flatten()
                # Quick stability check
                h_t = torch.tensor(fp, dtype=torch.float32, device=DEVICE).unsqueeze(0).requires_grad_(True)
                dhdt = model_fed.ode_func(0, h_t)
                jac = torch.zeros(HIDDEN_SIZE, HIDDEN_SIZE, device=DEVICE)
                for dim in range(HIDDEN_SIZE):
                    if h_t.grad is not None:
                        h_t.grad.zero_()
                    dhdt[0, dim].backward(retain_graph=True)
                    jac[dim] = h_t.grad[0].clone()
                eigs = np.linalg.eigvals(jac.cpu().numpy())
                is_stable = np.max(eigs.real) < 1e-6
                stype = 'stable' if is_stable else 'saddle'
                color = stability_colors[stype]
                marker = stability_markers[stype]
                sz = max(100, min(300, fp_sizes_fed[k] * 20))
                label_fp = stype if k == 0 or (k > 0 and stype != ('stable' if np.max(np.linalg.eigvals(
                    jac.cpu().numpy()).real) < 1e-6 else 'saddle')) else None
                ax.scatter(fp_p[0], fp_p[1], c=color, s=sz, marker=marker,
                           edgecolors='black', linewidths=1.5, zorder=10)

        # Also show fasted flow as lighter overlay
        h_fas_pca3 = pca3_fasted.transform(h_fasted)
        n_bg2 = min(1500, len(h_fas_pca3))
        bg_idx2 = np.random.choice(len(h_fas_pca3), n_bg2, replace=False)

        ax.set_xlabel('PC1', fontsize=11)
        ax.set_ylabel('PC2', fontsize=11)
        ax.set_title('D) Flow Field & Fixed Points (Fed)', fontsize=12)

        # Manual legend
        from matplotlib.lines import Line2D
        legend_elems = [
            Line2D([0], [0], marker='*', color='w', markerfacecolor='#2E7D32',
                   markersize=12, label='Stable FP'),
            Line2D([0], [0], marker='D', color='w', markerfacecolor='#E65100',
                   markersize=10, label='Saddle FP'),
        ]
        ax.legend(handles=legend_elems, fontsize=9, loc='upper right')

        plt.tight_layout(rect=[0, 0, 1, 0.94])
        outpath = Path("figures") / f"gru_ode_manifold_{region}.png"
        fig.savefig(outpath, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {outpath}")

        # Print summary stats
        print(f"\n  --- {region.upper()} Manifold Summary ---")
        print(f"  Local dim: Fed={local_pr_fed.mean():.2f}+/-{local_pr_fed.std():.2f}, "
              f"Fasted={local_pr_fasted.mean():.2f}+/-{local_pr_fasted.std():.2f}, p={p_local:.2e}")
        print(f"  Density entropy: Fed={fed_entropy:.3f}, Fasted={fas_entropy:.3f}")
        print(f"  Subspace alignment (cos): {np.round(cosines[:5], 3)}")
        print(f"  Procrustes disparity: {disparity:.4f}")
        print(f"  Explained var (Fed): {np.round(pca_fed.explained_variance_ratio_[:5]*100, 1)}%")
        print(f"  Explained var (Fasted): {np.round(pca_fasted.explained_variance_ratio_[:5]*100, 1)}%")
        print(f"  Fixed points: Fed={len(fps_fed)}, Fasted={len(fps_fasted)}")

    print("\nDone!")


if __name__ == "__main__":
    main()
