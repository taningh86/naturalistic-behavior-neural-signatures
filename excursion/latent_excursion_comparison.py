"""
Compare ALL complete excursions in Session 1 using GRU-ODE latent trajectories.
Computes dynamical features from latent hidden states, ranks by dissimilarity
to feeding (Exc 81) and digging (Exc 57), then generates flow field plots
for the most different excursions.
"""

import yaml
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import spikeinterface.extractors as se
import torch
import torch.nn as nn
from torchdiffeq import odeint
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import cdist
import warnings

warnings.filterwarnings('ignore')

# Config — must match the 10ms Poisson GRU-ODE training
BIN_SIZE_MS = 10
FS = 30000
BIN_SAMPLES = int(BIN_SIZE_MS * FS / 1000)
SEQ_LEN = 50
PRED_BINS = 10
D_SHARED = 32
HIDDEN_SIZE = 32
ODE_GATE_HIDDEN = 64
ODE_SOLVER = 'rk4'
ODE_STEP_SIZE = 1.0
ODE_DT = 1.0
LHA_DEPTH_MAX = 1300
RSP_DEPTH_MIN = 1300
STRIDE = 10  # one hidden state per 100ms
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open("paths.yaml") as f:
    paths_config = yaml.safe_load(f)


# --- Model classes (must match training exactly) ---
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

    def _ode_evolve(self, h):
        h_evolved = odeint(
            self.ode_func, h, self.t_span,
            method=ODE_SOLVER, options={'step_size': ODE_STEP_SIZE},
        )
        return h_evolved[-1]

    def extract_hidden_states(self, x, session_num):
        sn_key = str(session_num)
        with torch.no_grad():
            h = torch.zeros(x.shape[0], self.hidden_size, device=x.device)
            for k in range(x.shape[1]):
                h = self._ode_evolve(h)
                x_proj = self.input_projections[sn_key](x[:, k, :])
                h = self.obs_cell(x_proj, h)
            return h  # Only return final hidden state per window


# --- Data loading ---
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
    means = data.mean(axis=0, keepdims=True)
    stds = data.std(axis=0, keepdims=True)
    stds[stds < 1e-8] = 1.0
    zscore_data = (data - means) / stds
    time_sec = (np.arange(n_bins) * BIN_SIZE_MS / 1000) + (all_min / FS)
    return zscore_data, time_sec, n_bins


def load_behavior_timeseries(session_num, time_sec):
    sp = paths_config['single_probe']['coordinates_1']['mouse01']['sessions']
    sc = sp[f'session_{session_num}']
    behav_path = sc.get('behavior')
    if not behav_path or not Path(behav_path).exists():
        return {}
    behav_df = pd.read_csv(behav_path, header=None)
    result = {}
    for row_idx in range(behav_df.shape[0]):
        name = str(behav_df.iloc[row_idx, 0]).strip()
        if not name or name == 'nan':
            continue
        row_data = pd.to_numeric(behav_df.iloc[row_idx, 1:], errors='coerce').values
        aligned = np.zeros(len(time_sec))
        for ti, t in enumerate(time_sec):
            bi = int(t / 0.1)
            if 0 <= bi < len(row_data) and not np.isnan(row_data[bi]):
                aligned[ti] = row_data[bi]
        result[name] = aligned
    return result


def get_dominant_behavior(behav_dict, exc_mask):
    n = exc_mask.sum()
    labels = np.full(n, 'Other', dtype=object)
    priority = ['Feeding', 'Digging', 'Grooming', 'Quick arena exploration',
                'Arena wall exploration', 'Transition wall exploration',
                'Hesitant exploration', 'Quick one loop at home']
    for bname in reversed(priority):
        if bname in behav_dict:
            bdata = behav_dict[bname][exc_mask]
            labels[bdata > 0] = bname
    return labels


# --- Latent trajectory features ---
def compute_latent_features(exc_pca, ode_func, pca, device):
    """Compute dynamical features from a GRU-ODE latent trajectory in PCA space."""
    n_pts = len(exc_pca)
    features = {}

    # Basic spatial features
    centroid = exc_pca.mean(axis=0)
    features['centroid_pc1'] = centroid[0]
    features['centroid_pc2'] = centroid[1]
    features['spread_pc1'] = np.std(exc_pca[:, 0])
    features['spread_pc2'] = np.std(exc_pca[:, 1])
    features['pc1_range'] = np.ptp(exc_pca[:, 0])
    features['pc2_range'] = np.ptp(exc_pca[:, 1])

    # Elongation: ratio of spread in PC1 vs PC2
    s1, s2 = features['spread_pc1'], features['spread_pc2']
    features['elongation'] = s1 / s2 if s2 > 1e-8 else 0

    # Excursion's own PCA — participation ratio
    cov = np.cov(exc_pca[:, :min(10, exc_pca.shape[1])].T)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = eigvals[eigvals > 0]
    features['participation_ratio'] = (np.sum(eigvals)**2) / np.sum(eigvals**2) if len(eigvals) > 0 else 0

    # Velocity features (in PCA space)
    diffs = np.diff(exc_pca[:, :2], axis=0)
    speeds = np.sqrt(np.sum(diffs**2, axis=1))
    features['speed_mean'] = np.mean(speeds)
    features['speed_std'] = np.std(speeds)
    features['speed_cv'] = features['speed_std'] / features['speed_mean'] if features['speed_mean'] > 1e-8 else 0
    features['speed_skew'] = float(pd.Series(speeds).skew())
    features['speed_max'] = np.max(speeds)
    features['speed_min'] = np.min(speeds)

    # Tortuosity: path length / straight-line displacement
    path_length = np.sum(speeds)
    displacement = np.sqrt(np.sum((exc_pca[-1, :2] - exc_pca[0, :2])**2))
    features['tortuosity'] = path_length / displacement if displacement > 1e-8 else path_length

    # Flow alignment: how well the trajectory follows the learned flow field
    # Evaluate dh/dt at each trajectory point and compare direction to actual movement
    exc_hidden = pca.inverse_transform(
        np.pad(exc_pca[:, :pca.n_components_],
               ((0, 0), (0, max(0, pca.n_components_ - exc_pca.shape[1]))),
               mode='constant') if exc_pca.shape[1] < pca.n_components_ else exc_pca[:, :pca.n_components_])
    h_tensor = torch.tensor(exc_hidden[:-1], dtype=torch.float32).to(device)
    with torch.no_grad():
        dhdt = ode_func(0.0, h_tensor).cpu().numpy()
    dhdt_pca = dhdt @ pca.components_[:2].T  # project to PC1-PC2
    # Normalize both vectors
    dhdt_norm = dhdt_pca / (np.linalg.norm(dhdt_pca, axis=1, keepdims=True) + 1e-8)
    diffs_norm = diffs / (np.linalg.norm(diffs, axis=1, keepdims=True) + 1e-8)
    cos_sim = np.sum(dhdt_norm * diffs_norm, axis=1)
    features['flow_alignment_mean'] = np.mean(cos_sim)
    features['flow_alignment_std'] = np.std(cos_sim)

    # Convergence: average dot product of flow with vector toward centroid
    vecs_to_center = centroid[:2] - exc_pca[:-1, :2]
    vecs_to_center_norm = vecs_to_center / (np.linalg.norm(vecs_to_center, axis=1, keepdims=True) + 1e-8)
    convergence = np.sum(dhdt_norm * vecs_to_center_norm, axis=1)
    features['convergence_mean'] = np.mean(convergence)

    # Dwell-time features (how concentrated is occupancy in PC1-PC2?)
    n_grid = 30
    heatmap, _, _ = np.histogram2d(
        exc_pca[:, 0], exc_pca[:, 1], bins=n_grid)
    heatmap_flat = heatmap.ravel()
    heatmap_flat = heatmap_flat[heatmap_flat > 0]
    features['dwell_entropy'] = float(-np.sum(
        (heatmap_flat / heatmap_flat.sum()) * np.log(heatmap_flat / heatmap_flat.sum() + 1e-10)))
    features['dwell_max_over_mean'] = np.max(heatmap_flat) / np.mean(heatmap_flat) if len(heatmap_flat) > 0 else 0
    features['occupancy_fraction'] = len(heatmap_flat) / (n_grid * n_grid)

    # Recurrence: fraction of point-pairs within a threshold distance
    if n_pts > 200:
        idx = np.random.choice(n_pts, 200, replace=False)
        sub = exc_pca[idx, :2]
    else:
        sub = exc_pca[:, :2]
    dmat = cdist(sub, sub)
    thresh = np.median(dmat) * 0.3
    features['recurrence_rate'] = np.mean(dmat < thresh) - (1.0 / len(sub))

    # Attractor proximity: mean distance from trajectory to the global flow field's
    # slowest point (potential attractor)
    # We'll compute this externally where we have the flow field grid

    return features


def evaluate_flow_on_grid(ode_func, pca, grid_pc1, grid_pc2, device):
    n1, n2 = len(grid_pc1), len(grid_pc2)
    PC1, PC2 = np.meshgrid(grid_pc1, grid_pc2)
    points_pca = np.zeros((n1 * n2, pca.n_components_))
    points_pca[:, 0] = PC1.ravel()
    points_pca[:, 1] = PC2.ravel()
    points_hidden = pca.inverse_transform(points_pca)
    h_tensor = torch.tensor(points_hidden, dtype=torch.float32).to(device)
    with torch.no_grad():
        dhdt = ode_func(0.0, h_tensor).cpu().numpy()
    dhdt_pca = dhdt @ pca.components_.T
    U = dhdt_pca[:, 0].reshape(n2, n1)
    V = dhdt_pca[:, 1].reshape(n2, n1)
    speed = np.sqrt(U**2 + V**2)
    return PC1, PC2, U, V, speed


BEHAVIOR_COLORS = {
    'Feeding': '#D32F2F',
    'Digging': '#FF9800',
    'Grooming': '#4CAF50',
    'Quick arena exploration': '#00BCD4',
    'Arena wall exploration': '#9C27B0',
    'Transition wall exploration': '#2196F3',
    'Hesitant exploration': '#795548',
    'Quick one loop at home': '#E91E63',
}

ZONE_COLORS = {
    'Home': '#4CAF50',
    'Ladder': '#FF9800',
    'Transition zone': '#9C27B0',
    'Foraging arena': '#D32F2F',
}


def get_zone_labels(behav_dict, exc_mask):
    n = exc_mask.sum()
    labels = np.full(n, 'Other', dtype=object)
    for zone in ['Home', 'Ladder', 'Transition zone', 'Foraging arena']:
        if zone in behav_dict:
            zdata = behav_dict[zone][exc_mask]
            labels[zdata > 0] = zone
    return labels


def make_excursion_flow_figure(exc_pca, behav_labels, zone_labels,
                                grid_pc1, grid_pc2, U, V, speed_grid,
                                speed_max, pca, eid, region_label,
                                exc_label, duration, n_pts, var_pct):
    fig, axes = plt.subplots(2, 3, figsize=(22, 13))

    is_feeding = 'feeding' in exc_label.lower()
    is_digging = 'digging' in exc_label.lower()
    type_color = '#D32F2F' if is_feeding else '#FF9800' if is_digging else '#666666'

    fig.suptitle(
        f"{region_label} GRU-ODE Latent Flow Field — Excursion {eid} — Session 1 (Fed)\n"
        f"{exc_label} | {duration:.1f}s, {n_pts} latent states (10ms)",
        fontsize=14, fontweight='bold', color=type_color, y=0.98)

    step = max(1, n_pts // 1500)
    exc_sub = exc_pca[::step]
    beh_sub = behav_labels[::step]
    zone_sub = zone_labels[::step]

    pc1_label = f'PC1 ({var_pct[0]:.1f}%)'
    pc2_label = f'PC2 ({var_pct[1]:.1f}%)'
    xlim = (grid_pc1[0], grid_pc1[-1])
    ylim = (grid_pc2[0], grid_pc2[-1])
    skip = 3

    # TOP ROW: Streamplot + trajectory scatter
    for col, (title, color_by) in enumerate([
        ('Behavior', 'behavior'), ('Speed', 'speed'), ('Zone', 'zone')
    ]):
        ax = axes[0, col]
        ax.streamplot(grid_pc1, grid_pc2, U, V,
                      color=speed_grid, cmap='Greys', linewidth=0.8,
                      arrowsize=1.0, density=1.8, arrowstyle='->',
                      norm=mcolors.Normalize(vmin=0, vmax=speed_max * 0.8))

        if color_by == 'behavior':
            for bname, color in BEHAVIOR_COLORS.items():
                mask = beh_sub == bname
                if mask.any():
                    ax.scatter(exc_sub[mask, 0], exc_sub[mask, 1],
                               c=color, s=8, alpha=0.6, label=bname,
                               zorder=5, rasterized=True)
            other = beh_sub == 'Other'
            if other.any():
                ax.scatter(exc_sub[other, 0], exc_sub[other, 1],
                           c='#BDBDBD', s=3, alpha=0.15, zorder=2, rasterized=True)
            ax.legend(fontsize=6, loc='best', markerscale=2)

        elif color_by == 'speed':
            diffs = np.diff(exc_pca, axis=0)
            spd = np.sqrt(np.sum(diffs**2, axis=1))
            spd_sub = spd[::step][:len(exc_sub)-1]
            snorm = mcolors.Normalize(vmin=np.percentile(spd, 5),
                                      vmax=np.percentile(spd, 95))
            sc = ax.scatter(exc_sub[:-1, 0][::1][:len(spd_sub)],
                           exc_sub[:-1, 1][::1][:len(spd_sub)],
                           c=spd_sub, cmap='hot_r', norm=snorm, s=6,
                           alpha=0.6, zorder=5, rasterized=True)
            plt.colorbar(sc, ax=ax, label='Latent speed', shrink=0.8)

        elif color_by == 'zone':
            for zone, color in ZONE_COLORS.items():
                zmask = zone_sub == zone
                if zmask.any():
                    ax.scatter(exc_sub[zmask, 0], exc_sub[zmask, 1],
                               c=color, s=8, alpha=0.6, label=zone,
                               zorder=5, rasterized=True)
            ax.legend(fontsize=7, loc='best')

        ax.scatter(exc_pca[0, 0], exc_pca[0, 1], c='lime', s=200, marker='*',
                   edgecolors='black', linewidths=1.5, zorder=10)
        ax.scatter(exc_pca[-1, 0], exc_pca[-1, 1], c='red', s=150, marker='X',
                   edgecolors='black', linewidths=1.5, zorder=10)
        ax.set_xlabel(pc1_label, fontsize=11)
        ax.set_ylabel(pc2_label, fontsize=11)
        ax.set_title(f'Flow Field + {title}', fontsize=12)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    # BOTTOM ROW: Speed heatmap + quiver
    ax = axes[1, 0]
    PC1g, PC2g = np.meshgrid(grid_pc1, grid_pc2)
    im = ax.pcolormesh(PC1g, PC2g, speed_grid, cmap='magma_r', shading='auto',
                       vmin=0, vmax=speed_max)
    plt.colorbar(im, ax=ax, label='|dh/dt|', shrink=0.8)
    slow_thresh = np.percentile(speed_grid, 15)
    ax.contour(PC1g, PC2g, speed_grid, levels=[slow_thresh],
               colors='cyan', linewidths=1.5, linestyles='--')
    ax.quiver(PC1g[::skip, ::skip], PC2g[::skip, ::skip],
              U[::skip, ::skip], V[::skip, ::skip],
              color='white', alpha=0.5, scale=None, width=0.003)
    ax.set_xlabel(pc1_label, fontsize=11)
    ax.set_ylabel(pc2_label, fontsize=11)
    ax.set_title('Flow Speed + Direction\n(cyan = slow/attractor)', fontsize=11)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax = axes[1, 1]
    im = ax.pcolormesh(PC1g, PC2g, speed_grid, cmap='magma_r', shading='auto',
                       vmin=0, vmax=speed_max)
    plt.colorbar(im, ax=ax, label='|dh/dt|', shrink=0.8)
    ax.contour(PC1g, PC2g, speed_grid, levels=[slow_thresh],
               colors='cyan', linewidths=1.5, linestyles='--')
    ax.plot(exc_pca[::step, 0], exc_pca[::step, 1],
            color='white', linewidth=0.3, alpha=0.5)
    for bname, color in BEHAVIOR_COLORS.items():
        mask = beh_sub == bname
        if mask.any():
            ax.scatter(exc_sub[mask, 0], exc_sub[mask, 1],
                       c=color, s=8, alpha=0.7, zorder=5, rasterized=True)
    ax.scatter(exc_pca[0, 0], exc_pca[0, 1], c='lime', s=200, marker='*',
               edgecolors='white', linewidths=1.5, zorder=10)
    ax.scatter(exc_pca[-1, 0], exc_pca[-1, 1], c='red', s=150, marker='X',
               edgecolors='white', linewidths=1.5, zorder=10)
    ax.set_xlabel(pc1_label, fontsize=11)
    ax.set_ylabel(pc2_label, fontsize=11)
    ax.set_title('Flow Speed + Excursion Trajectory\n(colored by behavior)', fontsize=11)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax = axes[1, 2]
    n_grid = 50
    heatmap, xedges, yedges = np.histogram2d(
        exc_pca[:, 0], exc_pca[:, 1], bins=n_grid,
        range=[[xlim[0], xlim[1]], [ylim[0], ylim[1]]])
    heatmap = gaussian_filter(heatmap.T, sigma=2)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    im = ax.imshow(heatmap, origin='lower', extent=extent, cmap='inferno',
                   aspect='auto', interpolation='bilinear')
    plt.colorbar(im, ax=ax, label='Dwell time (bins)', shrink=0.8)
    ax.plot(exc_pca[::step, 0], exc_pca[::step, 1],
            color='white', linewidth=0.2, alpha=0.3)
    ax.scatter(exc_pca[0, 0], exc_pca[0, 1], c='lime', s=150, marker='*',
               edgecolors='white', linewidths=1, zorder=10)
    ax.scatter(exc_pca[-1, 0], exc_pca[-1, 1], c='red', s=120, marker='X',
               edgecolors='white', linewidths=1, zorder=10)
    ax.set_xlabel(pc1_label, fontsize=11)
    ax.set_ylabel(pc2_label, fontsize=11)
    ax.set_title('Dwell-Time Heatmap\n(bright = attractor region)', fontsize=11)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


# =============================================================================
# MAIN
# =============================================================================

def main():
    print(f"Latent Excursion Comparison — GRU-ODE 10ms Poisson")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    sp = paths_config['single_probe']['coordinates_1']['mouse01']['sessions']
    sc = sp['session_1']
    sorted_path = Path(sc['sorted'])
    sorting = se.read_kilosort(sorted_path)
    lha_ids, rsp_ids = get_good_units_by_region(sorted_path)

    exc_df = pd.read_csv("data/excursions_session_1.csv")
    complete = exc_df[exc_df['label'] == 'complete']

    REFERENCE_EXCURSIONS = [81, 57]  # feeding, digging
    MIN_LATENT_POINTS = 30  # skip very short excursions

    for region, unit_ids in [('lha', lha_ids), ('rsp', rsp_ids)]:
        region_label = region.upper()
        print(f"\n{'='*60}")
        print(f"  {region_label} — {len(unit_ids)} neurons")
        print(f"{'='*60}")

        # Load trained fed model
        model_path = Path("data") / f"gru_ode_10ms_poisson_{region}_fed_model.pt"
        if not model_path.exists():
            print(f"  Model not found: {model_path}")
            continue

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
        print(f"  Loaded model: {model_path.name}")

        # Extract hidden states for all fed sessions (shared PCA)
        all_fed_hidden = {}
        for sess_num in [1, 2, 3, 4]:
            sc_sess = sp[f'session_{sess_num}']
            sp_sess = Path(sc_sess['sorted'])
            if not sp_sess.exists():
                continue
            sorting_sess = se.read_kilosort(sp_sess)
            if region == 'lha':
                s_ids = get_good_units_by_region(sp_sess)[0]
            else:
                s_ids = get_good_units_by_region(sp_sess)[1]
            if len(s_ids) < 3:
                continue
            s_data, s_time, s_bins = bin_spike_trains(sorting_sess, s_ids)

            print(f"  Extracting S{sess_num} hidden states ({s_bins} bins)...")
            seqs = []
            for i in range(0, s_bins - SEQ_LEN, STRIDE):
                seqs.append(s_data[i:i + SEQ_LEN])
            seqs_np = np.array(seqs)

            chunk_size = 128
            all_hidden = []
            for start in range(0, len(seqs_np), chunk_size):
                chunk = torch.tensor(seqs_np[start:start + chunk_size],
                                     dtype=torch.float32).to(DEVICE)
                h = model.extract_hidden_states(chunk, sess_num)
                all_hidden.append(h.cpu().numpy())
            all_fed_hidden[sess_num] = np.concatenate(all_hidden, axis=0)
            print(f"    S{sess_num}: {all_fed_hidden[sess_num].shape[0]} hidden states")

            if sess_num == 1:
                time_sec = s_time

        hidden_all = all_fed_hidden[1]

        # Shared PCA
        print(f"  Fitting shared PCA...")
        pca_data = [all_fed_hidden[1][::50]]
        for sn in [2, 3, 4]:
            if sn in all_fed_hidden:
                pca_data.append(all_fed_hidden[sn][::50])
        combined_hidden = np.concatenate(pca_data, axis=0)
        pca = PCA(n_components=min(32, HIDDEN_SIZE)).fit(combined_hidden)
        var_pct = pca.explained_variance_ratio_ * 100
        print(f"  Shared PCA: PC1={var_pct[0]:.1f}%, PC2={var_pct[1]:.1f}%")

        # Flow field grid
        all_pca = pca.transform(combined_hidden)
        pad = 0.5
        pc1_min, pc1_max = all_pca[:, 0].min() - pad, all_pca[:, 0].max() + pad
        pc2_min, pc2_max = all_pca[:, 1].min() - pad, all_pca[:, 1].max() + pad
        grid_res = 50
        grid_pc1 = np.linspace(pc1_min, pc1_max, grid_res)
        grid_pc2 = np.linspace(pc2_min, pc2_max, grid_res)

        print(f"  Evaluating learned ODE flow field...")
        PC1, PC2, U, V, speed_grid = evaluate_flow_on_grid(
            model.ode_func, pca, grid_pc1, grid_pc2, DEVICE)
        speed_max = speed_grid.max()

        # Hidden state time alignment
        hs_indices = [i * STRIDE + SEQ_LEN - 1 for i in range(len(hidden_all))]
        hs_time_sec = time_sec[np.array(hs_indices)]
        behav_dict = load_behavior_timeseries(1, hs_time_sec)
        hidden_pca = pca.transform(hidden_all)

        # ===== Compute features for ALL complete excursions =====
        print(f"\n  Computing latent features for all {len(complete)} excursions...")
        all_features = []

        for _, erow in complete.iterrows():
            eid = int(erow['excursion_id'])
            mask = (hs_time_sec >= erow['start_time']) & (hs_time_sec <= erow['end_time'])
            exc_hidden_pca = hidden_pca[mask]
            n_pts = mask.sum()

            if n_pts < MIN_LATENT_POINTS:
                continue

            feats = compute_latent_features(exc_hidden_pca, model.ode_func, pca, DEVICE)
            feats['excursion_id'] = eid
            feats['duration'] = erow['duration']
            feats['n_latent_pts'] = n_pts
            feats['farthest_zone'] = erow['farthest_zone']

            # Dominant behavior
            beh_labels = get_dominant_behavior(behav_dict, mask)
            behavior_counts = pd.Series(beh_labels).value_counts()
            feats['dominant_behavior'] = behavior_counts.index[0]
            feats['dominant_behavior_pct'] = behavior_counts.iloc[0] / n_pts * 100

            all_features.append(feats)

        feat_df = pd.DataFrame(all_features)
        print(f"  {len(feat_df)} excursions with enough latent points")

        # ===== Compute dissimilarity to reference excursions =====
        feature_cols = [c for c in feat_df.columns if c not in
                       ['excursion_id', 'duration', 'n_latent_pts', 'farthest_zone',
                        'dominant_behavior', 'dominant_behavior_pct']]

        # Z-score features
        feat_z = feat_df[feature_cols].copy()
        for col in feature_cols:
            mu, sig = feat_z[col].mean(), feat_z[col].std()
            if sig > 1e-8:
                feat_z[col] = (feat_z[col] - mu) / sig
            else:
                feat_z[col] = 0

        # Distance from each excursion to Exc 81 and Exc 57
        for ref_eid in REFERENCE_EXCURSIONS:
            ref_idx = feat_df[feat_df['excursion_id'] == ref_eid].index
            if len(ref_idx) == 0:
                continue
            ref_idx = ref_idx[0]
            ref_vec = feat_z.loc[ref_idx].values.reshape(1, -1)
            dists = cdist(feat_z.values, ref_vec, metric='euclidean').ravel()
            feat_df[f'dist_to_exc{ref_eid}'] = dists

        # Combined distance (mean distance to both references)
        if 'dist_to_exc81' in feat_df.columns and 'dist_to_exc57' in feat_df.columns:
            feat_df['dist_combined'] = (feat_df['dist_to_exc81'] + feat_df['dist_to_exc57']) / 2

        # Save features
        out_csv = Path("data") / f"latent_excursion_features_{region}.csv"
        feat_df.to_csv(out_csv, index=False, float_format='%.4f')
        print(f"  Saved: {out_csv}")

        # ===== Print ranking =====
        print(f"\n  {'='*80}")
        print(f"  {region_label} — Excursion ranking by dissimilarity to Feeding (81) + Digging (57)")
        print(f"  {'='*80}")
        if 'dist_combined' in feat_df.columns:
            ranked = feat_df.sort_values('dist_combined', ascending=False)
            print(f"  {'ExcID':<7} {'Dur(s)':<8} {'Behavior':<25} {'Zone':<18} "
                  f"{'DistTo81':<10} {'DistTo57':<10} {'Combined':<10}")
            print(f"  {'-'*90}")
            for _, row in ranked.iterrows():
                eid = int(row['excursion_id'])
                marker = " <-- REF" if eid in REFERENCE_EXCURSIONS else ""
                print(f"  {eid:<7} {row['duration']:<8.1f} {row['dominant_behavior']:<25} "
                      f"{row['farthest_zone']:<18} "
                      f"{row.get('dist_to_exc81', 0):<10.2f} "
                      f"{row.get('dist_to_exc57', 0):<10.2f} "
                      f"{row['dist_combined']:<10.2f}{marker}")

        # ===== Plot flow fields for top 5 most different excursions =====
        if 'dist_combined' not in feat_df.columns:
            continue

        top_diff = feat_df.sort_values('dist_combined', ascending=False).head(8)
        # Exclude the reference excursions from plotting (we already have those)
        top_diff = top_diff[~top_diff['excursion_id'].isin(REFERENCE_EXCURSIONS)]
        top_diff = top_diff.head(5)
        already_plotted = [81, 57, 43, 53, 35, 45, 80]
        top_diff = top_diff[~top_diff['excursion_id'].isin(already_plotted)]

        print(f"\n  Generating flow field plots for top outlier excursions...")
        for rank, (_, row) in enumerate(top_diff.iterrows(), 1):
            eid = int(row['excursion_id'])
            erow = complete[complete['excursion_id'] == eid].iloc[0]

            mask = (hs_time_sec >= erow['start_time']) & (hs_time_sec <= erow['end_time'])
            exc_hidden_pca = hidden_pca[mask]
            n_pts = len(exc_hidden_pca)

            behav_labels = get_dominant_behavior(behav_dict, mask)
            zone_labels = get_zone_labels(behav_dict, mask)

            exc_label = f"Outlier #{rank} — {row['dominant_behavior']} (dist={row['dist_combined']:.1f})"

            fig = make_excursion_flow_figure(
                exc_hidden_pca, behav_labels, zone_labels,
                grid_pc1, grid_pc2, U, V, speed_grid, speed_max,
                pca, eid, region_label, exc_label,
                erow['duration'], n_pts, var_pct)

            outpath = Path("figures") / f"latent_flow_outlier_exc{eid}_{region}.png"
            fig.savefig(outpath, dpi=200, bbox_inches='tight')
            plt.close()
            print(f"    Saved: {outpath}")

        # ===== Summary scatter: all excursions in feature space =====
        print(f"\n  Creating summary scatter plot...")
        fig, axes = plt.subplots(2, 3, figsize=(22, 14))
        fig.suptitle(f"{region_label} — All Excursions in GRU-ODE Latent Feature Space\n"
                     f"(colored by dissimilarity to Feeding Exc 81 + Digging Exc 57)",
                     fontsize=14, fontweight='bold')

        # Panel 1: Centroid positions colored by combined distance
        ax = axes[0, 0]
        sc = ax.scatter(feat_df['centroid_pc1'], feat_df['centroid_pc2'],
                       c=feat_df['dist_combined'], cmap='RdYlBu_r', s=60,
                       edgecolors='black', linewidths=0.5)
        for ref_eid, marker, color in [(81, '*', 'red'), (57, 'D', 'orange')]:
            ref = feat_df[feat_df['excursion_id'] == ref_eid]
            if len(ref) > 0:
                ax.scatter(ref['centroid_pc1'], ref['centroid_pc2'],
                          c=color, s=200, marker=marker, edgecolors='black',
                          linewidths=2, zorder=10, label=f'Exc {ref_eid}')
        # Label top outliers
        for _, row in top_diff.iterrows():
            ax.annotate(f"{int(row['excursion_id'])}",
                       (row['centroid_pc1'], row['centroid_pc2']),
                       fontsize=8, fontweight='bold', ha='center', va='bottom',
                       xytext=(0, 5), textcoords='offset points')
        plt.colorbar(sc, ax=ax, label='Combined distance')
        ax.set_xlabel('Centroid PC1')
        ax.set_ylabel('Centroid PC2')
        ax.set_title('Trajectory Centroids')
        ax.legend()

        # Panel 2: Speed mean vs speed CV
        ax = axes[0, 1]
        sc = ax.scatter(feat_df['speed_mean'], feat_df['speed_cv'],
                       c=feat_df['dist_combined'], cmap='RdYlBu_r', s=60,
                       edgecolors='black', linewidths=0.5)
        for ref_eid, marker, color in [(81, '*', 'red'), (57, 'D', 'orange')]:
            ref = feat_df[feat_df['excursion_id'] == ref_eid]
            if len(ref) > 0:
                ax.scatter(ref['speed_mean'], ref['speed_cv'],
                          c=color, s=200, marker=marker, edgecolors='black',
                          linewidths=2, zorder=10, label=f'Exc {ref_eid}')
        for _, row in top_diff.iterrows():
            ax.annotate(f"{int(row['excursion_id'])}",
                       (row['speed_mean'], row['speed_cv']),
                       fontsize=8, fontweight='bold', ha='center', va='bottom',
                       xytext=(0, 5), textcoords='offset points')
        plt.colorbar(sc, ax=ax, label='Combined distance')
        ax.set_xlabel('Mean latent speed')
        ax.set_ylabel('Speed CV')
        ax.set_title('Speed Statistics')
        ax.legend()

        # Panel 3: Tortuosity vs flow alignment
        ax = axes[0, 2]
        sc = ax.scatter(feat_df['tortuosity'], feat_df['flow_alignment_mean'],
                       c=feat_df['dist_combined'], cmap='RdYlBu_r', s=60,
                       edgecolors='black', linewidths=0.5)
        for ref_eid, marker, color in [(81, '*', 'red'), (57, 'D', 'orange')]:
            ref = feat_df[feat_df['excursion_id'] == ref_eid]
            if len(ref) > 0:
                ax.scatter(ref['tortuosity'], ref['flow_alignment_mean'],
                          c=color, s=200, marker=marker, edgecolors='black',
                          linewidths=2, zorder=10, label=f'Exc {ref_eid}')
        for _, row in top_diff.iterrows():
            ax.annotate(f"{int(row['excursion_id'])}",
                       (row['tortuosity'], row['flow_alignment_mean']),
                       fontsize=8, fontweight='bold', ha='center', va='bottom',
                       xytext=(0, 5), textcoords='offset points')
        plt.colorbar(sc, ax=ax, label='Combined distance')
        ax.set_xlabel('Tortuosity')
        ax.set_ylabel('Flow alignment (cos sim)')
        ax.set_title('Trajectory Shape vs Flow Compliance')
        ax.legend()

        # Panel 4: Participation ratio vs dwell entropy
        ax = axes[1, 0]
        sc = ax.scatter(feat_df['participation_ratio'], feat_df['dwell_entropy'],
                       c=feat_df['dist_combined'], cmap='RdYlBu_r', s=60,
                       edgecolors='black', linewidths=0.5)
        for ref_eid, marker, color in [(81, '*', 'red'), (57, 'D', 'orange')]:
            ref = feat_df[feat_df['excursion_id'] == ref_eid]
            if len(ref) > 0:
                ax.scatter(ref['participation_ratio'], ref['dwell_entropy'],
                          c=color, s=200, marker=marker, edgecolors='black',
                          linewidths=2, zorder=10, label=f'Exc {ref_eid}')
        for _, row in top_diff.iterrows():
            ax.annotate(f"{int(row['excursion_id'])}",
                       (row['participation_ratio'], row['dwell_entropy']),
                       fontsize=8, fontweight='bold', ha='center', va='bottom',
                       xytext=(0, 5), textcoords='offset points')
        plt.colorbar(sc, ax=ax, label='Combined distance')
        ax.set_xlabel('Participation ratio')
        ax.set_ylabel('Dwell entropy')
        ax.set_title('Dimensionality vs Occupancy Spread')
        ax.legend()

        # Panel 5: Convergence vs recurrence rate
        ax = axes[1, 1]
        sc = ax.scatter(feat_df['convergence_mean'], feat_df['recurrence_rate'],
                       c=feat_df['dist_combined'], cmap='RdYlBu_r', s=60,
                       edgecolors='black', linewidths=0.5)
        for ref_eid, marker, color in [(81, '*', 'red'), (57, 'D', 'orange')]:
            ref = feat_df[feat_df['excursion_id'] == ref_eid]
            if len(ref) > 0:
                ax.scatter(ref['convergence_mean'], ref['recurrence_rate'],
                          c=color, s=200, marker=marker, edgecolors='black',
                          linewidths=2, zorder=10, label=f'Exc {ref_eid}')
        for _, row in top_diff.iterrows():
            ax.annotate(f"{int(row['excursion_id'])}",
                       (row['convergence_mean'], row['recurrence_rate']),
                       fontsize=8, fontweight='bold', ha='center', va='bottom',
                       xytext=(0, 5), textcoords='offset points')
        plt.colorbar(sc, ax=ax, label='Combined distance')
        ax.set_xlabel('Flow convergence')
        ax.set_ylabel('Recurrence rate')
        ax.set_title('Attractor Behavior')
        ax.legend()

        # Panel 6: Distance to Exc 81 vs distance to Exc 57
        ax = axes[1, 2]
        sc = ax.scatter(feat_df['dist_to_exc81'], feat_df['dist_to_exc57'],
                       c=feat_df['duration'], cmap='viridis', s=60,
                       edgecolors='black', linewidths=0.5)
        for ref_eid, marker, color in [(81, '*', 'red'), (57, 'D', 'orange')]:
            ref = feat_df[feat_df['excursion_id'] == ref_eid]
            if len(ref) > 0:
                ax.scatter(ref['dist_to_exc81'], ref['dist_to_exc57'],
                          c=color, s=200, marker=marker, edgecolors='black',
                          linewidths=2, zorder=10, label=f'Exc {ref_eid}')
        for _, row in top_diff.iterrows():
            ax.annotate(f"{int(row['excursion_id'])}",
                       (row['dist_to_exc81'], row['dist_to_exc57']),
                       fontsize=8, fontweight='bold', ha='center', va='bottom',
                       xytext=(0, 5), textcoords='offset points')
        plt.colorbar(sc, ax=ax, label='Duration (s)')
        ax.set_xlabel('Distance to Exc 81 (Feeding)')
        ax.set_ylabel('Distance to Exc 57 (Digging)')
        ax.set_title('Dissimilarity Space')
        ax.plot([0, ax.get_xlim()[1]], [0, ax.get_ylim()[1]], 'k--', alpha=0.3)
        ax.legend()

        plt.tight_layout(rect=[0, 0, 1, 0.94])
        outpath = Path("figures") / f"latent_excursion_comparison_{region}.png"
        fig.savefig(outpath, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {outpath}")

    print("\nDone!")


if __name__ == "__main__":
    main()
