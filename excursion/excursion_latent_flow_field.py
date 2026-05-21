"""
Excursion Flow Fields from GRU-ODE Latent States
=================================================
Uses the trained 10ms Poisson GRU-ODE models to extract hidden states
for each excursion, then evaluates the learned ODE dh/dt to get proper
flow fields in the latent space.

Session 1 = Fed, so uses the fed model.
"""

import yaml
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.cm import ScalarMappable
import spikeinterface.extractors as se
import torch
import torch.nn as nn
from torchdiffeq import odeint
from sklearn.decomposition import PCA
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
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open("paths.yaml") as f:
    paths_config = yaml.safe_load(f)

SESSION_INFO = {
    1: {'state': 'Fed', 'phase': 'Exploration'},
    2: {'state': 'Fed', 'phase': 'Foraging'},
    3: {'state': 'Fed', 'phase': 'Exploration'},
    4: {'state': 'Fed', 'phase': 'Foraging'},
}

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
        """Extract hidden states from batched windowed sequences."""
        sn_key = str(session_num)
        with torch.no_grad():
            h = torch.zeros(x.shape[0], self.hidden_size, device=x.device)
            hidden_seq = []
            for k in range(x.shape[1]):
                h = self._ode_evolve(h)
                x_proj = self.input_projections[sn_key](x[:, k, :])
                h = self.obs_cell(x_proj, h)
                hidden_seq.append(h.unsqueeze(1))
            return torch.cat(hidden_seq, dim=1)


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


def get_zone_labels(behav_dict, exc_mask):
    n = exc_mask.sum()
    labels = np.full(n, 'Other', dtype=object)
    for zone in ['Home', 'Ladder', 'Transition zone', 'Foraging arena']:
        if zone in behav_dict:
            zdata = behav_dict[zone][exc_mask]
            labels[zdata > 0] = zone
    return labels


def evaluate_flow_on_grid(ode_func, pca, grid_pc1, grid_pc2, device):
    """Evaluate learned dh/dt on a PC1-PC2 grid."""
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


def make_excursion_flow_figure(exc_pca, behav_labels, zone_labels,
                                grid_pc1, grid_pc2, U, V, speed_grid,
                                speed_max, pca, eid, region_label,
                                exc_label, duration, n_pts, var_pct):
    """Create 2x3 flow field figure for one excursion using learned ODE."""

    fig, axes = plt.subplots(2, 3, figsize=(22, 13))

    is_feeding = 'feeding' in exc_label.lower()
    is_digging = 'digging' in exc_label.lower()
    type_color = '#D32F2F' if is_feeding else '#FF9800' if is_digging else '#666666'

    fig.suptitle(
        f"{region_label} GRU-ODE Latent Flow Field — Excursion {eid} — Session 1 (Fed)\n"
        f"{exc_label} | {duration:.1f}s, {n_pts} latent states (10ms)",
        fontsize=14, fontweight='bold', color=type_color, y=0.98)

    # Subsample for scatter if too many points
    step = max(1, n_pts // 1500)
    exc_sub = exc_pca[::step]
    beh_sub = behav_labels[::step]
    zone_sub = zone_labels[::step]

    pc1_label = f'PC1 ({var_pct[0]:.1f}%)'
    pc2_label = f'PC2 ({var_pct[1]:.1f}%)'
    xlim = (grid_pc1[0], grid_pc1[-1])
    ylim = (grid_pc2[0], grid_pc2[-1])
    skip = 3

    # ====== TOP ROW: Streamplot + trajectory scatter ======

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
            # Speed in latent space
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

    # ====== BOTTOM ROW: Speed heatmap + quiver ======

    # Panel (1,0): Speed heatmap + behavior quiver
    ax = axes[1, 0]
    im = ax.pcolormesh(
        np.meshgrid(grid_pc1, grid_pc2)[0],
        np.meshgrid(grid_pc1, grid_pc2)[1],
        speed_grid, cmap='magma_r', shading='auto',
        vmin=0, vmax=speed_max)
    plt.colorbar(im, ax=ax, label='|dh/dt|', shrink=0.8)
    slow_thresh = np.percentile(speed_grid, 15)
    PC1g, PC2g = np.meshgrid(grid_pc1, grid_pc2)
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

    # Panel (1,1): Speed heatmap + excursion trajectory overlay
    ax = axes[1, 1]
    im = ax.pcolormesh(PC1g, PC2g, speed_grid, cmap='magma_r', shading='auto',
                       vmin=0, vmax=speed_max)
    plt.colorbar(im, ax=ax, label='|dh/dt|', shrink=0.8)
    ax.contour(PC1g, PC2g, speed_grid, levels=[slow_thresh],
               colors='cyan', linewidths=1.5, linestyles='--')
    # Overlay excursion trajectory
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

    # Panel (1,2): Dwell-time heatmap
    ax = axes[1, 2]
    from scipy.ndimage import gaussian_filter
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
    print(f"Excursion Latent Flow Fields — GRU-ODE 10ms Poisson")
    print(f"Device: {DEVICE}")
    print("=" * 55)

    # Load Session 1 data
    sp = paths_config['single_probe']['coordinates_1']['mouse01']['sessions']
    sc = sp['session_1']
    sorted_path = Path(sc['sorted'])
    sorting = se.read_kilosort(sorted_path)
    lha_ids, rsp_ids = get_good_units_by_region(sorted_path)

    exc_df = pd.read_csv("data/excursions_session_1.csv")
    complete = exc_df[exc_df['label'] == 'complete']

    excursions_to_plot = [
        {'exc_id': 81, 'label': 'Feeding (64% of time)'},
        {'exc_id': 57, 'label': 'Digging (6%) + H2 void'},
        {'exc_id': 43, 'label': 'Similar #2 — Hesitant exploration'},
        {'exc_id': 53, 'label': 'Similar #3 — Other'},
        {'exc_id': 35, 'label': 'Similar #4 — Hesitant exploration'},
        {'exc_id': 45, 'label': 'Similar #5 — Hesitant exploration'},
        {'exc_id': 80, 'label': 'Similar #6 — Other'},
    ]

    for region, unit_ids in [('lha', lha_ids), ('rsp', rsp_ids)]:
        region_label = region.upper()
        print(f"\n{'='*55}")
        print(f"  {region_label} — {len(unit_ids)} neurons")
        print(f"{'='*55}")

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

        # Extract hidden states using windowed batching (stride=1 for dense coverage)
        # For Session 1 we want every time bin's hidden state for excursion slicing
        # Use stride=1 but process in chunks for memory
        STRIDE = 10  # one hidden state per 100ms (10 * 10ms)

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

            # Build windowed sequences
            seqs = []
            for i in range(0, s_bins - SEQ_LEN, STRIDE):
                seqs.append(s_data[i:i + SEQ_LEN])
            seqs_np = np.array(seqs)
            print(f"    {len(seqs_np)} windows")

            # Extract in chunks
            chunk_size = 128
            all_hidden = []
            for start in range(0, len(seqs_np), chunk_size):
                chunk = torch.tensor(seqs_np[start:start + chunk_size],
                                     dtype=torch.float32).to(DEVICE)
                h = model.extract_hidden_states(chunk, sess_num)
                all_hidden.append(h[:, -1, :].cpu().numpy())
                if (start // chunk_size) % 200 == 0 and start > 0:
                    print(f"      {start}/{len(seqs_np)}...")
            all_fed_hidden[sess_num] = np.concatenate(all_hidden, axis=0)
            print(f"    S{sess_num}: {all_fed_hidden[sess_num].shape[0]} hidden states")

            if sess_num == 1:
                time_sec = s_time
                # The hidden states correspond to time bins SEQ_LEN onward
                # hidden_all[i] is the hidden state at time_sec[i + SEQ_LEN]
                n_bins_s1 = s_bins

        hidden_all = all_fed_hidden[1]

        # Fit shared PCA on all fed sessions (subsample others)
        print(f"  Fitting shared PCA...")
        pca_data = [all_fed_hidden[1][::50]]  # S1 subsampled
        for sn in [2, 3, 4]:
            if sn in all_fed_hidden:
                pca_data.append(all_fed_hidden[sn][::50])
        combined_hidden = np.concatenate(pca_data, axis=0)
        pca = PCA(n_components=min(32, HIDDEN_SIZE)).fit(combined_hidden)
        var_pct = pca.explained_variance_ratio_ * 100
        print(f"  Shared PCA: PC1={var_pct[0]:.1f}%, PC2={var_pct[1]:.1f}%")

        # Compute flow field grid from the learned ODE
        all_pca = pca.transform(combined_hidden)
        pad = 0.5
        pc1_min, pc1_max = all_pca[:, 0].min() - pad, all_pca[:, 0].max() + pad
        pc2_min, pc2_max = all_pca[:, 1].min() - pad, all_pca[:, 1].max() + pad

        grid_res = 50
        grid_pc1 = np.linspace(pc1_min, pc1_max, grid_res)
        grid_pc2 = np.linspace(pc2_min, pc2_max, grid_res)

        print(f"  Evaluating learned ODE flow field on grid...")
        PC1, PC2, U, V, speed_grid = evaluate_flow_on_grid(
            model.ode_func, pca, grid_pc1, grid_pc2, DEVICE)
        speed_max = speed_grid.max()
        print(f"    Speed range: {speed_grid.min():.3f} to {speed_max:.3f}")

        # Load behavior for Session 1
        # Hidden states correspond to the end of each window
        # Window i starts at i*STRIDE, ends at i*STRIDE + SEQ_LEN
        # So hidden state i corresponds to time_sec[i*STRIDE + SEQ_LEN - 1]
        hs_indices = [i * STRIDE + SEQ_LEN - 1 for i in range(len(hidden_all))]
        hs_time_sec = time_sec[np.array(hs_indices)]
        behav_dict = load_behavior_timeseries(1, hs_time_sec)

        # Project Session 1 hidden states into PCA
        hidden_pca = pca.transform(hidden_all)

        # Now extract per-excursion
        print(f"\n  Processing excursions...")
        for entry in excursions_to_plot:
            eid = entry['exc_id']
            exc_label = entry['label']

            erow = complete[complete['excursion_id'] == eid]
            if len(erow) == 0:
                continue
            erow = erow.iloc[0]

            # Find time bins within this excursion
            mask = (hs_time_sec >= erow['start_time']) & (hs_time_sec <= erow['end_time'])
            exc_hidden_pca = hidden_pca[mask]
            n_pts = len(exc_hidden_pca)

            if n_pts < 20:
                print(f"    Exc {eid}: only {n_pts} pts, skipping")
                continue

            print(f"    Exc {eid} ({exc_label}): {n_pts} latent states")

            # Behavior labels
            behav_labels = get_dominant_behavior(behav_dict, mask)
            zone_labels = get_zone_labels(behav_dict, mask)

            fig = make_excursion_flow_figure(
                exc_hidden_pca, behav_labels, zone_labels,
                grid_pc1, grid_pc2, U, V, speed_grid, speed_max,
                pca, eid, region_label, exc_label,
                erow['duration'], n_pts, var_pct)

            if 'Feeding' in exc_label:
                tag = 'feeding'
            elif 'Digging' in exc_label:
                tag = 'digging'
            else:
                tag = 'similar'
            outpath = Path("figures") / f"latent_flow_{tag}_exc{eid}_{region}.png"
            fig.savefig(outpath, dpi=200, bbox_inches='tight')
            plt.close()
            print(f"      Saved: {outpath}")

    print("\nDone!")


if __name__ == "__main__":
    main()
