"""
Loop Visualization for Excursions with Prominent Topology
==========================================================
Three complementary views per excursion:
  1. Smooth tube trajectory with direction arrows (3D + 2D)
  2. Circular coordinate projection (theta vs time)
  3. Recurrence distance matrix

Uses persistent cohomology to extract circular coordinates.
"""

import yaml
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, BoundaryNorm
from matplotlib.cm import ScalarMappable
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from scipy.interpolate import make_interp_spline
from scipy.ndimage import uniform_filter1d
from scipy.spatial.distance import pdist, squareform
from gtda.homology import VietorisRipsPersistence
import spikeinterface.extractors as se
import warnings

warnings.filterwarnings('ignore')

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

# =============================================================================
# CONFIG
# =============================================================================

FS = 30000
LHA_DEPTH_MAX = 1300
RSP_DEPTH_MIN = 1300

with open("paths.yaml") as f:
    paths_config = yaml.safe_load(f)

PROMINENT_EXCURSIONS = [
    # RSP H1 loops
    {'exc_id': 90, 'region': 'rsp', 'bin_ms': 200, 'smooth_ms': 500,
     'feature': 'H1 loop', 'gap': 14.8, 'note': 'Strongest loop in dataset'},
    {'exc_id': 89, 'region': 'rsp', 'bin_ms': 500, 'smooth_ms': 1000,
     'feature': 'H1 loop', 'gap': 6.7, 'note': 'Loop in both LHA+RSP'},
    {'exc_id': 71, 'region': 'rsp', 'bin_ms': 200, 'smooth_ms': 500,
     'feature': 'H1 loop', 'gap': 6.5, 'note': 'Strong loop'},
    {'exc_id': 5, 'region': 'rsp', 'bin_ms': 500, 'smooth_ms': 1000,
     'feature': 'H1 loop', 'gap': 5.5, 'note': 'Early excursion loop'},
    # RSP H2 voids
    {'exc_id': 57, 'region': 'rsp', 'bin_ms': 500, 'smooth_ms': 1000,
     'feature': 'H2 void', 'gap': 16.9, 'note': 'Strongest void — also has loops'},
    {'exc_id': 58, 'region': 'rsp', 'bin_ms': 500, 'smooth_ms': 1000,
     'feature': 'H2 void', 'gap': 6.5, 'note': 'Strong void'},
    # LHA H1 loops
    {'exc_id': 11, 'region': 'lha', 'bin_ms': 500, 'smooth_ms': 1000,
     'feature': 'H1 loop', 'gap': 4.3, 'note': 'Strongest LHA loop'},
    {'exc_id': 89, 'region': 'lha', 'bin_ms': 500, 'smooth_ms': 1000,
     'feature': 'H1 loop', 'gap': 3.4, 'note': 'Loop in both LHA+RSP'},
]

DIM_COLORS = {0: '#1976D2', 1: '#D32F2F', 2: '#4CAF50'}

ZONE_COLORS = {
    'Home': '#4CAF50',
    'Ladder': '#FF9800',
    'Transition Zone': '#9C27B0',
    'Foraging Arena': '#D32F2F',
}


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


def bin_and_smooth(sorting, unit_ids, bin_ms, smooth_ms):
    bin_samples = int(bin_ms * FS / 1000)
    smooth_bins = max(1, int(smooth_ms / bin_ms))
    spike_trains = {}
    all_min, all_max = np.inf, 0
    for uid in unit_ids:
        st = sorting.get_unit_spike_train(uid)
        spike_trains[uid] = st
        if len(st) > 0:
            all_min = min(all_min, np.min(st))
            all_max = max(all_max, np.max(st))
    n_bins = int((all_max - all_min) / bin_samples) + 1
    data = np.zeros((n_bins, len(unit_ids)), dtype=np.float32)
    for i, uid in enumerate(unit_ids):
        st = spike_trains[uid]
        if len(st) > 0:
            b = ((st - all_min) // bin_samples).astype(int)
            b = b[(b >= 0) & (b < n_bins)]
            np.add.at(data[:, i], b, 1)
    if smooth_bins > 1:
        for i in range(data.shape[1]):
            data[:, i] = uniform_filter1d(data[:, i], size=smooth_bins, mode='constant')
    means = data.mean(axis=0, keepdims=True)
    stds = data.std(axis=0, keepdims=True)
    stds[stds < 1e-8] = 1.0
    zscore_data = (data - means) / stds
    time_sec = (np.arange(n_bins) * bin_ms / 1000) + (all_min / FS)
    return zscore_data, time_sec


def load_behavior_zones(session_num, time_sec):
    sp = paths_config['single_probe']['coordinates_1']['mouse01']['sessions']
    sc = sp[f'session_{session_num}']
    behav_path = sc.get('behavior')
    if not behav_path or not Path(behav_path).exists():
        return np.full(len(time_sec), 'Unknown')
    behav_df = pd.read_csv(behav_path, header=None)
    zones = np.full(len(time_sec), 'Unknown', dtype=object)
    zone_names = ['Home', 'Ladder', 'Transition Zone', 'Foraging Arena']
    for zone_name in zone_names:
        for row_idx in range(behav_df.shape[0]):
            cell_val = str(behav_df.iloc[row_idx, 0]).strip()
            if zone_name.lower() in cell_val.lower():
                row_data = pd.to_numeric(behav_df.iloc[row_idx, 1:], errors='coerce').values
                for ti, t in enumerate(time_sec):
                    bi = int(t / 0.1)
                    if 0 <= bi < len(row_data) and row_data[bi] == 1:
                        zones[ti] = zone_name
                break
    return zones


# =============================================================================
# CIRCULAR COORDINATES via persistent cohomology
# =============================================================================

def compute_circular_coordinates(data_pca, diagrams):
    """
    Estimate circular coordinates using the longest-lived H1 cocycle.

    Approach: Use the Vietoris-Rips distance matrix and the birth/death
    of the top H1 feature to identify a filtration scale, then use a
    graph-based angle assignment around the detected loop.
    """
    # Get top H1 feature
    h1_mask = diagrams[:, 2] == 1
    h1_features = diagrams[h1_mask]
    if len(h1_features) == 0:
        return None

    lifetimes = h1_features[:, 1] - h1_features[:, 0]
    finite_mask = np.isfinite(lifetimes)
    if not finite_mask.any():
        return None

    finite_lt = lifetimes[finite_mask]
    top_idx = np.argmax(finite_lt)
    top_feature = h1_features[finite_mask][top_idx]
    birth, death = top_feature[0], top_feature[1]

    # Use filtration scale midway between birth and death
    eps = (birth + death) / 2

    # Build epsilon-neighborhood graph
    D = squareform(pdist(data_pca))
    n = len(data_pca)

    # Find connected path through the graph at scale eps
    # Use a greedy circular ordering based on nearest-neighbor traversal
    adjacency = D < eps
    np.fill_diagonal(adjacency, False)

    # If graph is too sparse, increase epsilon
    for scale_factor in [1.0, 1.2, 1.5, 2.0]:
        adj = D < (eps * scale_factor)
        np.fill_diagonal(adj, False)
        if adj.sum(axis=1).min() >= 1:
            adjacency = adj
            break

    # Compute angular coordinates using diffusion maps / graph Laplacian
    # Build weighted adjacency
    W = np.exp(-D**2 / (2 * eps**2)) * adjacency.astype(float)
    degree = W.sum(axis=1)
    degree[degree == 0] = 1
    # Normalized Laplacian
    D_inv_sqrt = np.diag(1.0 / np.sqrt(degree))
    L_norm = np.eye(n) - D_inv_sqrt @ W @ D_inv_sqrt

    # Eigenvectors — the Fiedler vector pair gives circular coordinates
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(L_norm)
        # Skip the trivial zero eigenvalue
        idx = np.argsort(eigenvalues)
        ev1 = eigenvectors[:, idx[1]]  # Fiedler vector
        ev2 = eigenvectors[:, idx[2]]  # Second non-trivial

        # Circular coordinate = angle in (ev1, ev2) plane
        theta = np.arctan2(ev2, ev1)  # range [-pi, pi]
        # Shift to [0, 2pi]
        theta = (theta + 2 * np.pi) % (2 * np.pi)

        return theta
    except Exception:
        return None


def unwrap_theta_along_time(theta):
    """Unwrap circular coordinate to show cumulative winding."""
    unwrapped = np.unwrap(theta)
    return unwrapped


# =============================================================================
# VISUALIZATION: Smooth tube trajectory
# =============================================================================

def interpolate_trajectory(coords, n_interp=300):
    """Spline-interpolate a trajectory for smooth rendering."""
    n = len(coords)
    if n < 4:
        return coords, np.linspace(0, 1, n)

    t_orig = np.linspace(0, 1, n)
    t_fine = np.linspace(0, 1, n_interp)

    k = min(3, n - 1)  # spline degree
    coords_fine = np.zeros((n_interp, coords.shape[1]))
    for d in range(coords.shape[1]):
        spl = make_interp_spline(t_orig, coords[:, d], k=k)
        coords_fine[:, d] = spl(t_fine)

    return coords_fine, t_fine


def plot_tube_trajectory_3d(ax, coords, time_norm, title, n_interp=300):
    """3D tube trajectory with color gradient and direction arrows."""
    coords_fine, t_fine = interpolate_trajectory(coords, n_interp)

    # Create colored line segments
    points = coords_fine.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Color by time
    colors = plt.cm.plasma(t_fine[:-1])

    # Variable linewidth: thicker in middle, thinner at ends
    n_seg = len(segments)
    lw_base = 3.5
    lw = np.ones(n_seg) * lw_base

    for i, (seg, c) in enumerate(zip(segments, colors)):
        ax.plot([seg[0, 0], seg[1, 0]],
                [seg[0, 1], seg[1, 1]],
                [seg[0, 2], seg[1, 2]],
                color=c, linewidth=lw[i], alpha=0.85, solid_capstyle='round')

    # Direction arrows at regular intervals
    n_arrows = min(8, len(coords_fine) // 10)
    arrow_indices = np.linspace(n_interp // 10, n_interp - n_interp // 10,
                                 n_arrows, dtype=int)
    for ai in arrow_indices:
        if ai + 2 < len(coords_fine):
            p1 = coords_fine[ai]
            p2 = coords_fine[ai + 2]
            dp = p2 - p1
            dp_norm = np.linalg.norm(dp)
            if dp_norm > 0:
                dp = dp / dp_norm
                arrow_len = dp_norm * 1.5
                ax.quiver(p1[0], p1[1], p1[2],
                          dp[0] * arrow_len, dp[1] * arrow_len, dp[2] * arrow_len,
                          color=plt.cm.plasma(t_fine[ai]),
                          arrow_length_ratio=0.4, linewidth=2, alpha=0.9)

    # Mark start (green star) and end (red X)
    ax.scatter(*coords_fine[0], c='lime', s=200, marker='*',
               edgecolors='black', linewidths=1.5, zorder=10, label='Start')
    ax.scatter(*coords_fine[-1], c='red', s=150, marker='X',
               edgecolors='black', linewidths=1.5, zorder=10, label='End')

    # Also plot original data points as small dots
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
               c='white', s=20, edgecolors='gray', linewidths=0.5,
               alpha=0.6, zorder=5)

    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Dim 1', fontsize=9)
    ax.set_ylabel('Dim 2', fontsize=9)
    ax.set_zlabel('Dim 3', fontsize=9)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=8, loc='upper left')


def plot_tube_trajectory_2d(ax, coords_2d, time_norm, title, n_interp=300):
    """2D tube trajectory with color gradient and direction arrows."""
    coords_fine, t_fine = interpolate_trajectory(coords_2d, n_interp)

    # Colored line segments using LineCollection
    points = coords_fine.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap='plasma',
                        norm=Normalize(0, 1), linewidths=3.5)
    lc.set_array(t_fine[:-1])
    ax.add_collection(lc)

    # Direction arrows
    n_arrows = min(8, len(coords_fine) // 10)
    arrow_indices = np.linspace(n_interp // 10, n_interp - n_interp // 10,
                                 n_arrows, dtype=int)
    for ai in arrow_indices:
        if ai + 3 < len(coords_fine):
            p1 = coords_fine[ai]
            p2 = coords_fine[ai + 3]
            dp = p2 - p1
            ax.annotate('', xy=p2, xytext=p1,
                        arrowprops=dict(arrowstyle='->', color=plt.cm.plasma(t_fine[ai]),
                                        lw=2, mutation_scale=15))

    # Start / end markers
    ax.scatter(*coords_fine[0], c='lime', s=200, marker='*',
               edgecolors='black', linewidths=1.5, zorder=10, label='Start')
    ax.scatter(*coords_fine[-1], c='red', s=150, marker='X',
               edgecolors='black', linewidths=1.5, zorder=10, label='End')

    # Original points
    ax.scatter(coords_2d[:, 0], coords_2d[:, 1],
               c='white', s=25, edgecolors='gray', linewidths=0.5,
               alpha=0.6, zorder=5)

    # Auto-scale
    margin = 0.1
    xmin, xmax = coords_fine[:, 0].min(), coords_fine[:, 0].max()
    ymin, ymax = coords_fine[:, 1].min(), coords_fine[:, 1].max()
    dx = (xmax - xmin) * margin
    dy = (ymax - ymin) * margin
    ax.set_xlim(xmin - dx, xmax + dx)
    ax.set_ylim(ymin - dy, ymax + dy)

    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Dim 1', fontsize=9)
    ax.set_ylabel('Dim 2', fontsize=9)
    ax.legend(fontsize=8)
    ax.set_aspect('equal', adjustable='datalim')


# =============================================================================
# VISUALIZATION: Recurrence matrix
# =============================================================================

def plot_recurrence_matrix(ax, data, time_sec, title, threshold_percentile=20):
    """Plot distance recurrence matrix. Loops show as off-diagonal stripes."""
    D = squareform(pdist(data))

    # Normalize to [0, 1]
    D_norm = D / D.max() if D.max() > 0 else D

    # Relative time
    t_rel = time_sec - time_sec[0]

    im = ax.imshow(D_norm, cmap='magma_r', origin='lower', aspect='equal',
                   extent=[t_rel[0], t_rel[-1], t_rel[0], t_rel[-1]])
    ax.set_xlabel('Time (s)', fontsize=10)
    ax.set_ylabel('Time (s)', fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold')

    # Add threshold contour to highlight recurrences
    threshold = np.percentile(D_norm, threshold_percentile)
    ax.contour(D_norm, levels=[threshold], colors=['cyan'], linewidths=0.8,
               extent=[t_rel[0], t_rel[-1], t_rel[0], t_rel[-1]],
               origin='lower')

    return im


# =============================================================================
# VISUALIZATION: Circular coordinates
# =============================================================================

def plot_circular_coordinates(axes, theta, time_sec, coords_2d, time_norm,
                               feature_type, gap_val):
    """
    Plot circular coordinate analysis:
      axes[0]: theta vs time (unwrapped shows winding)
      axes[1]: 2D embedding colored by theta (shows ring)
      axes[2]: polar plot of trajectory
    """
    t_rel = time_sec - time_sec[0]

    # Panel 1: Theta vs time (wrapped and unwrapped)
    ax = axes[0]
    theta_unwrapped = unwrap_theta_along_time(theta)
    total_winding = (theta_unwrapped[-1] - theta_unwrapped[0]) / (2 * np.pi)

    ax.plot(t_rel, np.degrees(theta), 'o-', color='#D32F2F', markersize=5,
            linewidth=1.5, alpha=0.8, label=f'Wrapped $\\theta$')
    ax2 = ax.twinx()
    ax2.plot(t_rel, np.degrees(theta_unwrapped), 's-', color='#1976D2',
             markersize=4, linewidth=1.5, alpha=0.8, label=f'Unwrapped')
    ax2.set_ylabel('Unwrapped angle (deg)', fontsize=10, color='#1976D2')
    ax2.tick_params(axis='y', labelcolor='#1976D2')

    ax.set_xlabel('Time (s)', fontsize=10)
    ax.set_ylabel('Circular coordinate $\\theta$ (deg)', fontsize=10, color='#D32F2F')
    ax.tick_params(axis='y', labelcolor='#D32F2F')
    ax.set_title(f'Circular Coordinate vs Time\n'
                 f'Total winding: {total_winding:.2f} turns',
                 fontsize=12, fontweight='bold')

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='upper left')

    # Panel 2: 2D embedding colored by theta
    ax = axes[1]
    sc = ax.scatter(coords_2d[:, 0], coords_2d[:, 1],
                    c=theta, cmap='hsv', s=60, alpha=0.9,
                    edgecolors='black', linewidths=0.5)
    plt.colorbar(sc, ax=ax, label='$\\theta$ (rad)', shrink=0.8)

    # Connect with line
    for i in range(len(coords_2d) - 1):
        ax.plot(coords_2d[i:i+2, 0], coords_2d[i:i+2, 1],
                color=plt.cm.hsv(theta[i] / (2 * np.pi)),
                linewidth=2, alpha=0.6)

    ax.scatter(*coords_2d[0], c='lime', s=200, marker='*',
               edgecolors='black', linewidths=1.5, zorder=10)
    ax.scatter(*coords_2d[-1], c='red', s=150, marker='X',
               edgecolors='black', linewidths=1.5, zorder=10)

    ax.set_title(f'2D Embedding colored by $\\theta$\n'
                 f'{feature_type} gap = {gap_val:.1f}x',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Dim 1', fontsize=10)
    ax.set_ylabel('Dim 2', fontsize=10)
    ax.set_aspect('equal', adjustable='datalim')

    # Panel 3: Polar plot
    ax = axes[2]
    ax_polar = ax.figure.add_axes(ax.get_position(), polar=True)
    ax.set_visible(False)

    # Radius = normalized time, angle = theta
    r = time_norm
    colors = plt.cm.plasma(time_norm)
    ax_polar.scatter(theta, r, c=time_norm, cmap='plasma', s=40, alpha=0.9,
                     edgecolors='black', linewidths=0.3)
    for i in range(len(theta) - 1):
        ax_polar.plot([theta[i], theta[i+1]], [r[i], r[i+1]],
                      color=colors[i], linewidth=2, alpha=0.7)

    ax_polar.set_title(f'Polar: $\\theta$ vs time\n'
                       f'(radius = normalized time)',
                       fontsize=11, fontweight='bold', pad=20)
    ax_polar.set_rticks([0.25, 0.5, 0.75, 1.0])
    ax_polar.set_rlabel_position(45)
    ax_polar.tick_params(labelsize=7)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("Loop Visualization — Tube Trajectories + Circular Coordinates + Recurrence")
    print("=" * 70)

    sp = paths_config['single_probe']['coordinates_1']['mouse01']['sessions']
    sc = sp['session_1']
    sorted_path = Path(sc['sorted'])

    sorting = se.read_kilosort(sorted_path)
    lha_ids, rsp_ids = get_good_units_by_region(sorted_path)
    print(f"LHA: {len(lha_ids)} neurons, RSP: {len(rsp_ids)} neurons")

    exc_df = pd.read_csv("data/excursions_session_1.csv")
    complete = exc_df[exc_df['label'] == 'complete']

    binned_cache = {}

    for entry in PROMINENT_EXCURSIONS:
        eid = entry['exc_id']
        region = entry['region']
        bin_ms = entry['bin_ms']
        smooth_ms = entry['smooth_ms']
        feature_type = entry['feature']
        gap_val = entry['gap']
        note = entry['note']
        region_label = region.upper()
        unit_ids = lha_ids if region == 'lha' else rsp_ids

        print(f"\n{'='*60}")
        print(f"  Exc {eid} | {region_label} | {feature_type} (gap={gap_val}x) | {note}")
        print(f"{'='*60}")

        erow = complete[complete['excursion_id'] == eid]
        if len(erow) == 0:
            print(f"  Excursion {eid} not found, skipping")
            continue
        erow = erow.iloc[0]

        cache_key = (region, bin_ms, smooth_ms)
        if cache_key not in binned_cache:
            print(f"  Binning {region_label} at {bin_ms}ms...", end='', flush=True)
            zscore, time_sec = bin_and_smooth(sorting, unit_ids, bin_ms, smooth_ms)
            binned_cache[cache_key] = (zscore, time_sec)
            print(f" done")
        else:
            zscore, time_sec = binned_cache[cache_key]

        mask = (time_sec >= erow['start_time']) & (time_sec <= erow['end_time'])
        exc_data = zscore[mask]
        exc_times = time_sec[mask]
        n_pts = len(exc_data)

        if n_pts < 10:
            print(f"  Only {n_pts} points, skipping")
            continue

        print(f"  {n_pts} time points, {erow['duration']:.1f}s")

        time_norm = (exc_times - exc_times.min()) / max(exc_times.max() - exc_times.min(), 1e-8)
        zones = load_behavior_zones(1, exc_times)

        # =====================================================
        # Embeddings
        # =====================================================
        pca3d = PCA(n_components=3).fit_transform(exc_data)
        pca2d = PCA(n_components=2).fit_transform(exc_data)

        # Isomap
        n_nb = min(10, n_pts - 1)
        try:
            iso3d = Isomap(n_components=3, n_neighbors=n_nb).fit_transform(exc_data)
            iso2d = Isomap(n_components=2, n_neighbors=n_nb).fit_transform(exc_data)
            has_iso = True
        except:
            has_iso = False
            iso3d = pca3d
            iso2d = pca2d

        # =====================================================
        # Persistent homology + circular coordinates
        # =====================================================
        print("  Computing persistence + circular coords...", end='', flush=True)
        n_comp = min(15, exc_data.shape[1], n_pts - 1)
        pca_full = PCA(n_components=n_comp)
        data_pca = pca_full.fit_transform(exc_data)
        var_cum = pca_full.explained_variance_ratio_.cumsum()
        n_keep = min(np.searchsorted(var_cum, 0.95) + 1, n_comp)
        n_keep = max(n_keep, 3)
        data_pca = data_pca[:, :n_keep]

        VR = VietorisRipsPersistence(
            homology_dimensions=[0, 1, 2], max_edge_length=np.inf, n_jobs=-1)
        diagrams = VR.fit_transform(data_pca[np.newaxis, :, :])[0]

        theta = compute_circular_coordinates(data_pca, diagrams)
        has_theta = theta is not None
        if has_theta:
            print(f" theta OK")
        else:
            print(f" theta FAILED (will skip circular panels)")

        # =====================================================
        # FIGURE: 3 rows x 3 cols
        # Row 0: 3D tube (PCA) | 3D tube (Isomap) | 2D tube (best embedding)
        # Row 1: Recurrence matrix | theta vs time | 2D colored by theta
        # Row 2: Polar theta plot | Persistence diagram | Zone-colored 2D tube
        # =====================================================

        fig = plt.figure(figsize=(22, 20))
        fig.suptitle(
            f"Excursion {eid} — {region_label} — Session 1 (Fed)\n"
            f"{feature_type} (gap = {gap_val:.1f}x) | "
            f"{erow['duration']:.1f}s, {n_pts} bins ({bin_ms}ms/{smooth_ms}ms) | "
            f"{len(unit_ids)} neurons | {note}",
            fontsize=15, fontweight='bold', y=0.98)

        # --- Row 0: Tube trajectories ---
        ax_3d_pca = fig.add_subplot(3, 3, 1, projection='3d')
        plot_tube_trajectory_3d(ax_3d_pca, pca3d, time_norm,
                                'PCA 3D — Tube Trajectory')

        ax_3d_iso = fig.add_subplot(3, 3, 2, projection='3d')
        plot_tube_trajectory_3d(ax_3d_iso, iso3d, time_norm,
                                'Isomap 3D — Tube Trajectory')

        ax_2d = fig.add_subplot(3, 3, 3)
        plot_tube_trajectory_2d(ax_2d, iso2d if has_iso else pca2d, time_norm,
                                f'{"Isomap" if has_iso else "PCA"} 2D — Tube Trajectory')

        # Add time colorbar
        sm = ScalarMappable(cmap='plasma', norm=Normalize(0, 1))
        sm.set_array([])

        # --- Row 1: Recurrence + Circular coordinates ---
        ax_rec = fig.add_subplot(3, 3, 4)
        im = plot_recurrence_matrix(ax_rec, data_pca, exc_times,
                                     'Recurrence Distance Matrix\n(cyan = close returns)')
        plt.colorbar(im, ax=ax_rec, label='Normalized distance', shrink=0.8)

        if has_theta:
            # Theta vs time
            ax_theta_time = fig.add_subplot(3, 3, 5)
            theta_unwrapped = unwrap_theta_along_time(theta)
            total_winding = (theta_unwrapped[-1] - theta_unwrapped[0]) / (2 * np.pi)
            t_rel = exc_times - exc_times[0]

            ax_theta_time.plot(t_rel, np.degrees(theta), 'o-', color='#D32F2F',
                               markersize=6, linewidth=1.5, label='Wrapped $\\theta$')
            ax_tw = ax_theta_time.twinx()
            ax_tw.plot(t_rel, np.degrees(theta_unwrapped), 's-', color='#1976D2',
                       markersize=4, linewidth=1.5, label='Unwrapped $\\theta$')
            ax_tw.set_ylabel('Unwrapped (deg)', fontsize=9, color='#1976D2')
            ax_tw.tick_params(axis='y', labelcolor='#1976D2')
            ax_theta_time.set_xlabel('Time (s)', fontsize=10)
            ax_theta_time.set_ylabel('$\\theta$ (deg)', fontsize=10, color='#D32F2F')
            ax_theta_time.tick_params(axis='y', labelcolor='#D32F2F')
            ax_theta_time.set_title(
                f'Circular Coordinate vs Time\nTotal winding: {total_winding:.2f} turns',
                fontsize=12, fontweight='bold')
            lines1, labels1 = ax_theta_time.get_legend_handles_labels()
            lines2, labels2 = ax_tw.get_legend_handles_labels()
            ax_theta_time.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

            # 2D colored by theta
            ax_theta_2d = fig.add_subplot(3, 3, 6)
            best_2d = iso2d if has_iso else pca2d
            sc = ax_theta_2d.scatter(best_2d[:, 0], best_2d[:, 1],
                                      c=theta, cmap='hsv', s=80, alpha=0.9,
                                      edgecolors='black', linewidths=0.5)
            for i in range(len(best_2d) - 1):
                ax_theta_2d.plot(best_2d[i:i+2, 0], best_2d[i:i+2, 1],
                                 color=plt.cm.hsv(theta[i] / (2 * np.pi)),
                                 linewidth=2.5, alpha=0.7)
            ax_theta_2d.scatter(*best_2d[0], c='lime', s=200, marker='*',
                                edgecolors='black', linewidths=1.5, zorder=10)
            ax_theta_2d.scatter(*best_2d[-1], c='red', s=150, marker='X',
                                edgecolors='black', linewidths=1.5, zorder=10)
            plt.colorbar(sc, ax=ax_theta_2d, label='$\\theta$ (rad)', shrink=0.8)
            ax_theta_2d.set_title('2D Embedding colored by $\\theta$\n'
                                   '(rainbow = ring structure)',
                                  fontsize=12, fontweight='bold')
            ax_theta_2d.set_aspect('equal', adjustable='datalim')
        else:
            ax5 = fig.add_subplot(3, 3, 5)
            ax5.text(0.5, 0.5, 'Circular coordinates\nnot available',
                     ha='center', va='center', fontsize=14)
            ax5.set_title('Circular Coordinate', fontsize=12)
            ax6 = fig.add_subplot(3, 3, 6)
            ax6.text(0.5, 0.5, 'Circular coordinates\nnot available',
                     ha='center', va='center', fontsize=14)

        # --- Row 2: Polar + Persistence + Zone-colored trajectory ---

        # Polar plot
        if has_theta:
            ax_polar = fig.add_subplot(3, 3, 7, polar=True)
            r = time_norm
            ax_polar.scatter(theta, r, c=time_norm, cmap='plasma', s=50, alpha=0.9,
                             edgecolors='black', linewidths=0.3, zorder=5)
            for i in range(len(theta) - 1):
                ax_polar.plot([theta[i], theta[i+1]], [r[i], r[i+1]],
                              color=plt.cm.plasma(time_norm[i]), linewidth=2.5, alpha=0.7)
            ax_polar.set_title(f'Polar: $\\theta$ vs time\n'
                               f'Winding = {total_winding:.2f} turns',
                               fontsize=12, fontweight='bold', pad=20)
            ax_polar.set_rticks([0.25, 0.5, 0.75, 1.0])
            ax_polar.set_rlabel_position(45)
        else:
            ax7 = fig.add_subplot(3, 3, 7)
            ax7.text(0.5, 0.5, 'No circular\ncoordinates', ha='center', va='center', fontsize=14)

        # Persistence diagram
        ax_pers = fig.add_subplot(3, 3, 8)
        max_val = 0
        for dim in range(3):
            dmask = diagrams[:, 2] == dim
            features = diagrams[dmask]
            if len(features) == 0:
                continue
            finite = features[np.isfinite(features[:, 1])]
            if len(finite) > 0:
                lt = finite[:, 1] - finite[:, 0]
                sizes = np.full(len(finite), 25)
                sizes[np.argmax(lt)] = 150
                ax_pers.scatter(finite[:, 0], finite[:, 1],
                                c=[DIM_COLORS[dim]] * len(finite), s=sizes, alpha=0.6,
                                label=f'H{dim}', zorder=2,
                                edgecolors='black' if dim > 0 else 'none',
                                linewidths=np.where(sizes > 50, 2, 0))
                max_val = max(max_val, finite.max())
        if max_val > 0:
            ax_pers.plot([0, max_val * 1.1], [0, max_val * 1.1], 'k--', alpha=0.3)
        ax_pers.set_xlabel('Birth', fontsize=10)
        ax_pers.set_ylabel('Death', fontsize=10)
        ax_pers.set_title(f'Persistence Diagram\n{feature_type} gap = {gap_val:.1f}x',
                          fontsize=12, fontweight='bold',
                          color='#D32F2F' if 'H1' in feature_type else '#4CAF50')
        ax_pers.legend(fontsize=8)

        # Zone-colored 2D tube
        ax_zone = fig.add_subplot(3, 3, 9)
        best_2d = iso2d if has_iso else pca2d
        # Interpolate
        coords_fine, t_fine = interpolate_trajectory(best_2d, 300)
        # Map zone to interpolated points
        zone_at_fine = np.array([zones[min(int(tf * (n_pts - 1)), n_pts - 1)] for tf in t_fine])

        for zone, color in ZONE_COLORS.items():
            zmask = zone_at_fine == zone
            if zmask.any():
                ax_zone.scatter(coords_fine[zmask, 0], coords_fine[zmask, 1],
                                c=color, s=8, alpha=0.7, label=zone)

        # Draw trajectory line colored by zone
        for i in range(len(coords_fine) - 1):
            z = zone_at_fine[i]
            c = ZONE_COLORS.get(z, '#BDBDBD')
            ax_zone.plot(coords_fine[i:i+2, 0], coords_fine[i:i+2, 1],
                         color=c, linewidth=3, alpha=0.7)

        ax_zone.scatter(*best_2d[0], c='lime', s=200, marker='*',
                        edgecolors='black', linewidths=1.5, zorder=10)
        ax_zone.scatter(*best_2d[-1], c='red', s=150, marker='X',
                        edgecolors='black', linewidths=1.5, zorder=10)
        ax_zone.set_title('2D Trajectory — Zone Coloring', fontsize=12, fontweight='bold')
        ax_zone.set_xlabel('Dim 1', fontsize=10)
        ax_zone.set_ylabel('Dim 2', fontsize=10)
        ax_zone.legend(fontsize=8, loc='best')
        ax_zone.set_aspect('equal', adjustable='datalim')

        plt.tight_layout(rect=[0, 0, 1, 0.94])
        outpath = Path("figures") / f"loop_viz_exc{eid}_{region}_s1.png"
        fig.savefig(outpath, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {outpath}")

    print("\nDone!")


if __name__ == "__main__":
    main()
