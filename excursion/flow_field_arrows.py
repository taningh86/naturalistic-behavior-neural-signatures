"""
Flow Field Diagrams for Feeding & Digging Excursions
=====================================================
Consistent with GRU-ODE flow field style:
  Top row: Streamplot + trajectory scatter (behavior / speed / zone coloring)
  Bottom row: Speed heatmap with quiver overlay (behavior / speed / zone coloring)
"""

import yaml
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from scipy.ndimage import uniform_filter1d, gaussian_filter
import spikeinterface.extractors as se
import warnings

warnings.filterwarnings('ignore')

FS = 30000
LHA_DEPTH_MAX = 1300
RSP_DEPTH_MIN = 1300

with open("paths.yaml") as f:
    paths_config = yaml.safe_load(f)

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


def interpolate_flow_to_grid(coords_2d, dx, dy, grid_res=40):
    """Compute kernel-averaged velocity field on a grid.
    Uses Gaussian kernel weighting so flow vectors are only meaningful
    where there is sufficient local data density. Grid is focused on
    the data-dense region (central 90% of points) rather than full range."""

    # Focus grid on data-dense region (trim outliers)
    p5_x, p95_x = np.percentile(coords_2d[:, 0], [5, 95])
    p5_y, p95_y = np.percentile(coords_2d[:, 1], [5, 95])
    range_x = p95_x - p5_x
    range_y = p95_y - p5_y
    pad_x = range_x * 0.15
    pad_y = range_y * 0.15
    x_min, x_max = p5_x - pad_x, p95_x + pad_x
    y_min, y_max = p5_y - pad_y, p95_y + pad_y

    grid_x = np.linspace(x_min, x_max, grid_res)
    grid_y = np.linspace(y_min, y_max, grid_res)
    GX, GY = np.meshgrid(grid_x, grid_y)

    # Velocity observation positions (midpoints between consecutive states)
    pts = (coords_2d[:-1] + coords_2d[1:]) / 2

    # Kernel bandwidth: adaptive based on data spread
    bandwidth = max(range_x, range_y) / 12

    # Gaussian kernel-averaged velocity at each grid point
    U = np.zeros_like(GX)
    V = np.zeros_like(GX)
    W = np.zeros_like(GX)  # total weight (for density masking)

    for i in range(len(pts)):
        d2 = (GX - pts[i, 0])**2 + (GY - pts[i, 1])**2
        w = np.exp(-d2 / (2 * bandwidth**2))
        U += w * dx[i]
        V += w * dy[i]
        W += w

    # Normalize by total weight
    mask = W > 1e-10
    U[mask] /= W[mask]
    V[mask] /= W[mask]
    U[~mask] = 0.0
    V[~mask] = 0.0

    # Mask out low-density regions (where total weight is negligible)
    # This ensures flow only appears where data actually exists
    w_thresh = np.percentile(W[W > 1e-10], 15) if (W > 1e-10).sum() > 10 else 0
    low_density = W < w_thresh
    U[low_density] = 0.0
    V[low_density] = 0.0

    speed = np.sqrt(U**2 + V**2)

    return grid_x, grid_y, GX, GY, U, V, speed


def make_flow_field_figure(coords_2d, dx, dy, speed_2d,
                           behav_labels, zone_labels,
                           eid, region_label, n_neurons, res_label,
                           exc_label, duration, n_pts):
    """Create 2x3 figure matching GRU-ODE flow field style.

    Top row: streamplot + trajectory scatter (behavior / speed / zone)
    Bottom row: speed heatmap + quiver overlay (behavior / speed / zone)
    """

    # Interpolate flow onto grid
    grid_x, grid_y, GX, GY, U, V, speed_grid = interpolate_flow_to_grid(
        coords_2d, dx, dy, grid_res=40)

    speed_max = speed_grid.max()

    is_feeding = 'feeding' in exc_label.lower()
    is_digging = 'digging' in exc_label.lower()
    type_color = '#D32F2F' if is_feeding else '#FF9800' if is_digging else '#666666'
    type_tag = 'FEEDING' if is_feeding else 'DIGGING' if is_digging else ''

    fig, axes = plt.subplots(2, 3, figsize=(22, 13))
    fig.suptitle(
        f"{region_label} Flow Field — Excursion {eid} [{type_tag}] — Session 1 (Fed)\n"
        f"{exc_label} | {duration:.1f}s, {n_pts} bins ({res_label}) | {n_neurons} neurons",
        fontsize=14, fontweight='bold', color=type_color, y=0.98)

    xlim = (grid_x[0], grid_x[-1])
    ylim = (grid_y[0], grid_y[-1])

    # Skip factor for quiver in bottom row
    skip = 3

    # ====== TOP ROW: Streamplot + trajectory scatter ======

    # --- Panel (0,0): Streamplot + Behavior scatter ---
    ax = axes[0, 0]
    ax.streamplot(grid_x, grid_y, U, V,
                  color=speed_grid, cmap='Greys', linewidth=0.8, arrowsize=1.0,
                  density=1.8, arrowstyle='->',
                  norm=mcolors.Normalize(vmin=0, vmax=speed_max * 0.8))
    for bname, color in BEHAVIOR_COLORS.items():
        mask = behav_labels == bname
        if mask.any():
            ax.scatter(coords_2d[mask, 0], coords_2d[mask, 1],
                       c=color, s=25, alpha=0.7, label=bname, zorder=5,
                       edgecolors='black', linewidths=0.3)
    other = behav_labels == 'Other'
    if other.any():
        ax.scatter(coords_2d[other, 0], coords_2d[other, 1],
                   c='#BDBDBD', s=8, alpha=0.25, zorder=2, rasterized=True)
    ax.scatter(*coords_2d[0], c='lime', s=200, marker='*',
               edgecolors='black', linewidths=1.5, zorder=10)
    ax.scatter(*coords_2d[-1], c='red', s=150, marker='X',
               edgecolors='black', linewidths=1.5, zorder=10)
    ax.set_xlabel('PC1', fontsize=11)
    ax.set_ylabel('PC2', fontsize=11)
    ax.set_title('Flow Field + Behavior', fontsize=12)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.legend(fontsize=6, loc='best', markerscale=1.3)

    # --- Panel (0,1): Streamplot + Speed scatter ---
    ax = axes[0, 1]
    ax.streamplot(grid_x, grid_y, U, V,
                  color=speed_grid, cmap='Greys', linewidth=0.8, arrowsize=1.0,
                  density=1.8, arrowstyle='->',
                  norm=mcolors.Normalize(vmin=0, vmax=speed_max * 0.8))
    speed_norm = mcolors.Normalize(vmin=np.percentile(speed_2d, 5),
                                   vmax=np.percentile(speed_2d, 95))
    sc = ax.scatter(coords_2d[:-1, 0], coords_2d[:-1, 1],
                    c=speed_2d, cmap='hot_r', norm=speed_norm, s=20, alpha=0.7,
                    edgecolors='none', zorder=5, rasterized=True)
    plt.colorbar(sc, ax=ax, label='Embedding speed', shrink=0.8)
    ax.scatter(*coords_2d[0], c='lime', s=200, marker='*',
               edgecolors='black', linewidths=1.5, zorder=10)
    ax.scatter(*coords_2d[-1], c='red', s=150, marker='X',
               edgecolors='black', linewidths=1.5, zorder=10)
    ax.set_xlabel('PC1', fontsize=11)
    ax.set_ylabel('PC2', fontsize=11)
    ax.set_title('Flow Field + Speed', fontsize=12)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # --- Panel (0,2): Streamplot + Zone scatter ---
    ax = axes[0, 2]
    ax.streamplot(grid_x, grid_y, U, V,
                  color=speed_grid, cmap='Greys', linewidth=0.8, arrowsize=1.0,
                  density=1.8, arrowstyle='->',
                  norm=mcolors.Normalize(vmin=0, vmax=speed_max * 0.8))
    for zone, color in ZONE_COLORS.items():
        zmask = zone_labels == zone
        if zmask.any():
            ax.scatter(coords_2d[zmask, 0], coords_2d[zmask, 1],
                       c=color, s=25, alpha=0.7, label=zone, zorder=5,
                       edgecolors='black', linewidths=0.3)
    ax.scatter(*coords_2d[0], c='lime', s=200, marker='*',
               edgecolors='black', linewidths=1.5, zorder=10)
    ax.scatter(*coords_2d[-1], c='red', s=150, marker='X',
               edgecolors='black', linewidths=1.5, zorder=10)
    ax.set_xlabel('PC1', fontsize=11)
    ax.set_ylabel('PC2', fontsize=11)
    ax.set_title('Flow Field + Zone', fontsize=12)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.legend(fontsize=8, loc='best')

    # ====== BOTTOM ROW: Speed heatmap + quiver overlay ======

    # --- Panel (1,0): Speed heatmap + quiver, behavior colored ---
    ax = axes[1, 0]
    im = ax.pcolormesh(GX, GY, speed_grid, cmap='magma_r', shading='auto',
                       vmin=0, vmax=speed_max)
    plt.colorbar(im, ax=ax, label='|dv/dt|', shrink=0.8)
    # Slow-region contour (attractor)
    slow_thresh = np.percentile(speed_grid, 15)
    ax.contour(GX, GY, speed_grid, levels=[slow_thresh],
               colors='cyan', linewidths=1.5, linestyles='--')
    # Quiver colored by behavior
    for i in range(0, len(dx), skip):
        bname = behav_labels[i]
        color = BEHAVIOR_COLORS.get(bname, 'white')
        alpha = 0.8 if bname != 'Other' else 0.4
        mid_x = (coords_2d[i, 0] + coords_2d[i+1, 0]) / 2
        mid_y = (coords_2d[i, 1] + coords_2d[i+1, 1]) / 2
        ax.quiver(mid_x, mid_y, dx[i], dy[i],
                  color=color, alpha=alpha, scale=None, width=0.004, zorder=5)
    ax.set_xlabel('PC1', fontsize=11)
    ax.set_ylabel('PC2', fontsize=11)
    ax.set_title('Flow Speed + Behavior Arrows\n(cyan = slow/attractor)', fontsize=11)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # --- Panel (1,1): Speed heatmap + white quiver ---
    ax = axes[1, 1]
    im = ax.pcolormesh(GX, GY, speed_grid, cmap='magma_r', shading='auto',
                       vmin=0, vmax=speed_max)
    plt.colorbar(im, ax=ax, label='|dv/dt|', shrink=0.8)
    ax.contour(GX, GY, speed_grid, levels=[slow_thresh],
               colors='cyan', linewidths=1.5, linestyles='--')
    ax.quiver(GX[::skip, ::skip], GY[::skip, ::skip],
              U[::skip, ::skip], V[::skip, ::skip],
              color='white', alpha=0.5, scale=None, width=0.003)
    ax.set_xlabel('PC1', fontsize=11)
    ax.set_ylabel('PC2', fontsize=11)
    ax.set_title('Flow Speed + Direction\n(light = slow/attractor)', fontsize=11)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # --- Panel (1,2): Dwell-time heatmap with trajectory ---
    ax = axes[1, 2]
    n_grid = 50
    heatmap, xedges, yedges = np.histogram2d(
        coords_2d[:, 0], coords_2d[:, 1], bins=n_grid,
        range=[[xlim[0], xlim[1]], [ylim[0], ylim[1]]])
    heatmap = gaussian_filter(heatmap.T, sigma=2)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    im = ax.imshow(heatmap, origin='lower', extent=extent, cmap='inferno',
                   aspect='auto', interpolation='bilinear')
    plt.colorbar(im, ax=ax, label='Dwell time (bins)', shrink=0.8)
    ax.plot(coords_2d[:, 0], coords_2d[:, 1], color='white', linewidth=0.3, alpha=0.4)
    ax.scatter(*coords_2d[0], c='lime', s=150, marker='*',
               edgecolors='white', linewidths=1, zorder=10)
    ax.scatter(*coords_2d[-1], c='red', s=120, marker='X',
               edgecolors='white', linewidths=1, zorder=10)
    ax.set_xlabel('PC1', fontsize=11)
    ax.set_ylabel('PC2', fontsize=11)
    ax.set_title('Dwell-Time Heatmap\n(bright = attractor region)', fontsize=11)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("Flow Field — Feeding & Digging Excursions")
    print("=" * 50)

    sp = paths_config['single_probe']['coordinates_1']['mouse01']['sessions']
    sc = sp['session_1']
    sorted_path = Path(sc['sorted'])
    sorting = se.read_kilosort(sorted_path)
    lha_ids, rsp_ids = get_good_units_by_region(sorted_path)

    exc_df = pd.read_csv("data/excursions_session_1.csv")
    complete = exc_df[exc_df['label'] == 'complete']

    excursions_to_plot = [
        {'exc_id': 81, 'label': 'Feeding (64% of time)'},
        {'exc_id': 57, 'label': 'Digging (6%) + H2 void 16.9x'},
        {'exc_id': 43, 'label': 'Similar #2 — Hesitant exploration'},
        {'exc_id': 53, 'label': 'Similar #3 — Other'},
        {'exc_id': 35, 'label': 'Similar #4 — Hesitant exploration'},
        {'exc_id': 45, 'label': 'Similar #5 — Hesitant exploration'},
        {'exc_id': 80, 'label': 'Similar #6 — Other'},
    ]

    for region, unit_ids in [('lha', lha_ids), ('rsp', rsp_ids)]:
        region_label = region.upper()

        for bin_ms, smooth_ms, res_label in [
            (50, 200, '50ms/200ms'),
            (200, 500, '200ms/500ms'),
            (500, 1000, '500ms/1s'),
        ]:
            print(f"\n  {region_label} @ {res_label}")
            zscore, time_sec = bin_and_smooth(sorting, unit_ids, bin_ms, smooth_ms)
            behav_dict = load_behavior_timeseries(1, time_sec)

            # Fit SHARED PCA on full session data
            pca_shared = PCA(n_components=2).fit(zscore)
            full_coords = pca_shared.transform(zscore)
            var_pct = pca_shared.explained_variance_ratio_ * 100
            print(f"    Shared PCA: PC1={var_pct[0]:.1f}%, PC2={var_pct[1]:.1f}%")

            for entry in excursions_to_plot:
                eid = entry['exc_id']
                exc_label = entry['label']

                erow = complete[complete['excursion_id'] == eid]
                if len(erow) == 0:
                    continue
                erow = erow.iloc[0]

                mask = (time_sec >= erow['start_time']) & (time_sec <= erow['end_time'])
                exc_data = zscore[mask]
                exc_times = time_sec[mask]
                n_pts = len(exc_data)

                if n_pts < 10:
                    print(f"    Exc {eid}: only {n_pts} pts, skipping")
                    continue

                print(f"    Exc {eid} ({exc_label}): {n_pts} pts")

                # Project into SHARED PCA space
                coords_2d = pca_shared.transform(exc_data)

                # Flow vectors
                dx = np.diff(coords_2d[:, 0])
                dy = np.diff(coords_2d[:, 1])
                speed_2d = np.sqrt(dx**2 + dy**2)

                # Behavior & zone labels
                behav_labels = get_dominant_behavior(behav_dict, mask)
                zone_labels = get_zone_labels(behav_dict, mask)

                fig = make_flow_field_figure(
                    coords_2d, dx, dy, speed_2d,
                    behav_labels, zone_labels,
                    eid, region_label, len(unit_ids), res_label,
                    exc_label, erow['duration'], n_pts)

                if 'Feeding' in exc_label:
                    tag = 'feeding'
                elif 'Digging' in exc_label:
                    tag = 'digging'
                else:
                    tag = 'similar'
                res_tag = f"{bin_ms}ms"
                outpath = Path("figures") / f"flow_field_{tag}_exc{eid}_{region}_{res_tag}.png"
                fig.savefig(outpath, dpi=200, bbox_inches='tight')
                plt.close()
                print(f"    Saved: {outpath}")

    print("\nDone!")


if __name__ == "__main__":
    main()
