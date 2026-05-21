"""
Flow State Plots for Feeding & Digging Excursions
==================================================
Visualizes neural state dynamics as a flow field:
  - 2D embedding with velocity arrows (quiver) showing direction of state evolution
  - Speed (magnitude of state change) colored along trajectory
  - Streamline-style visualization with behavior overlays
  - Flow speed vs time with behavior annotations
  - Density/dwell time heatmap showing attractor regions
"""

import yaml
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, LogNorm
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Patch, FancyArrowPatch
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from scipy.interpolate import make_interp_spline
from scipy.ndimage import uniform_filter1d, gaussian_filter
from scipy.spatial.distance import pdist, squareform
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
    """Get dominant behavior label for each time bin."""
    n = exc_mask.sum()
    labels = np.full(n, 'Other', dtype=object)
    priority = ['Feeding', 'Digging', 'Grooming', 'Quick arena exploration',
                'Arena wall exploration', 'Transition wall exploration',
                'Hesitant exploration', 'Quick one loop at home']
    for bname in reversed(priority):  # later in list = higher priority
        if bname in behav_dict:
            bdata = behav_dict[bname][exc_mask]
            labels[bdata > 0] = bname
    return labels


def get_zone_labels(behav_dict, exc_mask):
    """Get zone label for each time bin."""
    n = exc_mask.sum()
    labels = np.full(n, 'Other', dtype=object)
    for zone in ['Home', 'Ladder', 'Transition zone', 'Foraging arena']:
        if zone in behav_dict:
            zdata = behav_dict[zone][exc_mask]
            labels[zdata > 0] = zone
    return labels


def compute_flow_vectors(coords_2d):
    """Compute velocity vectors (dx, dy) at each time point."""
    dx = np.diff(coords_2d[:, 0])
    dy = np.diff(coords_2d[:, 1])
    speed = np.sqrt(dx**2 + dy**2)
    return dx, dy, speed


def compute_hd_speed(exc_data):
    """Compute speed in full high-dimensional neural state space."""
    diffs = np.diff(exc_data, axis=0)
    speed = np.sqrt(np.sum(diffs**2, axis=1))
    return speed


def make_flow_figure(exc_data, coords_2d, coords_3d, exc_times, behav_dict,
                     exc_mask, eid, region_label, n_neurons, res_label,
                     exc_label, duration):
    """Create the full flow state figure for one excursion."""

    n_pts = len(exc_data)
    t_rel = exc_times - exc_times[0]
    behav_labels = get_dominant_behavior(behav_dict, exc_mask)
    zone_labels = get_zone_labels(behav_dict, exc_mask)

    # Flow vectors in 2D
    dx, dy, speed_2d = compute_flow_vectors(coords_2d)

    # Speed in high-dimensional space
    speed_hd = compute_hd_speed(exc_data)
    # Smooth speed for cleaner visualization
    speed_hd_smooth = uniform_filter1d(speed_hd, size=3, mode='nearest')

    # =====================================================================
    # FIGURE: 3 rows x 3 cols
    # Row 0: Flow quiver (behavior) | Flow quiver (speed) | Flow quiver (zone)
    # Row 1: Streamline (time) | Dwell heatmap | 3D flow tube (speed)
    # Row 2: HD speed vs time | 2D speed vs time | Behavior-segmented speed boxplot
    # =====================================================================

    fig = plt.figure(figsize=(24, 22))

    is_feeding = 'Feeding' in exc_label or 'feeding' in exc_label.lower()
    is_digging = 'Digging' in exc_label or 'digging' in exc_label.lower()
    type_color = '#D32F2F' if is_feeding else '#FF9800' if is_digging else '#666666'
    type_tag = 'FEEDING' if is_feeding else 'DIGGING' if is_digging else 'OTHER'

    fig.suptitle(
        f"[{type_tag}] Flow State — Excursion {eid} — {region_label} — Session 1 (Fed)\n"
        f"{exc_label} | {duration:.1f}s, {n_pts} bins ({res_label}) | {n_neurons} neurons",
        fontsize=15, fontweight='bold', color=type_color, y=0.98)

    # --- Row 0: Flow quiver plots ---

    # Panel A: Flow arrows colored by BEHAVIOR
    ax = fig.add_subplot(3, 3, 1)
    # Background trajectory
    ax.plot(coords_2d[:, 0], coords_2d[:, 1], color='#E0E0E0', linewidth=0.5, zorder=1)
    # Arrows colored by behavior
    arrow_step = max(1, n_pts // 60)
    for i in range(0, len(dx), arrow_step):
        bname = behav_labels[i]
        color = BEHAVIOR_COLORS.get(bname, '#BDBDBD')
        alpha = 0.9 if bname != 'Other' else 0.3
        s = speed_2d[i]
        if s > 0:
            ax.annotate('', xy=(coords_2d[i, 0] + dx[i], coords_2d[i, 1] + dy[i]),
                        xytext=(coords_2d[i, 0], coords_2d[i, 1]),
                        arrowprops=dict(arrowstyle='->', color=color,
                                        lw=1.8, mutation_scale=12, alpha=alpha))
    # Scatter points
    for bname, color in BEHAVIOR_COLORS.items():
        mask = behav_labels == bname
        if mask.any():
            ax.scatter(coords_2d[mask, 0], coords_2d[mask, 1],
                       c=color, s=25, alpha=0.7, label=bname, zorder=5,
                       edgecolors='black', linewidths=0.3)
    other = behav_labels == 'Other'
    if other.any():
        ax.scatter(coords_2d[other, 0], coords_2d[other, 1],
                   c='#BDBDBD', s=10, alpha=0.3, zorder=2)
    ax.scatter(*coords_2d[0], c='lime', s=200, marker='*',
               edgecolors='black', linewidths=1.5, zorder=10)
    ax.scatter(*coords_2d[-1], c='red', s=150, marker='X',
               edgecolors='black', linewidths=1.5, zorder=10)
    ax.set_title('Flow Arrows — Behavior Coloring', fontsize=12, fontweight='bold')
    ax.set_xlabel('Dim 1', fontsize=10)
    ax.set_ylabel('Dim 2', fontsize=10)
    ax.legend(fontsize=6, loc='best', markerscale=1.5)
    ax.set_aspect('equal', adjustable='datalim')

    # Panel B: Flow arrows colored by SPEED
    ax = fig.add_subplot(3, 3, 2)
    ax.plot(coords_2d[:, 0], coords_2d[:, 1], color='#E0E0E0', linewidth=0.5, zorder=1)
    speed_norm = Normalize(vmin=np.percentile(speed_2d, 5),
                           vmax=np.percentile(speed_2d, 95))
    for i in range(0, len(dx), arrow_step):
        s = speed_2d[i]
        if s > 0:
            color = plt.cm.hot_r(speed_norm(s))
            ax.annotate('', xy=(coords_2d[i, 0] + dx[i], coords_2d[i, 1] + dy[i]),
                        xytext=(coords_2d[i, 0], coords_2d[i, 1]),
                        arrowprops=dict(arrowstyle='->', color=color,
                                        lw=1.8, mutation_scale=12, alpha=0.8))
    sc = ax.scatter(coords_2d[:-1, 0], coords_2d[:-1, 1],
                    c=speed_2d, cmap='hot_r', norm=speed_norm, s=20, alpha=0.7,
                    edgecolors='none', zorder=5)
    plt.colorbar(sc, ax=ax, label='2D speed', shrink=0.8)
    ax.scatter(*coords_2d[0], c='lime', s=200, marker='*',
               edgecolors='black', linewidths=1.5, zorder=10)
    ax.scatter(*coords_2d[-1], c='red', s=150, marker='X',
               edgecolors='black', linewidths=1.5, zorder=10)
    ax.set_title('Flow Arrows — Speed Coloring', fontsize=12, fontweight='bold')
    ax.set_xlabel('Dim 1', fontsize=10)
    ax.set_ylabel('Dim 2', fontsize=10)
    ax.set_aspect('equal', adjustable='datalim')

    # Panel C: Flow arrows colored by ZONE
    ax = fig.add_subplot(3, 3, 3)
    ax.plot(coords_2d[:, 0], coords_2d[:, 1], color='#E0E0E0', linewidth=0.5, zorder=1)
    for i in range(0, len(dx), arrow_step):
        zone = zone_labels[i]
        color = ZONE_COLORS.get(zone, '#BDBDBD')
        alpha = 0.8 if zone != 'Other' else 0.3
        s = speed_2d[i]
        if s > 0:
            ax.annotate('', xy=(coords_2d[i, 0] + dx[i], coords_2d[i, 1] + dy[i]),
                        xytext=(coords_2d[i, 0], coords_2d[i, 1]),
                        arrowprops=dict(arrowstyle='->', color=color,
                                        lw=1.8, mutation_scale=12, alpha=alpha))
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
    ax.set_title('Flow Arrows — Zone Coloring', fontsize=12, fontweight='bold')
    ax.set_xlabel('Dim 1', fontsize=10)
    ax.set_ylabel('Dim 2', fontsize=10)
    ax.legend(fontsize=7, loc='best')
    ax.set_aspect('equal', adjustable='datalim')

    # --- Row 1 ---

    # Panel D: Streamline colored by time with variable width = speed
    ax = fig.add_subplot(3, 3, 4)
    # Create line segments with variable width and time coloring
    points = coords_2d.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    # Width proportional to speed
    lw = 1.0 + 4.0 * (speed_2d / (speed_2d.max() + 1e-8))
    time_colors = np.linspace(0, 1, len(segments))
    for i, (seg, tc, w) in enumerate(zip(segments, time_colors, lw)):
        ax.plot([seg[0, 0], seg[1, 0]], [seg[0, 1], seg[1, 1]],
                color=plt.cm.plasma(tc), linewidth=w, alpha=0.8, solid_capstyle='round')
    # Direction arrows at intervals
    n_arrows = min(12, n_pts // 5)
    arrow_idx = np.linspace(5, n_pts - 5, n_arrows, dtype=int)
    for ai in arrow_idx:
        if ai + 1 < n_pts:
            ax.annotate('', xy=coords_2d[ai + 1], xytext=coords_2d[ai],
                        arrowprops=dict(arrowstyle='->', color=plt.cm.plasma(ai / n_pts),
                                        lw=2, mutation_scale=15))
    ax.scatter(*coords_2d[0], c='lime', s=200, marker='*',
               edgecolors='black', linewidths=1.5, zorder=10, label='Start')
    ax.scatter(*coords_2d[-1], c='red', s=150, marker='X',
               edgecolors='black', linewidths=1.5, zorder=10, label='End')
    sm = ScalarMappable(cmap='plasma', norm=Normalize(0, duration))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='Time (s)', shrink=0.8)
    ax.set_title('Streamline — Time + Speed (width)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Dim 1', fontsize=10)
    ax.set_ylabel('Dim 2', fontsize=10)
    ax.legend(fontsize=8)
    ax.set_aspect('equal', adjustable='datalim')

    # Panel E: Dwell-time heatmap
    ax = fig.add_subplot(3, 3, 5)
    # 2D histogram of where the trajectory spends time
    x_range = coords_2d[:, 0].max() - coords_2d[:, 0].min()
    y_range = coords_2d[:, 1].max() - coords_2d[:, 1].min()
    n_grid = 50
    heatmap, xedges, yedges = np.histogram2d(
        coords_2d[:, 0], coords_2d[:, 1], bins=n_grid)
    heatmap = gaussian_filter(heatmap.T, sigma=2)  # smooth
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    im = ax.imshow(heatmap, origin='lower', extent=extent, cmap='inferno',
                   aspect='equal', interpolation='bilinear')
    plt.colorbar(im, ax=ax, label='Dwell time (bins)', shrink=0.8)
    # Overlay trajectory
    ax.plot(coords_2d[:, 0], coords_2d[:, 1], color='white', linewidth=0.3, alpha=0.4)
    ax.scatter(*coords_2d[0], c='lime', s=150, marker='*',
               edgecolors='white', linewidths=1, zorder=10)
    ax.scatter(*coords_2d[-1], c='red', s=120, marker='X',
               edgecolors='white', linewidths=1, zorder=10)
    ax.set_title('Dwell-Time Heatmap\n(attractor = bright region)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Dim 1', fontsize=10)
    ax.set_ylabel('Dim 2', fontsize=10)

    # Panel F: 3D tube colored by speed
    ax = fig.add_subplot(3, 3, 6, projection='3d')
    speed_hd_full = np.concatenate([speed_hd_smooth, [speed_hd_smooth[-1]]])
    speed_norm_3d = Normalize(vmin=np.percentile(speed_hd_smooth, 5),
                              vmax=np.percentile(speed_hd_smooth, 95))
    for i in range(len(coords_3d) - 1):
        c = plt.cm.hot_r(speed_norm_3d(speed_hd_full[i]))
        ax.plot([coords_3d[i, 0], coords_3d[i+1, 0]],
                [coords_3d[i, 1], coords_3d[i+1, 1]],
                [coords_3d[i, 2], coords_3d[i+1, 2]],
                color=c, linewidth=3, alpha=0.85, solid_capstyle='round')
    ax.scatter(*coords_3d[0], c='lime', s=200, marker='*',
               edgecolors='black', linewidths=1.5, zorder=10)
    ax.scatter(*coords_3d[-1], c='red', s=150, marker='X',
               edgecolors='black', linewidths=1.5, zorder=10)
    ax.set_title('3D Tube — HD Speed Coloring\n(dark=fast, light=slow)',
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('Dim 1', fontsize=8)
    ax.set_ylabel('Dim 2', fontsize=8)
    ax.set_zlabel('Dim 3', fontsize=8)
    ax.tick_params(labelsize=7)

    # --- Row 2 ---

    # Panel G: HD speed vs time with behavior shading
    ax = fig.add_subplot(3, 3, 7)
    t_speed = t_rel[:-1] + np.diff(t_rel) / 2  # midpoints
    ax.plot(t_speed, speed_hd_smooth, color='black', linewidth=1.5, alpha=0.8, zorder=5)
    ax.fill_between(t_speed, 0, speed_hd_smooth, alpha=0.15, color='gray')

    # Shade behavior periods
    for bname, color in BEHAVIOR_COLORS.items():
        if bname in behav_dict:
            bdata = behav_dict[bname][exc_mask]
            active = bdata > 0
            if active.any():
                # Find contiguous blocks
                starts = np.where(np.diff(np.concatenate([[0], active.astype(int)])) == 1)[0]
                ends = np.where(np.diff(np.concatenate([active.astype(int), [0]])) == -1)[0]
                for s, e in zip(starts, ends):
                    t_s = t_rel[s]
                    t_e = t_rel[min(e, len(t_rel) - 1)]
                    ax.axvspan(t_s, t_e, alpha=0.25, color=color, zorder=1)
                    # Label first occurrence
                    if s == starts[0]:
                        ax.text((t_s + t_e) / 2, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else speed_hd_smooth.max(),
                                bname, ha='center', va='bottom', fontsize=7,
                                color=color, fontweight='bold', rotation=45)

    ax.set_xlabel('Time within excursion (s)', fontsize=10)
    ax.set_ylabel('Neural state speed\n(HD Euclidean)', fontsize=10)
    ax.set_title('Flow Speed vs Time\n(colored bands = behaviors)',
                 fontsize=12, fontweight='bold')
    ax.set_xlim(t_rel[0], t_rel[-1])

    # Panel H: 2D speed vs time
    ax = fig.add_subplot(3, 3, 8)
    ax.plot(t_speed, speed_2d, color='#1976D2', linewidth=1.5, alpha=0.8)
    ax.fill_between(t_speed, 0, speed_2d, alpha=0.15, color='#1976D2')

    # Zone shading
    for zone, color in ZONE_COLORS.items():
        if zone in behav_dict:
            zdata = behav_dict[zone][exc_mask]
            active = zdata > 0
            if active.any():
                starts = np.where(np.diff(np.concatenate([[0], active.astype(int)])) == 1)[0]
                ends = np.where(np.diff(np.concatenate([active.astype(int), [0]])) == -1)[0]
                for s, e in zip(starts, ends):
                    t_s = t_rel[s]
                    t_e = t_rel[min(e, len(t_rel) - 1)]
                    ax.axvspan(t_s, t_e, alpha=0.2, color=color, zorder=1)

    zone_handles = [Patch(facecolor=c, alpha=0.3, label=n) for n, c in ZONE_COLORS.items()]
    ax.legend(handles=zone_handles, fontsize=7, loc='upper right')
    ax.set_xlabel('Time within excursion (s)', fontsize=10)
    ax.set_ylabel('2D embedding speed', fontsize=10)
    ax.set_title('Embedding Speed vs Time\n(colored bands = zones)',
                 fontsize=12, fontweight='bold')
    ax.set_xlim(t_rel[0], t_rel[-1])

    # Panel I: Speed by behavior (boxplot)
    ax = fig.add_subplot(3, 3, 9)
    behav_speed_data = {}
    for i in range(len(speed_hd_smooth)):
        b = behav_labels[i]
        if b not in behav_speed_data:
            behav_speed_data[b] = []
        behav_speed_data[b].append(speed_hd_smooth[i])

    # Sort by median speed
    labels_sorted = sorted(behav_speed_data.keys(),
                            key=lambda k: np.median(behav_speed_data[k]))
    positions = np.arange(len(labels_sorted))
    bp_data = [behav_speed_data[k] for k in labels_sorted]
    bp_colors = [BEHAVIOR_COLORS.get(k, '#BDBDBD') for k in labels_sorted]

    bp = ax.boxplot(bp_data, positions=positions, vert=True, patch_artist=True,
                    widths=0.6, showfliers=False)
    for patch, color in zip(bp['boxes'], bp_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for patch in bp['medians']:
        patch.set_color('black')
        patch.set_linewidth(2)

    # Overlay individual points
    for i, (k, vals) in enumerate(zip(labels_sorted, bp_data)):
        jitter = np.random.normal(0, 0.08, len(vals))
        ax.scatter(positions[i] + jitter, vals,
                   c=BEHAVIOR_COLORS.get(k, '#BDBDBD'), s=8, alpha=0.4, zorder=5)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels_sorted, fontsize=8, rotation=45, ha='right')
    ax.set_ylabel('Neural state speed (HD)', fontsize=10)
    ax.set_title('Speed by Behavior State\n(is feeding slower?)',
                 fontsize=12, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("Flow State Analysis — Feeding & Digging Excursions")
    print("=" * 60)

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
    ]

    for region, unit_ids in [('rsp', rsp_ids), ('lha', lha_ids)]:
        region_label = region.upper()

        for bin_ms, smooth_ms, res_label in [(500, 1000, '500ms/1s')]:
            print(f"\n  {region_label} @ {res_label}")
            zscore, time_sec = bin_and_smooth(sorting, unit_ids, bin_ms, smooth_ms)
            behav_dict = load_behavior_timeseries(1, time_sec)

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
                    continue

                print(f"    Exc {eid} ({exc_label}): {n_pts} pts")

                n_nb = min(15, n_pts - 1)
                try:
                    iso2d = Isomap(n_components=2, n_neighbors=n_nb).fit_transform(exc_data)
                    iso3d = Isomap(n_components=3, n_neighbors=n_nb).fit_transform(exc_data)
                except:
                    iso2d = PCA(n_components=2).fit_transform(exc_data)
                    iso3d = PCA(n_components=3).fit_transform(exc_data)

                fig = make_flow_figure(
                    exc_data, iso2d, iso3d, exc_times, behav_dict, mask,
                    eid, region_label, len(unit_ids), res_label,
                    exc_label, erow['duration'])

                tag = 'feeding' if 'Feeding' in exc_label else 'digging'
                outpath = Path("figures") / f"flow_state_{tag}_exc{eid}_{region}.png"
                fig.savefig(outpath, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"    Saved: {outpath}")

    print("\nDone!")


if __name__ == "__main__":
    main()
