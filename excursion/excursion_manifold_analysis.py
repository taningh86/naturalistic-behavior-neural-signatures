"""
Excursion-Resolved Neural Manifolds
====================================
For each session and region (LHA, RSP):
  1. Build UMAP manifold from raw z-scored spike trains
  2. Map behavioral zone and excursion identity onto each time bin
  3. Compute per-excursion manifold statistics (centroid, spread, speed)
  4. Visualize manifold evolution within and across sessions

Outputs:
  - Per-session figures: figures/excursion_manifold_{region}_s{N}_{state}.png
  - Cross-session comparison: figures/excursion_manifold_comparison_{region}.png
  - Per-excursion stats: data/excursion_manifold_stats.csv
"""

import yaml
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
import umap
import spikeinterface.extractors as se
from scipy.ndimage import uniform_filter1d
from scipy.stats import mannwhitneyu
import warnings
import time as timer

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

BIN_SIZE_MS = 10
FS = 30000
BIN_SAMPLES = int(BIN_SIZE_MS * FS / 1000)
SMOOTH_BINS = 10  # 100ms smoothing
LHA_DEPTH_MAX = 1300
RSP_DEPTH_MIN = 1300
STRIDE = 5  # 50ms effective resolution
N_UMAP_MAX = 10000

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

ZONE_COLORS = {
    'Home': '#2196F3',
    'Ladder': '#FF9800',
    'Transition': '#9C27B0',
    'Arena': '#4CAF50',
    'None': '#BDBDBD',
}

STATE_COLORS = {'Fed': '#1976D2', 'Fasted': '#D32F2F'}


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


def bin_and_smooth(sorting, unit_ids):
    """Bin spikes at 10ms, smooth, z-score. Returns (zscore, smoothed, time_sec)."""
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

    for i in range(data.shape[1]):
        data[:, i] = uniform_filter1d(data[:, i], size=SMOOTH_BINS, mode='constant')

    means = data.mean(axis=0, keepdims=True)
    stds = data.std(axis=0, keepdims=True)
    stds[stds < 1e-8] = 1.0
    zscore_data = (data - means) / stds

    # Time in seconds for each bin
    time_sec = (np.arange(n_bins) * BIN_SIZE_MS / 1000) + (all_min / FS)

    return zscore_data, data, time_sec


def load_behavior_zones(beh_path: str):
    """Load zone time series from EthoVision CSV.

    Returns (beh_times, zone_ids) where zone_ids maps to:
        0=None, 1=Home, 2=Ladder, 3=Transition, 4=Arena
    """
    df = pd.read_csv(beh_path, header=None)
    labels = df.iloc[:, 0].values

    def get_row(name):
        idx = np.where(labels == name)[0]
        if len(idx) == 0:
            return None
        return df.iloc[idx[0], 1:].astype(float).values

    beh_times = get_row('Recording time')
    home = get_row('Home')
    ladder = get_row('Ladder')
    transition = get_row('Transition zone')
    arena = get_row('Foraging arena')

    # Priority: Arena > Transition > Ladder > Home > None
    zone_ids = np.zeros(len(beh_times), dtype=int)
    zone_ids[home == 1] = 1
    zone_ids[ladder == 1] = 2
    zone_ids[transition == 1] = 3
    zone_ids[arena == 1] = 4

    return beh_times, zone_ids


def map_zones_to_neural(neural_times, beh_times, beh_zones):
    """Map behavioral zones to neural time bins via nearest-neighbor."""
    zones = np.zeros(len(neural_times), dtype=int)
    for i, t in enumerate(neural_times):
        idx = np.searchsorted(beh_times, t)
        idx = min(idx, len(beh_zones) - 1)
        zones[i] = beh_zones[idx]
    return zones


def assign_excursions(times_sec, exc_df):
    """Map each neural time bin to an excursion ID (0 = at home / inter-excursion)."""
    exc_ids = np.zeros(len(times_sec), dtype=int)
    exc_labels = np.full(len(times_sec), '', dtype='U20')

    for _, row in exc_df.iterrows():
        mask = (times_sec >= row['start_time']) & (times_sec <= row['end_time'])
        exc_ids[mask] = int(row['excursion_id'])
        exc_labels[mask] = row['label']

    return exc_ids, exc_labels


def compute_excursion_stats(emb_2d, exc_ids, exc_df, times_sec, zones):
    """Compute per-excursion manifold statistics."""
    stats = []
    session_centroid = emb_2d.mean(axis=0)

    for _, row in exc_df.iterrows():
        eid = int(row['excursion_id'])
        mask = exc_ids == eid
        if mask.sum() < 3:
            continue

        pts = emb_2d[mask]
        centroid = pts.mean(axis=0)
        spread = np.sqrt(np.mean(np.sum((pts - centroid) ** 2, axis=1)))

        diffs = np.diff(pts, axis=0)
        step_dists = np.sqrt(np.sum(diffs ** 2, axis=1))
        path_length = np.sum(step_dists)
        speed = path_length / max(row['duration'], 0.1)

        dist_from_center = np.linalg.norm(centroid - session_centroid)

        # Time in arena during this excursion
        exc_zones = zones[mask]
        frac_arena = np.mean(exc_zones == 4) if len(exc_zones) > 0 else 0

        stats.append({
            'excursion_id': eid,
            'start_time': row['start_time'],
            'end_time': row['end_time'],
            'duration': row['duration'],
            'label': row['label'],
            'farthest_zone': row['farthest_zone'],
            'centroid_x': centroid[0],
            'centroid_y': centroid[1],
            'spread': spread,
            'path_length': path_length,
            'speed': speed,
            'dist_from_center': dist_from_center,
            'frac_in_arena': frac_arena,
            'n_points': int(mask.sum()),
        })

    return pd.DataFrame(stats)


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_session_manifold(emb_2d, zones, exc_ids, exc_labels, exc_stats,
                          region_label, sess_num, info, n_neurons, outpath):
    """Create per-session 2x3 figure."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 13))
    fig.suptitle(f"{region_label} Session {sess_num} — {info['state']} {info['phase']}\n"
                 f"Neural manifold by excursion ({n_neurons} neurons, "
                 f"{len(exc_stats)} excursions)",
                 fontsize=14, fontweight='bold')

    zone_name_map = {0: 'None', 1: 'Home', 2: 'Ladder', 3: 'Transition', 4: 'Arena'}
    zone_color_list = ['#BDBDBD', '#2196F3', '#FF9800', '#9C27B0', '#4CAF50']

    # --- Panel A: UMAP colored by behavioral zone ---
    ax = axes[0, 0]
    for zid, zname in zone_name_map.items():
        mask = zones == zid
        if mask.sum() > 0:
            ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1],
                       c=zone_color_list[zid], s=3, alpha=0.4,
                       label=zname, rasterized=True)
    ax.legend(fontsize=8, markerscale=3, loc='best')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_title('Colored by Behavioral Zone', fontsize=11)

    # --- Panel B: UMAP colored by excursion ID (temporal) ---
    ax = axes[0, 1]
    # Inter-excursion in gray
    home_mask = exc_ids == 0
    if home_mask.sum() > 0:
        ax.scatter(emb_2d[home_mask, 0], emb_2d[home_mask, 1],
                   c='#E0E0E0', s=2, alpha=0.2, rasterized=True, zorder=1)
    # Excursions colored by ID
    exc_mask = exc_ids > 0
    if exc_mask.sum() > 0:
        sc = ax.scatter(emb_2d[exc_mask, 0], emb_2d[exc_mask, 1],
                        c=exc_ids[exc_mask], cmap='turbo', s=3, alpha=0.5,
                        rasterized=True, zorder=2)
        plt.colorbar(sc, ax=ax, label='Excursion #', shrink=0.8)
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_title('Colored by Excursion Number', fontsize=11)

    # --- Panel C: Complete vs Incomplete excursions ---
    ax = axes[0, 2]
    home_mask = exc_ids == 0
    comp_mask = exc_labels == 'complete'
    inc_mask = exc_labels == 'incomplete'

    if home_mask.sum() > 0:
        ax.scatter(emb_2d[home_mask, 0], emb_2d[home_mask, 1],
                   c='#E0E0E0', s=2, alpha=0.15, rasterized=True, zorder=1)
    if comp_mask.sum() > 0:
        ax.scatter(emb_2d[comp_mask, 0], emb_2d[comp_mask, 1],
                   c='#1976D2', s=3, alpha=0.4, label='Complete',
                   rasterized=True, zorder=2)
    if inc_mask.sum() > 0:
        ax.scatter(emb_2d[inc_mask, 0], emb_2d[inc_mask, 1],
                   c='#D32F2F', s=3, alpha=0.4, label='Incomplete',
                   rasterized=True, zorder=2)
    ax.legend(fontsize=9, markerscale=3, loc='best')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_title('Complete vs Incomplete Excursions', fontsize=11)

    if len(exc_stats) == 0:
        for r in range(2):
            for c in range(3):
                if r > 0:
                    axes[r, c].set_visible(False)
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        fig.savefig(outpath, dpi=150, bbox_inches='tight')
        plt.close()
        return

    # --- Panel D: Centroid trajectory in UMAP space ---
    ax = axes[1, 0]
    cx = exc_stats['centroid_x'].values
    cy = exc_stats['centroid_y'].values
    colors_exc = np.arange(len(exc_stats))

    # Draw trajectory lines
    for i in range(len(cx) - 1):
        ax.plot([cx[i], cx[i + 1]], [cy[i], cy[i + 1]],
                color='gray', alpha=0.3, linewidth=0.8, zorder=1)
    sc = ax.scatter(cx, cy, c=colors_exc, cmap='turbo', s=30, zorder=2,
                    edgecolors='k', linewidths=0.3)
    plt.colorbar(sc, ax=ax, label='Excursion #', shrink=0.8)
    # Mark first and last
    ax.scatter(cx[0], cy[0], c='lime', s=80, marker='^', edgecolors='k',
               linewidths=1, zorder=3, label='First')
    ax.scatter(cx[-1], cy[-1], c='red', s=80, marker='v', edgecolors='k',
               linewidths=1, zorder=3, label='Last')
    ax.legend(fontsize=8, loc='best')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_title('Excursion Centroid Trajectory', fontsize=11)

    # --- Panel E: Spread over excursion number ---
    ax = axes[1, 1]
    exc_nums = exc_stats['excursion_id'].values
    spreads = exc_stats['spread'].values
    label_colors = ['#1976D2' if l == 'complete' else '#D32F2F'
                    for l in exc_stats['label'].values]
    ax.bar(exc_nums, spreads, color=label_colors, alpha=0.7, width=0.8)
    # Trend line
    if len(exc_nums) > 3:
        z = np.polyfit(exc_nums, spreads, 1)
        ax.plot(exc_nums, np.polyval(z, exc_nums), 'k--', alpha=0.6,
                label=f'trend: {z[0]:.3f}/exc')
        ax.legend(fontsize=8)
    ax.set_xlabel('Excursion #')
    ax.set_ylabel('Manifold Spread (UMAP units)')
    ax.set_title('Per-Excursion Spread', fontsize=11)
    # Legend
    legend_elements = [Line2D([0], [0], marker='s', color='w',
                              markerfacecolor='#1976D2', markersize=8,
                              label='Complete'),
                       Line2D([0], [0], marker='s', color='w',
                              markerfacecolor='#D32F2F', markersize=8,
                              label='Incomplete')]
    ax.legend(handles=legend_elements, fontsize=8, loc='upper right')

    # --- Panel F: Speed over excursion number ---
    ax = axes[1, 2]
    speeds = exc_stats['speed'].values
    ax.bar(exc_nums, speeds, color=label_colors, alpha=0.7, width=0.8)
    if len(exc_nums) > 3:
        z = np.polyfit(exc_nums, speeds, 1)
        ax.plot(exc_nums, np.polyval(z, exc_nums), 'k--', alpha=0.6,
                label=f'trend: {z[0]:.3f}/exc')
        ax.legend(fontsize=8)
    ax.set_xlabel('Excursion #')
    ax.set_ylabel('Manifold Speed (UMAP units/s)')
    ax.set_title('Per-Excursion Trajectory Speed', fontsize=11)
    ax.legend(handles=legend_elements, fontsize=8, loc='upper right')

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()


def plot_cross_session(all_stats, region_label, outpath):
    """Cross-session comparison figure."""
    if len(all_stats) == 0:
        return

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(f"{region_label} — Cross-Session Manifold Comparison\n"
                 f"Per-excursion statistics across sessions",
                 fontsize=14, fontweight='bold')

    # Group by session
    sessions = sorted(all_stats['session'].unique())

    # --- Panel A: Mean spread per session ---
    ax = axes[0, 0]
    for sess in sessions:
        sdf = all_stats[all_stats['session'] == sess]
        info = SESSION_INFO[sess]
        color = STATE_COLORS[info['state']]
        marker = 'o' if info['phase'] == 'Exploration' else 's'
        ax.scatter(sess, sdf['spread'].mean(), c=color, s=80, marker=marker,
                   edgecolors='k', linewidths=0.5, zorder=3)
        ax.errorbar(sess, sdf['spread'].mean(), yerr=sdf['spread'].sem(),
                    color=color, capsize=3, zorder=2)
    ax.set_xlabel('Session')
    ax.set_ylabel('Mean Manifold Spread')
    ax.set_title('Manifold Spread per Session', fontsize=11)
    ax.set_xticks(sessions)

    # --- Panel B: Mean speed per session ---
    ax = axes[0, 1]
    for sess in sessions:
        sdf = all_stats[all_stats['session'] == sess]
        info = SESSION_INFO[sess]
        color = STATE_COLORS[info['state']]
        marker = 'o' if info['phase'] == 'Exploration' else 's'
        ax.scatter(sess, sdf['speed'].mean(), c=color, s=80, marker=marker,
                   edgecolors='k', linewidths=0.5, zorder=3)
        ax.errorbar(sess, sdf['speed'].mean(), yerr=sdf['speed'].sem(),
                    color=color, capsize=3, zorder=2)
    ax.set_xlabel('Session')
    ax.set_ylabel('Mean Trajectory Speed')
    ax.set_title('Manifold Speed per Session', fontsize=11)
    ax.set_xticks(sessions)

    # --- Panel C: Centroid drift (distance from session center) ---
    ax = axes[0, 2]
    for sess in sessions:
        sdf = all_stats[all_stats['session'] == sess]
        info = SESSION_INFO[sess]
        color = STATE_COLORS[info['state']]
        marker = 'o' if info['phase'] == 'Exploration' else 's'
        ax.scatter(sess, sdf['dist_from_center'].mean(), c=color, s=80,
                   marker=marker, edgecolors='k', linewidths=0.5, zorder=3)
        ax.errorbar(sess, sdf['dist_from_center'].mean(),
                    yerr=sdf['dist_from_center'].sem(),
                    color=color, capsize=3, zorder=2)
    ax.set_xlabel('Session')
    ax.set_ylabel('Mean Dist from Session Centroid')
    ax.set_title('Centroid Drift per Session', fontsize=11)
    ax.set_xticks(sessions)

    # --- Panel D: Fed vs Fasted comparison (spread) ---
    ax = axes[1, 0]
    fed_spreads = []
    fasted_spreads = []
    for sess in sessions:
        sdf = all_stats[all_stats['session'] == sess]
        if SESSION_INFO[sess]['state'] == 'Fed':
            fed_spreads.append(sdf['spread'].mean())
        else:
            fasted_spreads.append(sdf['spread'].mean())

    positions = [1, 2]
    bp = ax.boxplot([fed_spreads, fasted_spreads], positions=positions,
                    widths=0.5, patch_artist=True)
    bp['boxes'][0].set_facecolor('#BBDEFB')
    bp['boxes'][1].set_facecolor('#FFCDD2')
    ax.scatter([1] * len(fed_spreads), fed_spreads, c='#1976D2', s=50,
               zorder=3, edgecolors='k', linewidths=0.5)
    ax.scatter([2] * len(fasted_spreads), fasted_spreads, c='#D32F2F', s=50,
               zorder=3, edgecolors='k', linewidths=0.5)
    ax.set_xticks(positions)
    ax.set_xticklabels(['Fed', 'Fasted'])
    ax.set_ylabel('Mean Manifold Spread')
    if len(fed_spreads) > 1 and len(fasted_spreads) > 1:
        stat, pval = mannwhitneyu(fed_spreads, fasted_spreads, alternative='two-sided')
        ax.set_title(f'Spread: Fed vs Fasted (p={pval:.3f})', fontsize=11)
    else:
        ax.set_title('Spread: Fed vs Fasted', fontsize=11)

    # --- Panel E: Fed vs Fasted comparison (speed) ---
    ax = axes[1, 1]
    fed_speeds = []
    fasted_speeds = []
    for sess in sessions:
        sdf = all_stats[all_stats['session'] == sess]
        if SESSION_INFO[sess]['state'] == 'Fed':
            fed_speeds.append(sdf['speed'].mean())
        else:
            fasted_speeds.append(sdf['speed'].mean())

    bp = ax.boxplot([fed_speeds, fasted_speeds], positions=positions,
                    widths=0.5, patch_artist=True)
    bp['boxes'][0].set_facecolor('#BBDEFB')
    bp['boxes'][1].set_facecolor('#FFCDD2')
    ax.scatter([1] * len(fed_speeds), fed_speeds, c='#1976D2', s=50,
               zorder=3, edgecolors='k', linewidths=0.5)
    ax.scatter([2] * len(fasted_speeds), fasted_speeds, c='#D32F2F', s=50,
               zorder=3, edgecolors='k', linewidths=0.5)
    ax.set_xticks(positions)
    ax.set_xticklabels(['Fed', 'Fasted'])
    ax.set_ylabel('Mean Trajectory Speed')
    if len(fed_speeds) > 1 and len(fasted_speeds) > 1:
        stat, pval = mannwhitneyu(fed_speeds, fasted_speeds, alternative='two-sided')
        ax.set_title(f'Speed: Fed vs Fasted (p={pval:.3f})', fontsize=11)
    else:
        ax.set_title('Speed: Fed vs Fasted', fontsize=11)

    # --- Panel F: Complete vs Incomplete excursion comparison ---
    ax = axes[1, 2]
    comp_spreads = all_stats[all_stats['label'] == 'complete']['spread'].values
    inc_spreads = all_stats[all_stats['label'] == 'incomplete']['spread'].values
    comp_speeds = all_stats[all_stats['label'] == 'complete']['speed'].values
    inc_speeds = all_stats[all_stats['label'] == 'incomplete']['speed'].values

    x = np.arange(2)
    width = 0.35
    ax.bar(x - width / 2,
           [np.mean(comp_spreads) if len(comp_spreads) else 0,
            np.mean(comp_speeds) if len(comp_speeds) else 0],
           width, label='Complete', color='#1976D2', alpha=0.7)
    ax.bar(x + width / 2,
           [np.mean(inc_spreads) if len(inc_spreads) else 0,
            np.mean(inc_speeds) if len(inc_speeds) else 0],
           width, label='Incomplete', color='#D32F2F', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(['Spread', 'Speed'])
    ax.legend(fontsize=9)
    ax.set_title('Complete vs Incomplete Excursions', fontsize=11)

    # Shared legend for state/phase
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#1976D2',
               markersize=10, label='Fed Exploration'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#1976D2',
               markersize=10, label='Fed Foraging'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#D32F2F',
               markersize=10, label='Fasted Exploration'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#D32F2F',
               markersize=10, label='Fasted Foraging'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4,
               fontsize=10, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("Excursion-Resolved Neural Manifolds")
    print("=" * 50)

    sp = paths_config['single_probe']['coordinates_1']['mouse01']['sessions']
    out_dir = Path("data")
    fig_dir = Path("figures")

    all_stats_list = []

    for region, region_label in [('lha', 'LHA'), ('rsp', 'RSP')]:
        print(f"\n{'=' * 60}")
        print(f"  {region_label}")
        print(f"{'=' * 60}")

        region_stats = []

        for sess_num, info in SESSION_INFO.items():
            key = f"session_{sess_num}"
            sc = sp[key]
            sorted_path = Path(sc['sorted'])
            beh_path = sc.get('behavior')

            if not sorted_path.exists():
                continue
            if not beh_path or not Path(beh_path).exists():
                print(f"  S{sess_num}: no behavior, skipping")
                continue

            # Load excursion CSV
            exc_file = out_dir / f"excursions_session_{sess_num}.csv"
            if not exc_file.exists():
                print(f"  S{sess_num}: no excursion CSV, skipping")
                continue

            exc_df = pd.read_csv(exc_file)

            print(f"\n  S{sess_num} ({info['state']}, {info['phase']})")

            # Load spike data
            sorting = se.read_kilosort(sorted_path)
            lha_ids, rsp_ids = get_good_units_by_region(sorted_path)
            unit_ids = lha_ids if region == 'lha' else rsp_ids
            if len(unit_ids) < 5:
                print(f"    Only {len(unit_ids)} neurons, skipping")
                continue

            print(f"    {len(unit_ids)} neurons")

            # Bin, smooth, z-score
            t0 = timer.time()
            zscore, smoothed, neural_times = bin_and_smooth(sorting, unit_ids)
            print(f"    Binned: {zscore.shape[0]} bins, "
                  f"{zscore.shape[0] * BIN_SIZE_MS / 1000:.0f}s "
                  f"({timer.time() - t0:.1f}s)")

            # Downsample
            data_sub = zscore[::STRIDE]
            times_sub = neural_times[::STRIDE]
            n_pts = len(data_sub)

            # Limit for UMAP
            n_umap = min(N_UMAP_MAX, n_pts)
            if n_pts > n_umap:
                idx = np.random.choice(n_pts, n_umap, replace=False)
                idx.sort()
            else:
                idx = np.arange(n_pts)

            data_umap = data_sub[idx]
            times_umap = times_sub[idx]

            # Load behavioral zones
            beh_times, beh_zones = load_behavior_zones(beh_path)
            zones_umap = map_zones_to_neural(times_umap, beh_times, beh_zones)

            # Assign excursions
            exc_ids_umap, exc_labels_umap = assign_excursions(times_umap, exc_df)

            # UMAP 2D
            print(f"    UMAP 2D ({n_umap} pts, {len(unit_ids)} dims)...")
            t0 = timer.time()
            reducer = umap.UMAP(n_components=2, n_neighbors=30,
                                min_dist=0.05, metric='euclidean',
                                random_state=42)
            emb_2d = reducer.fit_transform(data_umap)
            print(f"    done ({timer.time() - t0:.1f}s)")

            # Compute per-excursion stats
            exc_stats = compute_excursion_stats(
                emb_2d, exc_ids_umap, exc_df, times_umap, zones_umap)

            if len(exc_stats) > 0:
                exc_stats['session'] = sess_num
                exc_stats['region'] = region_label
                exc_stats['state'] = info['state']
                exc_stats['phase'] = info['phase']
                region_stats.append(exc_stats)

                print(f"    Excursions with stats: {len(exc_stats)} "
                      f"(complete={sum(exc_stats['label'] == 'complete')}, "
                      f"incomplete={sum(exc_stats['label'] == 'incomplete')})")
                print(f"    Mean spread: {exc_stats['spread'].mean():.2f}, "
                      f"Mean speed: {exc_stats['speed'].mean():.2f}")

            # Per-session figure
            outpath = fig_dir / (f"excursion_manifold_{region}_s{sess_num}_"
                                 f"{info['state'].lower()}.png")
            plot_session_manifold(emb_2d, zones_umap, exc_ids_umap,
                                 exc_labels_umap, exc_stats,
                                 region_label, sess_num, info,
                                 len(unit_ids), outpath)
            print(f"    Saved: {outpath}")

        # Cross-session comparison
        if region_stats:
            all_region = pd.concat(region_stats, ignore_index=True)
            all_stats_list.append(all_region)

            comp_path = fig_dir / f"excursion_manifold_comparison_{region}.png"
            plot_cross_session(all_region, region_label, comp_path)
            print(f"\n  Cross-session figure: {comp_path}")

    # Save all stats
    if all_stats_list:
        all_stats = pd.concat(all_stats_list, ignore_index=True)
        stats_path = out_dir / "excursion_manifold_stats.csv"
        all_stats.to_csv(stats_path, index=False)
        print(f"\nAll stats saved: {stats_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
