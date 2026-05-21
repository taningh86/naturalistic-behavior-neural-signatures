"""
Excursion Manifold Evolution — Session 1 (Fed, Exploration)
===========================================================
Detailed within-session analysis showing how individual excursions
trace paths through the neural manifold and how this evolves.

Figures:
  1. Grid of individual excursion trajectories (small multiples)
  2. All excursions overlaid, colored early→late
  3. Early vs Late excursion comparison
"""

import yaml
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
import umap
import spikeinterface.extractors as se
from scipy.ndimage import uniform_filter1d
import warnings
import time as timer

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIG
# =============================================================================

BIN_SIZE_MS = 10
FS = 30000
BIN_SAMPLES = int(BIN_SIZE_MS * FS / 1000)
SMOOTH_BINS = 10
LHA_DEPTH_MAX = 1300
RSP_DEPTH_MIN = 1300
STRIDE = 5  # 50ms

with open("paths.yaml") as f:
    paths_config = yaml.safe_load(f)

ZONE_COLORS = {0: '#BDBDBD', 1: '#2196F3', 2: '#FF9800', 3: '#9C27B0', 4: '#4CAF50'}
ZONE_NAMES = {0: 'None', 1: 'Home', 2: 'Ladder', 3: 'Transition', 4: 'Arena'}


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
    time_sec = (np.arange(n_bins) * BIN_SIZE_MS / 1000) + (all_min / FS)
    return zscore_data, data, time_sec


def load_behavior_zones(beh_path):
    df = pd.read_csv(beh_path, header=None)
    labels = df.iloc[:, 0].values

    def get_row(name):
        idx = np.where(labels == name)[0]
        return df.iloc[idx[0], 1:].astype(float).values if len(idx) > 0 else None

    beh_times = get_row('Recording time')
    home = get_row('Home')
    ladder = get_row('Ladder')
    transition = get_row('Transition zone')
    arena = get_row('Foraging arena')

    zone_ids = np.zeros(len(beh_times), dtype=int)
    zone_ids[home == 1] = 1
    zone_ids[ladder == 1] = 2
    zone_ids[transition == 1] = 3
    zone_ids[arena == 1] = 4
    return beh_times, zone_ids


def map_zones_to_neural(neural_times, beh_times, beh_zones):
    zones = np.zeros(len(neural_times), dtype=int)
    for i, t in enumerate(neural_times):
        idx = min(np.searchsorted(beh_times, t), len(beh_zones) - 1)
        zones[i] = beh_zones[idx]
    return zones


def draw_trajectory(ax, pts, zones, alpha=0.8, lw=1.5, arrow_every=5):
    """Draw a trajectory colored by zone with direction arrows."""
    for i in range(len(pts) - 1):
        ax.plot([pts[i, 0], pts[i + 1, 0]], [pts[i, 1], pts[i + 1, 1]],
                color=ZONE_COLORS[zones[i]], alpha=alpha, linewidth=lw,
                solid_capstyle='round')
    # Direction arrows at intervals
    for i in range(0, len(pts) - 1, max(1, len(pts) // arrow_every)):
        dx = pts[min(i + 1, len(pts) - 1), 0] - pts[i, 0]
        dy = pts[min(i + 1, len(pts) - 1), 1] - pts[i, 1]
        if abs(dx) + abs(dy) > 0.01:
            ax.annotate('', xy=(pts[i, 0] + dx * 0.5, pts[i, 1] + dy * 0.5),
                        xytext=(pts[i, 0], pts[i, 1]),
                        arrowprops=dict(arrowstyle='->', color='k',
                                        lw=0.8, mutation_scale=8),
                        zorder=10)
    # Start and end markers
    ax.scatter(pts[0, 0], pts[0, 1], c='lime', s=40, marker='^',
               edgecolors='k', linewidths=0.8, zorder=11)
    ax.scatter(pts[-1, 0], pts[-1, 1], c='red', s=40, marker='v',
               edgecolors='k', linewidths=0.8, zorder=11)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("Excursion Manifold Evolution — Session 1 Fed")
    print("=" * 50)

    sp = paths_config['single_probe']['coordinates_1']['mouse01']['sessions']
    sc = sp['session_1']
    sorted_path = Path(sc['sorted'])
    beh_path = sc['behavior']

    sorting = se.read_kilosort(sorted_path)
    lha_ids, rsp_ids = get_good_units_by_region(sorted_path)

    # Load excursions
    exc_df = pd.read_csv("data/excursions_session_1.csv")
    complete_exc = exc_df[exc_df['label'] == 'complete'].reset_index(drop=True)
    print(f"  Total excursions: {len(exc_df)}, Complete: {len(complete_exc)}")

    # Load behavior zones
    beh_times, beh_zones = load_behavior_zones(beh_path)

    for region, region_label, unit_ids in [('lha', 'LHA', lha_ids),
                                            ('rsp', 'RSP', rsp_ids)]:
        if len(unit_ids) < 5:
            print(f"  {region_label}: too few neurons, skipping")
            continue

        print(f"\n  {region_label}: {len(unit_ids)} neurons")

        # Bin and smooth
        t0 = timer.time()
        zscore, smoothed, neural_times = bin_and_smooth(sorting, unit_ids)
        print(f"    Binned: {zscore.shape[0]} bins ({timer.time() - t0:.1f}s)")

        # Downsample
        data_sub = zscore[::STRIDE]
        times_sub = neural_times[::STRIDE]
        n_pts = len(data_sub)

        # UMAP — use ALL points (no subsampling) for consistent mapping
        n_umap = min(N_UMAP_MAX, n_pts) if 'N_UMAP_MAX' in dir() else min(10000, n_pts)
        if n_pts > n_umap:
            idx = np.random.choice(n_pts, n_umap, replace=False)
            idx.sort()
        else:
            idx = np.arange(n_pts)

        data_umap = data_sub[idx]
        times_umap = times_sub[idx]
        zones_umap = map_zones_to_neural(times_umap, beh_times, beh_zones)

        print(f"    UMAP 2D ({len(idx)} pts)...")
        t0 = timer.time()
        reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.05,
                            metric='euclidean', random_state=42)
        emb_2d = reducer.fit_transform(data_umap)
        print(f"    done ({timer.time() - t0:.1f}s)")

        # Map excursions to UMAP indices
        exc_ids = np.zeros(len(times_umap), dtype=int)
        for _, row in exc_df.iterrows():
            mask = (times_umap >= row['start_time']) & (times_umap <= row['end_time'])
            exc_ids[mask] = int(row['excursion_id'])

        # Get UMAP range for consistent axes
        x_range = [emb_2d[:, 0].min() - 1, emb_2d[:, 0].max() + 1]
        y_range = [emb_2d[:, 1].min() - 1, emb_2d[:, 1].max() + 1]

        # =============================================================
        # FIGURE 1: Grid of individual complete excursion trajectories
        # Select ~20 evenly spaced complete excursions
        # =============================================================
        n_show = min(20, len(complete_exc))
        show_indices = np.linspace(0, len(complete_exc) - 1, n_show, dtype=int)
        n_cols = 5
        n_rows = int(np.ceil(n_show / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
        fig.suptitle(f"{region_label} Session 1 (Fed Exploration) — "
                     f"Individual Excursion Trajectories\n"
                     f"({len(unit_ids)} neurons, {len(complete_exc)} complete excursions, "
                     f"showing {n_show} evenly spaced)",
                     fontsize=13, fontweight='bold')

        axes_flat = axes.flatten()
        for pi in range(len(axes_flat)):
            if pi >= n_show:
                axes_flat[pi].set_visible(False)
                continue

            ei = show_indices[pi]
            row = complete_exc.iloc[ei]
            eid = int(row['excursion_id'])
            ax = axes_flat[pi]

            # Background: full manifold in light gray
            ax.scatter(emb_2d[:, 0], emb_2d[:, 1], c='#F0F0F0', s=1,
                       alpha=0.3, rasterized=True, zorder=0)

            # This excursion's points
            mask = exc_ids == eid
            if mask.sum() < 3:
                ax.set_title(f"Exc {eid} (too few pts)", fontsize=9)
                ax.set_xlim(x_range)
                ax.set_ylim(y_range)
                continue

            pts = emb_2d[mask]
            z = zones_umap[mask]
            draw_trajectory(ax, pts, z, alpha=0.7, lw=1.2, arrow_every=4)

            ax.set_xlim(x_range)
            ax.set_ylim(y_range)
            ax.set_title(f"Exc {eid}  ({row['duration']:.0f}s, "
                         f"t={row['start_time']:.0f}-{row['end_time']:.0f}s)",
                         fontsize=8)
            ax.tick_params(labelsize=6)
            if pi % n_cols != 0:
                ax.set_yticklabels([])
            if pi < (n_rows - 1) * n_cols:
                ax.set_xticklabels([])

        # Add zone legend
        legend_elements = [
            Line2D([0], [0], color=ZONE_COLORS[1], lw=2, label='Home'),
            Line2D([0], [0], color=ZONE_COLORS[2], lw=2, label='Ladder'),
            Line2D([0], [0], color=ZONE_COLORS[3], lw=2, label='Transition'),
            Line2D([0], [0], color=ZONE_COLORS[4], lw=2, label='Arena'),
            Line2D([0], [0], marker='^', color='w', markerfacecolor='lime',
                   markersize=8, label='Start'),
            Line2D([0], [0], marker='v', color='w', markerfacecolor='red',
                   markersize=8, label='End'),
        ]
        fig.legend(handles=legend_elements, loc='lower center', ncol=6,
                   fontsize=10, bbox_to_anchor=(0.5, -0.01))

        plt.tight_layout(rect=[0, 0.03, 1, 0.94])
        outpath = Path("figures") / f"excursion_evolution_grid_{region}_s1_fed.png"
        fig.savefig(outpath, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    Saved: {outpath}")

        # =============================================================
        # FIGURE 2: All complete excursions overlaid, colored early→late
        # + early vs late comparison + centroid drift
        # =============================================================
        fig, axes = plt.subplots(2, 3, figsize=(20, 13))
        fig.suptitle(f"{region_label} Session 1 (Fed Exploration) — "
                     f"Excursion Evolution Through Session\n"
                     f"({len(unit_ids)} neurons, {len(complete_exc)} complete excursions)",
                     fontsize=14, fontweight='bold')

        # --- Panel A: All excursions overlaid, colored by excursion order ---
        ax = axes[0, 0]
        ax.scatter(emb_2d[:, 0], emb_2d[:, 1], c='#F0F0F0', s=1, alpha=0.2,
                   rasterized=True, zorder=0)
        cmap = plt.cm.coolwarm
        for ei in range(len(complete_exc)):
            row = complete_exc.iloc[ei]
            eid = int(row['excursion_id'])
            mask = exc_ids == eid
            if mask.sum() < 3:
                continue
            pts = emb_2d[mask]
            frac = ei / max(len(complete_exc) - 1, 1)
            color = cmap(frac)
            ax.plot(pts[:, 0], pts[:, 1], color=color, alpha=0.5,
                    linewidth=0.8, zorder=1)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
        sm.set_array([])
        cb = plt.colorbar(sm, ax=ax, shrink=0.8)
        cb.set_label('Session progress (early→late)', fontsize=9)
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.set_title('All Excursion Trajectories\n(blue=early, red=late)', fontsize=11)

        # --- Panel B: First 1/3 excursions ---
        ax = axes[0, 1]
        n_third = len(complete_exc) // 3
        ax.scatter(emb_2d[:, 0], emb_2d[:, 1], c='#F0F0F0', s=1, alpha=0.2,
                   rasterized=True, zorder=0)
        for ei in range(n_third):
            row = complete_exc.iloc[ei]
            eid = int(row['excursion_id'])
            mask = exc_ids == eid
            if mask.sum() < 3:
                continue
            pts = emb_2d[mask]
            z = zones_umap[mask]
            for j in range(len(pts) - 1):
                ax.plot([pts[j, 0], pts[j + 1, 0]], [pts[j, 1], pts[j + 1, 1]],
                        color=ZONE_COLORS[z[j]], alpha=0.6, linewidth=1)
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.set_title(f'Early Excursions (1-{n_third})\nColored by zone', fontsize=11)
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)

        # --- Panel C: Last 1/3 excursions ---
        ax = axes[0, 2]
        ax.scatter(emb_2d[:, 0], emb_2d[:, 1], c='#F0F0F0', s=1, alpha=0.2,
                   rasterized=True, zorder=0)
        for ei in range(len(complete_exc) - n_third, len(complete_exc)):
            row = complete_exc.iloc[ei]
            eid = int(row['excursion_id'])
            mask = exc_ids == eid
            if mask.sum() < 3:
                continue
            pts = emb_2d[mask]
            z = zones_umap[mask]
            for j in range(len(pts) - 1):
                ax.plot([pts[j, 0], pts[j + 1, 0]], [pts[j, 1], pts[j + 1, 1]],
                        color=ZONE_COLORS[z[j]], alpha=0.6, linewidth=1)
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.set_title(f'Late Excursions ({len(complete_exc)-n_third+1}-'
                     f'{len(complete_exc)})\nColored by zone', fontsize=11)
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)

        # --- Panel D: Centroid trajectory with time coloring ---
        ax = axes[1, 0]
        centroids = []
        for ei in range(len(complete_exc)):
            row = complete_exc.iloc[ei]
            eid = int(row['excursion_id'])
            mask = exc_ids == eid
            if mask.sum() < 3:
                continue
            pts = emb_2d[mask]
            centroids.append(pts.mean(axis=0))
        centroids = np.array(centroids)

        if len(centroids) > 1:
            # Color trajectory
            for i in range(len(centroids) - 1):
                frac = i / max(len(centroids) - 2, 1)
                ax.plot([centroids[i, 0], centroids[i + 1, 0]],
                        [centroids[i, 1], centroids[i + 1, 1]],
                        color=cmap(frac), alpha=0.5, linewidth=1, zorder=1)
            sc = ax.scatter(centroids[:, 0], centroids[:, 1],
                            c=np.arange(len(centroids)), cmap='coolwarm',
                            s=40, edgecolors='k', linewidths=0.3, zorder=2)
            plt.colorbar(sc, ax=ax, label='Excursion order', shrink=0.8)
            ax.scatter(centroids[0, 0], centroids[0, 1], c='lime', s=100,
                       marker='^', edgecolors='k', linewidths=1, zorder=3,
                       label='First')
            ax.scatter(centroids[-1, 0], centroids[-1, 1], c='red', s=100,
                       marker='v', edgecolors='k', linewidths=1, zorder=3,
                       label='Last')
            ax.legend(fontsize=8)
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.set_title('Excursion Centroid Drift\n(each dot = 1 excursion centroid)',
                      fontsize=11)

        # --- Panel E: Per-excursion path length and spread over time ---
        ax = axes[1, 1]
        path_lengths = []
        spreads = []
        exc_numbers = []
        for ei in range(len(complete_exc)):
            row = complete_exc.iloc[ei]
            eid = int(row['excursion_id'])
            mask = exc_ids == eid
            if mask.sum() < 3:
                continue
            pts = emb_2d[mask]
            diffs = np.diff(pts, axis=0)
            pl = np.sum(np.sqrt(np.sum(diffs ** 2, axis=1)))
            sp = np.sqrt(np.mean(np.sum((pts - pts.mean(axis=0)) ** 2, axis=1)))
            path_lengths.append(pl)
            spreads.append(sp)
            exc_numbers.append(ei + 1)

        ax2 = ax.twinx()
        ax.bar(exc_numbers, path_lengths, color='#1976D2', alpha=0.5,
               width=0.8, label='Path length')
        ax2.plot(exc_numbers, spreads, 'o-', color='#D32F2F', markersize=3,
                 alpha=0.7, label='Spread')
        ax.set_xlabel('Excursion # (chronological)')
        ax.set_ylabel('Path Length (UMAP units)', color='#1976D2')
        ax2.set_ylabel('Spread (UMAP units)', color='#D32F2F')
        ax.set_title('Path Length & Spread Over Session', fontsize=11)
        # Combined legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='upper right')

        # --- Panel F: Zone time fraction per excursion ---
        ax = axes[1, 2]
        zone_fracs = {z: [] for z in [1, 2, 3, 4]}
        exc_nums_for_zones = []
        for ei in range(len(complete_exc)):
            row = complete_exc.iloc[ei]
            eid = int(row['excursion_id'])
            mask = exc_ids == eid
            if mask.sum() < 3:
                continue
            z = zones_umap[mask]
            total = len(z)
            exc_nums_for_zones.append(ei + 1)
            for zid in [1, 2, 3, 4]:
                zone_fracs[zid].append(np.sum(z == zid) / total)

        if exc_nums_for_zones:
            bottom = np.zeros(len(exc_nums_for_zones))
            for zid in [1, 2, 3, 4]:
                vals = np.array(zone_fracs[zid])
                ax.bar(exc_nums_for_zones, vals, bottom=bottom, width=0.8,
                       color=ZONE_COLORS[zid], alpha=0.8, label=ZONE_NAMES[zid])
                bottom += vals
        ax.set_xlabel('Excursion # (chronological)')
        ax.set_ylabel('Fraction of time')
        ax.set_title('Zone Occupancy Per Excursion', fontsize=11)
        ax.legend(fontsize=8, loc='upper right')
        ax.set_ylim(0, 1.05)

        # Zone legend for panels B/C
        zone_legend = [
            Line2D([0], [0], color=ZONE_COLORS[1], lw=2, label='Home'),
            Line2D([0], [0], color=ZONE_COLORS[2], lw=2, label='Ladder'),
            Line2D([0], [0], color=ZONE_COLORS[3], lw=2, label='Transition'),
            Line2D([0], [0], color=ZONE_COLORS[4], lw=2, label='Arena'),
        ]
        axes[0, 1].legend(handles=zone_legend, fontsize=7, loc='best')

        plt.tight_layout(rect=[0, 0, 1, 0.93])
        outpath2 = Path("figures") / f"excursion_evolution_summary_{region}_s1_fed.png"
        fig.savefig(outpath2, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    Saved: {outpath2}")

        # =============================================================
        # FIGURE 3: Excursion trajectory "archetypal paths"
        # Align all complete excursions to a common normalized time
        # (0=start, 1=end) and show mean trajectory ± spread
        # =============================================================
        fig, axes3 = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f"{region_label} Session 1 — Mean Excursion Trajectory\n"
                     f"All complete excursions time-warped to [0,1]",
                     fontsize=13, fontweight='bold')

        n_interp = 50  # normalize each excursion to 50 points
        all_trajectories = []
        all_zone_sequences = []

        for ei in range(len(complete_exc)):
            row = complete_exc.iloc[ei]
            eid = int(row['excursion_id'])
            mask = exc_ids == eid
            if mask.sum() < 5:
                continue
            pts = emb_2d[mask]
            z = zones_umap[mask]

            # Interpolate to n_interp points
            t_orig = np.linspace(0, 1, len(pts))
            t_new = np.linspace(0, 1, n_interp)
            interp_x = np.interp(t_new, t_orig, pts[:, 0])
            interp_y = np.interp(t_new, t_orig, pts[:, 1])
            all_trajectories.append(np.column_stack([interp_x, interp_y]))

            # Interpolate zones (nearest neighbor)
            interp_z = np.array([z[min(int(ti * (len(z) - 1)), len(z) - 1)]
                                 for ti in t_new])
            all_zone_sequences.append(interp_z)

        all_traj = np.array(all_trajectories)  # (n_exc, n_interp, 2)
        all_zseq = np.array(all_zone_sequences)  # (n_exc, n_interp)
        mean_traj = all_traj.mean(axis=0)
        std_traj = all_traj.std(axis=0)

        # Panel A: Mean trajectory with spread envelope
        ax = axes3[0]
        ax.scatter(emb_2d[:, 0], emb_2d[:, 1], c='#F0F0F0', s=1, alpha=0.2,
                   rasterized=True, zorder=0)
        # Individual trajectories in light color
        for traj in all_traj[::3]:  # show every 3rd
            ax.plot(traj[:, 0], traj[:, 1], color='gray', alpha=0.1,
                    linewidth=0.5, zorder=1)
        # Mean trajectory
        t_colors = np.linspace(0, 1, n_interp)
        for i in range(n_interp - 1):
            ax.plot([mean_traj[i, 0], mean_traj[i + 1, 0]],
                    [mean_traj[i, 1], mean_traj[i + 1, 1]],
                    color=plt.cm.viridis(t_colors[i]), linewidth=3, zorder=3)
        ax.scatter(mean_traj[0, 0], mean_traj[0, 1], c='lime', s=100,
                   marker='^', edgecolors='k', linewidths=1.5, zorder=4,
                   label='Start (Home)')
        ax.scatter(mean_traj[-1, 0], mean_traj[-1, 1], c='red', s=100,
                   marker='v', edgecolors='k', linewidths=1.5, zorder=4,
                   label='End (Home)')
        ax.legend(fontsize=9)
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.set_title('Mean Trajectory (colored by time 0→1)', fontsize=11)

        # Panel B: Zone probability along normalized time
        ax = axes3[1]
        t_norm = np.linspace(0, 1, n_interp)
        for zid in [1, 2, 3, 4]:
            prob = (all_zseq == zid).mean(axis=0)
            ax.plot(t_norm, prob, color=ZONE_COLORS[zid], linewidth=2,
                    label=ZONE_NAMES[zid])
            ax.fill_between(t_norm, 0, prob, color=ZONE_COLORS[zid], alpha=0.15)
        ax.set_xlabel('Normalized excursion time (0=start, 1=end)')
        ax.set_ylabel('Zone probability')
        ax.set_title('Zone Occupancy Along Excursion', fontsize=11)
        ax.legend(fontsize=9)
        ax.set_ylim(0, 1)

        # Panel C: Early vs Late mean trajectories
        ax = axes3[2]
        ax.scatter(emb_2d[:, 0], emb_2d[:, 1], c='#F0F0F0', s=1, alpha=0.2,
                   rasterized=True, zorder=0)

        n_half = len(all_traj) // 2
        early_mean = all_traj[:n_half].mean(axis=0)
        late_mean = all_traj[n_half:].mean(axis=0)

        ax.plot(early_mean[:, 0], early_mean[:, 1], color='#1976D2',
                linewidth=2.5, label=f'Early (1-{n_half})', zorder=2)
        ax.plot(late_mean[:, 0], late_mean[:, 1], color='#D32F2F',
                linewidth=2.5, label=f'Late ({n_half+1}-{len(all_traj)})', zorder=2)

        # Start/end markers
        ax.scatter(early_mean[0, 0], early_mean[0, 1], c='#1976D2', s=80,
                   marker='^', edgecolors='k', linewidths=1, zorder=3)
        ax.scatter(early_mean[-1, 0], early_mean[-1, 1], c='#1976D2', s=80,
                   marker='v', edgecolors='k', linewidths=1, zorder=3)
        ax.scatter(late_mean[0, 0], late_mean[0, 1], c='#D32F2F', s=80,
                   marker='^', edgecolors='k', linewidths=1, zorder=3)
        ax.scatter(late_mean[-1, 0], late_mean[-1, 1], c='#D32F2F', s=80,
                   marker='v', edgecolors='k', linewidths=1, zorder=3)

        ax.legend(fontsize=9)
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.set_title('Early vs Late Mean Trajectory', fontsize=11)

        plt.tight_layout(rect=[0, 0, 1, 0.92])
        outpath3 = Path("figures") / f"excursion_evolution_mean_{region}_s1_fed.png"
        fig.savefig(outpath3, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    Saved: {outpath3}")

    print("\nDone!")


if __name__ == "__main__":
    main()
