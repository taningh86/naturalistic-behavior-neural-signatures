"""
Neural Manifold Topology -- Raw Neural Data
============================================
UMAP on actual z-scored spike trains (not model hidden states).
Reveals the true geometric structure of the neural population manifold.

For each session/region: bin spikes at 10ms, smooth with 100ms window,
then UMAP in 2D and 3D. Color by time and firing rate to reveal
rings, tori, or other geometric patterns.
"""

import yaml
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import umap
import spikeinterface.extractors as se
from scipy.ndimage import uniform_filter1d
import warnings
import time

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

BIN_SIZE_MS = 10
FS = 30000
BIN_SAMPLES = int(BIN_SIZE_MS * FS / 1000)
SMOOTH_BINS = 10  # 100ms smoothing window (10 bins x 10ms)

LHA_DEPTH_MAX = 1300
RSP_DEPTH_MIN = 1300

# Downsample stride (every Nth bin) to keep UMAP manageable
STRIDE = 5  # every 50ms

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


def bin_and_smooth(sorting, unit_ids, smooth_bins=SMOOTH_BINS):
    """Bin spikes at 10ms, smooth with uniform filter, z-score."""
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

    # Smooth each neuron with 100ms window
    for i in range(data.shape[1]):
        data[:, i] = uniform_filter1d(data[:, i], size=smooth_bins, mode='constant')

    # Z-score
    means = data.mean(axis=0, keepdims=True)
    stds = data.std(axis=0, keepdims=True)
    stds[stds < 1e-8] = 1.0
    zscore_data = (data - means) / stds

    return zscore_data, data  # zscore and smoothed raw


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("Neural Manifold Topology -- Raw Spike Data")
    print(f"Bin: {BIN_SIZE_MS}ms, Smooth: {SMOOTH_BINS*BIN_SIZE_MS}ms, "
          f"Stride: {STRIDE} ({STRIDE*BIN_SIZE_MS}ms)\n")

    sp = paths_config['single_probe']['coordinates_1']['mouse01']['sessions']

    for region, region_label in [('lha', 'LHA'), ('rsp', 'RSP')]:
        print(f"\n{'='*60}")
        print(f"  {region_label} -- Raw Neural Manifold")
        print(f"{'='*60}")

        for sess_num, info in SESSION_INFO.items():
            key = f"session_{sess_num}"
            sc = sp[key]
            sorted_path = Path(sc['sorted'])
            if not sorted_path.exists():
                continue

            sorting = se.read_kilosort(sorted_path)
            lha_ids, rsp_ids = get_good_units_by_region(sorted_path)
            unit_ids = lha_ids if region == 'lha' else rsp_ids
            if len(unit_ids) < 5:
                print(f"  S{sess_num}: only {len(unit_ids)} neurons, skipping")
                continue

            print(f"\n  S{sess_num} ({info['state']}, {info['phase']}): "
                  f"{len(unit_ids)} neurons")

            # Bin, smooth, z-score
            zscore, smoothed = bin_and_smooth(sorting, unit_ids)
            print(f"    {zscore.shape[0]} time bins ({zscore.shape[0]*BIN_SIZE_MS/1000:.0f}s)")

            # Downsample
            data_sub = zscore[::STRIDE]
            n_pts = len(data_sub)
            time_frac = np.arange(n_pts) / n_pts  # 0 to 1

            # Population firing rate (mean across neurons)
            pop_rate = smoothed[::STRIDE].mean(axis=1)

            # Limit UMAP points
            n_umap = min(10000, n_pts)
            if n_pts > n_umap:
                idx = np.random.choice(n_pts, n_umap, replace=False)
                idx.sort()  # keep temporal order
            else:
                idx = np.arange(n_pts)

            data_umap = data_sub[idx]
            time_umap = time_frac[idx]
            rate_umap = pop_rate[idx]

            # UMAP 2D
            print(f"    UMAP 2D ({n_umap} points, {len(unit_ids)} dims)...")
            t0 = time.time()
            reducer_2d = umap.UMAP(n_components=2, n_neighbors=30,
                                    min_dist=0.05, metric='euclidean',
                                    random_state=42)
            emb_2d = reducer_2d.fit_transform(data_umap)
            print(f"    2D done ({time.time()-t0:.1f}s)")

            # UMAP 3D
            print(f"    UMAP 3D...")
            t0 = time.time()
            reducer_3d = umap.UMAP(n_components=3, n_neighbors=30,
                                    min_dist=0.05, metric='euclidean',
                                    random_state=42)
            emb_3d = reducer_3d.fit_transform(data_umap)
            print(f"    3D done ({time.time()-t0:.1f}s)")

            # =============================================================
            # Figure: 2x3
            # Row 1: 2D colored by time, pop rate, individual neuron
            # Row 2: 3D from 3 angles colored by time
            # =============================================================
            fig = plt.figure(figsize=(18, 11))
            fig.suptitle(f"{region_label} Session {sess_num} -- "
                         f"{info['state']} {info['phase']}\n"
                         f"Raw neural manifold ({len(unit_ids)} neurons, "
                         f"{BIN_SIZE_MS}ms bins, {SMOOTH_BINS*BIN_SIZE_MS}ms smooth)",
                         fontsize=13, fontweight='bold')

            # 2D: colored by time
            ax = fig.add_subplot(2, 3, 1)
            sc = ax.scatter(emb_2d[:, 0], emb_2d[:, 1], c=time_umap,
                            cmap='viridis', s=3, alpha=0.5, rasterized=True)
            plt.colorbar(sc, ax=ax, label='Time (fraction)', shrink=0.8)
            ax.set_xlabel('UMAP 1')
            ax.set_ylabel('UMAP 2')
            ax.set_title('Colored by Time', fontsize=11)

            # 2D: colored by population rate
            ax = fig.add_subplot(2, 3, 2)
            rate_pct = np.percentile(rate_umap, [2, 98])
            sc = ax.scatter(emb_2d[:, 0], emb_2d[:, 1], c=rate_umap,
                            cmap='hot_r', s=3, alpha=0.5, rasterized=True,
                            vmin=rate_pct[0], vmax=rate_pct[1])
            plt.colorbar(sc, ax=ax, label='Pop. firing rate', shrink=0.8)
            ax.set_xlabel('UMAP 1')
            ax.set_ylabel('UMAP 2')
            ax.set_title('Colored by Population Rate', fontsize=11)

            # 2D: colored by top-variance neuron
            ax = fig.add_subplot(2, 3, 3)
            neuron_vars = data_umap.var(axis=0)
            top_neuron = np.argmax(neuron_vars)
            neuron_act = data_umap[:, top_neuron]
            n_pct = np.percentile(neuron_act, [2, 98])
            sc = ax.scatter(emb_2d[:, 0], emb_2d[:, 1], c=neuron_act,
                            cmap='coolwarm', s=3, alpha=0.5, rasterized=True,
                            vmin=n_pct[0], vmax=n_pct[1])
            plt.colorbar(sc, ax=ax, label=f'Neuron {top_neuron} activity', shrink=0.8)
            ax.set_xlabel('UMAP 1')
            ax.set_ylabel('UMAP 2')
            ax.set_title(f'Colored by Top-Var Neuron (#{top_neuron})', fontsize=11)

            # 3D from 3 angles
            views = [(30, 45, 'Front-left'), (30, 135, 'Front-right'),
                     (80, 45, 'Top-down')]
            for vi, (elev, azim, vname) in enumerate(views):
                ax = fig.add_subplot(2, 3, 4 + vi, projection='3d')
                sc = ax.scatter(emb_3d[:, 0], emb_3d[:, 1], emb_3d[:, 2],
                                c=time_umap, cmap='viridis', s=2, alpha=0.3,
                                rasterized=True)
                ax.view_init(elev=elev, azim=azim)
                ax.set_xlabel('U1', fontsize=8)
                ax.set_ylabel('U2', fontsize=8)
                ax.set_zlabel('U3', fontsize=8)
                ax.set_title(f'3D: {vname}', fontsize=10)
                ax.tick_params(labelsize=7)

            plt.tight_layout(rect=[0, 0, 1, 0.93])
            outpath = (Path("figures") /
                       f"neural_manifold_{region}_s{sess_num}_{info['state'].lower()}.png")
            fig.savefig(outpath, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"    Saved: {outpath}")

    print("\nDone!")


if __name__ == "__main__":
    main()
