"""
Neural Manifold Subgroups -- Raw Neural Data
=============================================
Cluster neurons by pairwise correlation structure, then UMAP each
subgroup separately. Goal: find geometric patterns (rings, tori)
in subpopulations that get washed out in the full population.

Strategy:
  1. Compute pairwise neuron-neuron correlation from smoothed spike trains
  2. Hierarchical clustering -> 3-5 functional subgroups
  3. UMAP on each subgroup's spike trains (time x subgroup_neurons)
  4. Also UMAP on "subgroup-activity space" (time x n_subgroups, mean per group)
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
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import warnings
import time

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

BIN_SIZE_MS = 10
FS = 30000
BIN_SAMPLES = int(BIN_SIZE_MS * FS / 1000)
SMOOTH_BINS = 10  # 100ms smoothing window

LHA_DEPTH_MAX = 1300
RSP_DEPTH_MIN = 1300

STRIDE = 5  # every 50ms
N_SUBGROUPS = 4  # number of functional clusters
MIN_NEURONS_PER_SUBGROUP = 3  # skip subgroups smaller than this

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

# Colors for subgroups
SUBGROUP_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']


# =============================================================================
# DATA LOADING (reused from neural_manifold_topology.py)
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

    return zscore_data, data


def cluster_neurons(zscore_data: np.ndarray, n_clusters: int = N_SUBGROUPS):
    """Cluster neurons by pairwise correlation using hierarchical clustering."""
    # Correlation matrix across neurons (columns)
    corr = np.corrcoef(zscore_data.T)  # shape (n_neurons, n_neurons)
    # Convert correlation to distance: 1 - |corr| (anti-correlated neurons are also similar)
    dist = 1.0 - np.abs(corr)
    np.fill_diagonal(dist, 0)
    # Make symmetric and clip
    dist = (dist + dist.T) / 2
    dist = np.clip(dist, 0, None)

    # Hierarchical clustering
    condensed = squareform(dist)
    Z = linkage(condensed, method='ward')
    labels = fcluster(Z, t=n_clusters, criterion='maxclust')
    return labels, corr


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("Neural Manifold Subgroups -- Raw Spike Data")
    print(f"Bin: {BIN_SIZE_MS}ms, Smooth: {SMOOTH_BINS*BIN_SIZE_MS}ms, "
          f"Stride: {STRIDE} ({STRIDE*BIN_SIZE_MS}ms)")
    print(f"N subgroups: {N_SUBGROUPS}, Min neurons/subgroup: {MIN_NEURONS_PER_SUBGROUP}\n")

    sp = paths_config['single_probe']['coordinates_1']['mouse01']['sessions']

    # Focus on sessions with most neurons: S1 (fed) and S5 (fasted) for each region
    focus_sessions = [1, 5]

    for region, region_label in [('lha', 'LHA'), ('rsp', 'RSP')]:
        print(f"\n{'='*60}")
        print(f"  {region_label} -- Neuron Subgroup Manifolds")
        print(f"{'='*60}")

        for sess_num in focus_sessions:
            info = SESSION_INFO[sess_num]
            key = f"session_{sess_num}"
            sc = sp[key]
            sorted_path = Path(sc['sorted'])
            if not sorted_path.exists():
                continue

            sorting = se.read_kilosort(sorted_path)
            lha_ids, rsp_ids = get_good_units_by_region(sorted_path)
            unit_ids = lha_ids if region == 'lha' else rsp_ids
            if len(unit_ids) < 8:
                print(f"  S{sess_num}: only {len(unit_ids)} neurons, need >=8, skipping")
                continue

            print(f"\n  S{sess_num} ({info['state']}, {info['phase']}): "
                  f"{len(unit_ids)} neurons")

            # Bin, smooth, z-score
            zscore, smoothed = bin_and_smooth(sorting, unit_ids)
            print(f"    {zscore.shape[0]} time bins ({zscore.shape[0]*BIN_SIZE_MS/1000:.0f}s)")

            # Cluster neurons
            n_clust = min(N_SUBGROUPS, len(unit_ids) // MIN_NEURONS_PER_SUBGROUP)
            if n_clust < 2:
                print(f"    Too few neurons for subgroups, skipping")
                continue

            labels, corr_mat = cluster_neurons(zscore, n_clusters=n_clust)
            print(f"    Clustered into {n_clust} subgroups: "
                  + ", ".join([f"G{g}={np.sum(labels==g)}" for g in range(1, n_clust+1)]))

            # Downsample
            data_sub = zscore[::STRIDE]
            smoothed_sub = smoothed[::STRIDE]
            n_pts = len(data_sub)
            time_frac = np.arange(n_pts) / n_pts

            # Limit UMAP points
            n_umap = min(10000, n_pts)
            if n_pts > n_umap:
                idx = np.random.choice(n_pts, n_umap, replace=False)
                idx.sort()
            else:
                idx = np.arange(n_pts)

            time_umap = time_frac[idx]

            # =================================================================
            # Figure 1: Subgroup UMAP (one per subgroup) + correlation matrix
            # Layout: top row = correlation matrix + dendrogram-like sorted matrix
            #         bottom rows = one 2D + one 3D per subgroup
            # =================================================================
            valid_groups = []
            for g in range(1, n_clust + 1):
                mask = labels == g
                if np.sum(mask) >= MIN_NEURONS_PER_SUBGROUP:
                    valid_groups.append(g)

            n_valid = len(valid_groups)
            if n_valid < 2:
                print(f"    Only {n_valid} valid subgroup(s), skipping")
                continue

            # Run UMAP for each subgroup
            subgroup_emb2d = {}
            subgroup_emb3d = {}
            for g in valid_groups:
                mask = labels == g
                sub_data = data_sub[:, mask][idx]
                n_neurons_g = np.sum(mask)

                print(f"    Subgroup {g} ({n_neurons_g} neurons): UMAP 2D+3D...")
                t0 = time.time()

                # 2D
                reducer = umap.UMAP(n_components=2, n_neighbors=min(30, n_umap-1),
                                     min_dist=0.05, metric='euclidean',
                                     random_state=42)
                emb2d = reducer.fit_transform(sub_data)
                subgroup_emb2d[g] = emb2d

                # 3D
                reducer3d = umap.UMAP(n_components=3, n_neighbors=min(30, n_umap-1),
                                       min_dist=0.05, metric='euclidean',
                                       random_state=42)
                emb3d = reducer3d.fit_transform(sub_data)
                subgroup_emb3d[g] = emb3d

                print(f"      done ({time.time()-t0:.1f}s)")

            # Also run UMAP on subgroup-mean activity space
            print(f"    Subgroup-mean activity space UMAP...")
            subgroup_means = np.zeros((n_umap, n_valid), dtype=np.float32)
            for vi, g in enumerate(valid_groups):
                mask = labels == g
                subgroup_means[:, vi] = data_sub[:, mask][idx].mean(axis=1)

            reducer_sg = umap.UMAP(n_components=2, n_neighbors=min(30, n_umap-1),
                                    min_dist=0.05, metric='euclidean',
                                    random_state=42)
            emb_sg_2d = reducer_sg.fit_transform(subgroup_means)

            reducer_sg3d = umap.UMAP(n_components=3, n_neighbors=min(30, n_umap-1),
                                      min_dist=0.05, metric='euclidean',
                                      random_state=42)
            emb_sg_3d = reducer_sg3d.fit_transform(subgroup_means)
            print(f"    done")

            # =================================================================
            # FIGURE: (n_valid+1) rows x 3 cols
            # Row 0: Correlation matrix, sorted corr matrix, subgroup-mean UMAP
            # Rows 1-N: Per-subgroup 2D (time), 2D (rate), 3D
            # =================================================================
            n_rows = n_valid + 1
            fig = plt.figure(figsize=(18, 5 * n_rows))
            fig.suptitle(f"{region_label} Session {sess_num} -- {info['state']} {info['phase']}\n"
                         f"Neuron subgroup manifolds ({len(unit_ids)} neurons -> "
                         f"{n_valid} subgroups, raw spike data)",
                         fontsize=14, fontweight='bold')

            # --- Row 0, Col 1: Correlation matrix (sorted by cluster) ---
            ax = fig.add_subplot(n_rows, 3, 1)
            sort_idx = np.argsort(labels)
            sorted_corr = corr_mat[np.ix_(sort_idx, sort_idx)]
            im = ax.imshow(sorted_corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
            plt.colorbar(im, ax=ax, shrink=0.8)
            # Draw cluster boundaries
            cumsum = 0
            for g in valid_groups:
                n_g = np.sum(labels == g)
                ax.axhline(cumsum + n_g - 0.5, color='k', linewidth=1)
                ax.axvline(cumsum + n_g - 0.5, color='k', linewidth=1)
                ax.text(cumsum + n_g/2, -1.5, f'G{g}', ha='center', fontsize=9,
                        color=SUBGROUP_COLORS[g-1], fontweight='bold')
                cumsum += n_g
            ax.set_title('Neuron-Neuron Correlation\n(sorted by subgroup)', fontsize=11)
            ax.set_xlabel('Neuron')
            ax.set_ylabel('Neuron')

            # --- Row 0, Col 2: Subgroup-mean UMAP 2D ---
            ax = fig.add_subplot(n_rows, 3, 2)
            sc_plot = ax.scatter(emb_sg_2d[:, 0], emb_sg_2d[:, 1], c=time_umap,
                                  cmap='viridis', s=3, alpha=0.5, rasterized=True)
            plt.colorbar(sc_plot, ax=ax, label='Time', shrink=0.8)
            ax.set_xlabel('UMAP 1')
            ax.set_ylabel('UMAP 2')
            ax.set_title(f'Subgroup-Mean Activity Space\n({n_valid}D -> UMAP, colored by time)',
                         fontsize=11)

            # --- Row 0, Col 3: Subgroup-mean UMAP 3D ---
            ax = fig.add_subplot(n_rows, 3, 3, projection='3d')
            ax.scatter(emb_sg_3d[:, 0], emb_sg_3d[:, 1], emb_sg_3d[:, 2],
                       c=time_umap, cmap='viridis', s=2, alpha=0.3, rasterized=True)
            ax.view_init(elev=30, azim=45)
            ax.set_xlabel('U1', fontsize=8)
            ax.set_ylabel('U2', fontsize=8)
            ax.set_zlabel('U3', fontsize=8)
            ax.set_title('Subgroup-Mean 3D', fontsize=10)
            ax.tick_params(labelsize=7)

            # --- Per-subgroup rows ---
            for ri, g in enumerate(valid_groups):
                mask = labels == g
                n_g = np.sum(mask)
                row_base = (ri + 1) * 3

                emb2d = subgroup_emb2d[g]
                emb3d = subgroup_emb3d[g]

                # Mean firing rate for this subgroup
                sub_rate = smoothed_sub[:, mask][idx].mean(axis=1)

                # Col 1: 2D colored by time
                ax = fig.add_subplot(n_rows, 3, row_base + 1)
                sc_plot = ax.scatter(emb2d[:, 0], emb2d[:, 1], c=time_umap,
                                      cmap='viridis', s=3, alpha=0.5, rasterized=True)
                plt.colorbar(sc_plot, ax=ax, label='Time', shrink=0.8)
                ax.set_xlabel('UMAP 1')
                ax.set_ylabel('UMAP 2')
                ax.set_title(f'Subgroup {g} ({n_g} neurons) — Time',
                             fontsize=11, color=SUBGROUP_COLORS[g-1])

                # Col 2: 2D colored by subgroup firing rate
                ax = fig.add_subplot(n_rows, 3, row_base + 2)
                rpct = np.percentile(sub_rate, [2, 98])
                sc_plot = ax.scatter(emb2d[:, 0], emb2d[:, 1], c=sub_rate,
                                      cmap='hot_r', s=3, alpha=0.5, rasterized=True,
                                      vmin=rpct[0], vmax=rpct[1])
                plt.colorbar(sc_plot, ax=ax, label='Subgroup rate', shrink=0.8)
                ax.set_xlabel('UMAP 1')
                ax.set_ylabel('UMAP 2')
                ax.set_title(f'Subgroup {g} — Firing Rate',
                             fontsize=11, color=SUBGROUP_COLORS[g-1])

                # Col 3: 3D colored by time
                ax = fig.add_subplot(n_rows, 3, row_base + 3, projection='3d')
                ax.scatter(emb3d[:, 0], emb3d[:, 1], emb3d[:, 2],
                           c=time_umap, cmap='viridis', s=2, alpha=0.3,
                           rasterized=True)
                ax.view_init(elev=30, azim=45)
                ax.set_xlabel('U1', fontsize=8)
                ax.set_ylabel('U2', fontsize=8)
                ax.set_zlabel('U3', fontsize=8)
                ax.set_title(f'Subgroup {g} 3D', fontsize=10,
                             color=SUBGROUP_COLORS[g-1])
                ax.tick_params(labelsize=7)

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            outpath = (Path("figures") /
                       f"neural_subgroups_{region}_s{sess_num}_{info['state'].lower()}.png")
            fig.savefig(outpath, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"    Saved: {outpath}")

            # =================================================================
            # FIGURE 2: Pairwise subgroup UMAP
            # For each pair of subgroups, combine their neurons and UMAP
            # Color by which subgroup each neuron belongs to — reveals
            # whether subgroups separate or interleave in state space
            # =================================================================
            if n_valid >= 2:
                n_pairs = n_valid * (n_valid - 1) // 2
                fig2, axes2 = plt.subplots(1, n_pairs, figsize=(6*n_pairs, 5))
                if n_pairs == 1:
                    axes2 = [axes2]
                fig2.suptitle(f"{region_label} S{sess_num} {info['state']} — "
                              f"Pairwise Subgroup UMAP (colored by subgroup membership)",
                              fontsize=12, fontweight='bold')

                pi = 0
                for i_g, g1 in enumerate(valid_groups):
                    for g2 in valid_groups[i_g+1:]:
                        mask1 = labels == g1
                        mask2 = labels == g2
                        combined_mask = mask1 | mask2
                        sub_data = data_sub[:, combined_mask][idx]

                        # Track which neuron belongs to which group
                        neuron_labels = np.zeros(np.sum(combined_mask))
                        # Within combined_mask, first sum(mask1) are g1
                        # But we need to be careful about ordering
                        combined_indices = np.where(combined_mask)[0]
                        g1_set = set(np.where(mask1)[0])
                        neuron_assignment = np.array([1 if ci in g1_set else 2
                                                      for ci in combined_indices])

                        # UMAP
                        reducer = umap.UMAP(n_components=2, n_neighbors=min(30, n_umap-1),
                                             min_dist=0.05, random_state=42)
                        emb = reducer.fit_transform(sub_data)

                        # Color each timepoint by which subgroup dominates
                        # (mean z-scored activity of g1 neurons vs g2 neurons)
                        g1_cols = neuron_assignment == 1
                        g2_cols = neuron_assignment == 2
                        g1_act = sub_data[:, g1_cols].mean(axis=1)
                        g2_act = sub_data[:, g2_cols].mean(axis=1)
                        dominance = g1_act - g2_act  # positive = g1 dominant

                        ax = axes2[pi]
                        dpct = np.percentile(dominance, [2, 98])
                        sc_plot = ax.scatter(emb[:, 0], emb[:, 1], c=dominance,
                                              cmap='RdBu_r', s=3, alpha=0.5,
                                              rasterized=True,
                                              vmin=dpct[0], vmax=dpct[1])
                        plt.colorbar(sc_plot, ax=ax, shrink=0.8,
                                     label=f'G{g1} - G{g2} activity')
                        ax.set_xlabel('UMAP 1')
                        ax.set_ylabel('UMAP 2')
                        n1 = np.sum(mask1)
                        n2 = np.sum(mask2)
                        ax.set_title(f'G{g1}({n1}n) + G{g2}({n2}n)',
                                     fontsize=11)
                        pi += 1

                plt.tight_layout(rect=[0, 0, 1, 0.92])
                outpath2 = (Path("figures") /
                            f"neural_subgroups_pairs_{region}_s{sess_num}_{info['state'].lower()}.png")
                fig2.savefig(outpath2, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"    Saved: {outpath2}")

    print("\nDone!")


if __name__ == "__main__":
    main()
