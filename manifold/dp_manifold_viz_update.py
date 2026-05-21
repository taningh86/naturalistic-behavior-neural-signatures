"""
Regenerate manifold visualization with behavioral annotation coloring.
Adds row 3: feeding + digging_sand on PCA/UMAP.
"""

import yaml
import sys
import time as timer
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

from dp_avalanche_criticality import (
    get_good_units_p0, get_good_units_p1_lha,
    load_spike_times_for_region, FS,
)
import spikeinterface.extractors as se

with open("paths.yaml") as f:
    cfg = yaml.safe_load(f)
sessions_cfg = cfg["double_probe"]["coordinates_1"]["mouse01"]["sessions"]

BIN_MS = 50.0
SMOOTH_SIGMA = 1.0
figdir = Path("figures/manifold")


def load_and_preprocess(session_num, region):
    sval = sessions_cfg[f"session_{session_num}"]
    if region == 'ACA':
        sp = Path(sval['probe_0_aca']['sorted'])
        uids = get_good_units_p0(sp)
    else:
        sp = Path(sval['probe_1_lha_rsp']['sorted'])
        uids = get_good_units_p1_lha(sp)
    sorting = se.read_kilosort(sp)
    avail = set(sorting.get_unit_ids())
    uids = np.array([u for u in uids if u in avail])
    spike_dict = load_spike_times_for_region(sorting, uids)
    all_sp = np.concatenate(list(spike_dict.values()))
    dur = float(all_sp.max()) + 1.0
    dt = BIN_MS / 1000.0
    n_bins = int(dur / dt)
    bin_edges = np.arange(0, n_bins + 1) * dt
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    unit_ids = sorted(spike_dict.keys())
    matrix = np.zeros((n_bins, len(unit_ids)))
    for j, uid in enumerate(unit_ids):
        counts, _ = np.histogram(spike_dict[uid], bins=bin_edges)
        matrix[:, j] = gaussian_filter1d(counts.astype(float), sigma=SMOOTH_SIGMA)
    means = matrix.mean(axis=0)
    stds = matrix.std(axis=0)
    stds[stds == 0] = 1.0
    matrix = (matrix - means) / stds
    return matrix, bin_centers, len(unit_ids)


def load_behavioral(session_num):
    sval = sessions_cfg[f"session_{session_num}"]
    raw = pd.read_excel(sval['behavior'], header=None)
    col_names = list(raw.iloc[34].values)
    data = raw.iloc[36:].copy()
    data.columns = col_names
    data = data.reset_index(drop=True)
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    return data


def align_to_bins(behav_df, bin_centers):
    behav_times = behav_df['Trial time'].values.astype(float)
    indices = np.searchsorted(behav_times, bin_centers, side='left')
    indices = np.clip(indices, 0, len(behav_times) - 1)
    prev = np.clip(indices - 1, 0, len(behav_times) - 1)
    use_prev = np.abs(behav_times[prev] - bin_centers) < np.abs(behav_times[indices] - bin_centers)
    indices[use_prev] = prev[use_prev]
    return indices


def make_viz(session_num, region, matrix, bin_centers, behav_df, indices):
    print(f"  Computing PCA...", end='', flush=True)
    t0 = timer.time()
    pca = PCA(n_components=min(50, matrix.shape[1]))
    pca_coords = pca.fit_transform(matrix)
    print(f" {timer.time()-t0:.1f}s")

    print(f"  Computing UMAP...", end='', flush=True)
    t0 = timer.time()
    import umap
    reducer_2d = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
    umap_2d = reducer_2d.fit_transform(matrix)
    reducer_3d = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1, random_state=42)
    umap_3d = reducer_3d.fit_transform(matrix)
    print(f" {timer.time()-t0:.1f}s")

    # Extract behavioral variables
    vel = behav_df['Velocity(Center-point)'].values[indices].astype(float)
    vel_clean = np.where(np.isnan(vel), 0, vel)

    # Compartment
    pot_zone_cols = [c for c in behav_df.columns
                     if 'Zone(Pot-' in str(c) and ' zone' not in c
                     and 'Distance' not in str(c)]
    home_col = [c for c in behav_df.columns if 'Zone(Home' in str(c)
                and 'corner' not in c and 'Distance' not in str(c)]
    ladder_col = [c for c in behav_df.columns if 'Zone(ladder' in str(c)
                  and 'Distance' not in str(c)]
    compartment = np.full(len(indices), 'Arena', dtype=object)
    if home_col:
        compartment[behav_df[home_col[0]].values[indices] == 1] = 'Home'
    if ladder_col:
        compartment[behav_df[ladder_col[0]].values[indices] == 1] = 'Ladder'
    if pot_zone_cols:
        at_pot = np.zeros(len(indices), dtype=bool)
        for c in pot_zone_cols:
            at_pot |= (behav_df[c].values[indices] == 1)
        compartment[at_pot] = 'AtPot'

    # Scored behaviors
    feeding = behav_df['Feeding'].values[indices].astype(float)
    digging = behav_df['Digging sand'].values[indices].astype(float)

    # --- Figure: 3 rows x 4 cols ---
    fig = plt.figure(figsize=(20, 15))
    gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)

    # === Row 1: Time ===
    ax = fig.add_subplot(gs[0, 0])
    sc = ax.scatter(pca_coords[:, 0], pca_coords[:, 1], c=bin_centers, s=0.5,
                    alpha=0.3, cmap='viridis', rasterized=True)
    ax.set_xlabel(f'PC1 ({100*pca.explained_variance_ratio_[0]:.1f}%)')
    ax.set_ylabel(f'PC2 ({100*pca.explained_variance_ratio_[1]:.1f}%)')
    ax.set_title('PCA 2D (time)')
    plt.colorbar(sc, ax=ax, label='Time (s)')

    ax = fig.add_subplot(gs[0, 1], projection='3d')
    sc = ax.scatter(pca_coords[:, 0], pca_coords[:, 1], pca_coords[:, 2],
                    c=bin_centers, s=0.3, alpha=0.2, cmap='viridis', rasterized=True)
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); ax.set_zlabel('PC3')
    ax.set_title('PCA 3D (time)')

    ax = fig.add_subplot(gs[0, 2])
    sc = ax.scatter(umap_2d[:, 0], umap_2d[:, 1], c=bin_centers, s=0.5,
                    alpha=0.3, cmap='viridis', rasterized=True)
    ax.set_xlabel('UMAP1'); ax.set_ylabel('UMAP2')
    ax.set_title('UMAP 2D (time)')
    plt.colorbar(sc, ax=ax, label='Time (s)')

    ax = fig.add_subplot(gs[0, 3], projection='3d')
    sc = ax.scatter(umap_3d[:, 0], umap_3d[:, 1], umap_3d[:, 2],
                    c=bin_centers, s=0.3, alpha=0.2, cmap='viridis', rasterized=True)
    ax.set_xlabel('UMAP1'); ax.set_ylabel('UMAP2'); ax.set_zlabel('UMAP3')
    ax.set_title('UMAP 3D (time)')

    # === Row 2: Velocity + Compartment ===
    vmax = np.percentile(vel_clean[vel_clean > 0], 95) if np.any(vel_clean > 0) else 1

    ax = fig.add_subplot(gs[1, 0])
    sc = ax.scatter(pca_coords[:, 0], pca_coords[:, 1], c=vel_clean, s=0.5,
                    alpha=0.3, cmap='hot', rasterized=True, vmin=0, vmax=vmax)
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
    ax.set_title('PCA 2D (velocity)')
    plt.colorbar(sc, ax=ax, label='cm/s')

    ax = fig.add_subplot(gs[1, 1])
    sc = ax.scatter(umap_2d[:, 0], umap_2d[:, 1], c=vel_clean, s=0.5,
                    alpha=0.3, cmap='hot', rasterized=True, vmin=0, vmax=vmax)
    ax.set_xlabel('UMAP1'); ax.set_ylabel('UMAP2')
    ax.set_title('UMAP 2D (velocity)')
    plt.colorbar(sc, ax=ax, label='cm/s')

    comp_colors = {'Home': 'blue', 'Ladder': 'green', 'Arena': 'gray', 'AtPot': 'red'}
    ax = fig.add_subplot(gs[1, 2])
    for label, color in comp_colors.items():
        mask = compartment == label
        if mask.sum() > 0:
            ax.scatter(pca_coords[mask, 0], pca_coords[mask, 1], c=color, s=0.5,
                       alpha=0.3, label=label, rasterized=True)
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
    ax.set_title('PCA 2D (compartment)')
    ax.legend(markerscale=10, fontsize=8)

    ax = fig.add_subplot(gs[1, 3])
    for label, color in comp_colors.items():
        mask = compartment == label
        if mask.sum() > 0:
            ax.scatter(umap_2d[mask, 0], umap_2d[mask, 1], c=color, s=0.5,
                       alpha=0.3, label=label, rasterized=True)
    ax.set_xlabel('UMAP1'); ax.set_ylabel('UMAP2')
    ax.set_title('UMAP 2D (compartment)')
    ax.legend(markerscale=10, fontsize=8)

    # === Row 3: Feeding + Digging ===
    feed_mask = feeding == 1
    dig_mask = digging == 1

    # PCA - Feeding
    ax = fig.add_subplot(gs[2, 0])
    ax.scatter(pca_coords[~feed_mask, 0], pca_coords[~feed_mask, 1],
               c='lightgray', s=0.3, alpha=0.15, rasterized=True, label='Other')
    ax.scatter(pca_coords[feed_mask, 0], pca_coords[feed_mask, 1],
               c='darkorange', s=1.5, alpha=0.6, rasterized=True, label=f'Feeding ({feed_mask.sum()})')
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
    ax.set_title('PCA 2D (feeding)')
    ax.legend(markerscale=5, fontsize=8)

    # UMAP - Feeding
    ax = fig.add_subplot(gs[2, 1])
    ax.scatter(umap_2d[~feed_mask, 0], umap_2d[~feed_mask, 1],
               c='lightgray', s=0.3, alpha=0.15, rasterized=True, label='Other')
    ax.scatter(umap_2d[feed_mask, 0], umap_2d[feed_mask, 1],
               c='darkorange', s=1.5, alpha=0.6, rasterized=True, label=f'Feeding ({feed_mask.sum()})')
    ax.set_xlabel('UMAP1'); ax.set_ylabel('UMAP2')
    ax.set_title('UMAP 2D (feeding)')
    ax.legend(markerscale=5, fontsize=8)

    # PCA - Digging
    ax = fig.add_subplot(gs[2, 2])
    ax.scatter(pca_coords[~dig_mask, 0], pca_coords[~dig_mask, 1],
               c='lightgray', s=0.3, alpha=0.15, rasterized=True, label='Other')
    ax.scatter(pca_coords[dig_mask, 0], pca_coords[dig_mask, 1],
               c='purple', s=1.5, alpha=0.6, rasterized=True, label=f'Digging ({dig_mask.sum()})')
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
    ax.set_title('PCA 2D (digging)')
    ax.legend(markerscale=5, fontsize=8)

    # UMAP - Digging
    ax = fig.add_subplot(gs[2, 3])
    ax.scatter(umap_2d[~dig_mask, 0], umap_2d[~dig_mask, 1],
               c='lightgray', s=0.3, alpha=0.15, rasterized=True, label='Other')
    ax.scatter(umap_2d[dig_mask, 0], umap_2d[dig_mask, 1],
               c='purple', s=1.5, alpha=0.6, rasterized=True, label=f'Digging ({dig_mask.sum()})')
    ax.set_xlabel('UMAP1'); ax.set_ylabel('UMAP2')
    ax.set_title('UMAP 2D (digging)')
    ax.legend(markerscale=5, fontsize=8)

    fig.suptitle(f'Manifold Visualization -- S{session_num} {region}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig_path = figdir / f"S{session_num}_{region}_manifold_viz.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {fig_path}")


def main():
    session_num = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    sval = sessions_cfg[f"session_{session_num}"]
    print(f"Manifold Viz Update -- S{session_num} ({sval['state']}/{sval['phase']})")

    behav_df = load_behavioral(session_num)

    for region in ['ACA', 'LHA']:
        print(f"\n  {region}:")
        matrix, bin_centers, n_units = load_and_preprocess(session_num, region)
        print(f"    {n_units} units, {matrix.shape[0]} bins")
        indices = align_to_bins(behav_df, bin_centers)
        make_viz(session_num, region, matrix, bin_centers, behav_df, indices)

    print("\nDone.")


if __name__ == '__main__':
    main()
