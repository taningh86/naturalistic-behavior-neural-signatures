"""
Manifold Analysis of Feeding & Digging Excursions
==================================================
Same visualization pipeline as loop excursions:
  - 3D tube trajectories (PCA, Isomap) colored by time and behavior
  - Persistent homology (persistence diagram, barcode)
  - Circular coordinates + recurrence matrix
  - Behavior-colored overlays (feeding, digging, exploration periods)

Plus comparison: feeding/digging excursions vs loop excursions vs typical.
"""

import yaml
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, ListedColormap
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from scipy.interpolate import make_interp_spline
from scipy.ndimage import uniform_filter1d
from scipy.spatial.distance import pdist, squareform
from gtda.homology import VietorisRipsPersistence
import spikeinterface.extractors as se
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIG
# =============================================================================

FS = 30000
LHA_DEPTH_MAX = 1300
RSP_DEPTH_MIN = 1300

with open("paths.yaml") as f:
    paths_config = yaml.safe_load(f)

# Excursions to analyze
FEEDING_EXCURSIONS = [
    {'exc_id': 81, 'label': 'Feeding (64% of time)', 'duration': 111.6},
]
DIGGING_EXCURSIONS = [
    {'exc_id': 57, 'label': 'Digging (6% of time) + H2 void 16.9x', 'duration': 37.8},
]
# Typical non-feeding, non-loop excursions for comparison
TYPICAL_EXCURSIONS = [
    {'exc_id': 80, 'label': 'Typical (no feeding/loops)', 'duration': 32.8},
    {'exc_id': 65, 'label': 'Typical (no feeding/loops)', 'duration': 31.4},
    {'exc_id': 84, 'label': 'Typical (no feeding/loops)', 'duration': 31.3},
]
# Loop excursions for comparison
LOOP_EXCURSIONS = [
    {'exc_id': 90, 'label': 'H1 loop 14.8x', 'duration': 6.5},
    {'exc_id': 89, 'label': 'H1 loop 6.7x', 'duration': 9.5},
]

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


def load_behavior_timeseries(session_num, time_sec):
    """Load all behavioral variables aligned to neural time bins."""
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
        # Map to neural time bins (behavior is 100ms bins from t=0)
        aligned = np.zeros(len(time_sec))
        for ti, t in enumerate(time_sec):
            bi = int(t / 0.1)
            if 0 <= bi < len(row_data) and not np.isnan(row_data[bi]):
                aligned[ti] = row_data[bi]
        result[name] = aligned

    return result


def interpolate_trajectory(coords, n_interp=300):
    n = len(coords)
    if n < 4:
        return coords, np.linspace(0, 1, n)
    t_orig = np.linspace(0, 1, n)
    t_fine = np.linspace(0, 1, n_interp)
    k = min(3, n - 1)
    coords_fine = np.zeros((n_interp, coords.shape[1]))
    for d in range(coords.shape[1]):
        spl = make_interp_spline(t_orig, coords[:, d], k=k)
        coords_fine[:, d] = spl(t_fine)
    return coords_fine, t_fine


def run_persistence(data, max_pts=500, n_pca=15, max_dim=2):
    n_comp = min(n_pca, data.shape[1], len(data) - 1)
    pca = PCA(n_components=n_comp)
    data_pca = pca.fit_transform(data)
    var_exp = pca.explained_variance_ratio_.cumsum()
    n_keep = min(np.searchsorted(var_exp, 0.95) + 1, n_comp)
    n_keep = max(n_keep, 3)
    data_pca = data_pca[:, :n_keep]
    if len(data_pca) > max_pts:
        idx = np.random.choice(len(data_pca), max_pts, replace=False)
        idx.sort()
        data_pca = data_pca[idx]
    VR = VietorisRipsPersistence(
        homology_dimensions=list(range(max_dim + 1)),
        max_edge_length=np.inf, n_jobs=-1)
    diagrams = VR.fit_transform(data_pca[np.newaxis, :, :])[0]
    return diagrams, data_pca, n_keep, var_exp[n_keep - 1]


DIM_COLORS = {0: '#1976D2', 1: '#D32F2F', 2: '#4CAF50'}


# =============================================================================
# PLOTTING
# =============================================================================

def plot_tube_3d(ax, coords, color_vals, cmap, title, n_interp=300):
    coords_fine, t_fine = interpolate_trajectory(coords, n_interp)
    for i in range(len(coords_fine) - 1):
        c = plt.get_cmap(cmap)(color_vals[min(int(t_fine[i] * (len(color_vals) - 1)),
                                                len(color_vals) - 1)])
        ax.plot([coords_fine[i, 0], coords_fine[i+1, 0]],
                [coords_fine[i, 1], coords_fine[i+1, 1]],
                [coords_fine[i, 2], coords_fine[i+1, 2]],
                color=c, linewidth=3, alpha=0.85, solid_capstyle='round')
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
               c='white', s=15, edgecolors='gray', linewidths=0.5, alpha=0.5, zorder=5)
    ax.scatter(*coords[0], c='lime', s=200, marker='*',
               edgecolors='black', linewidths=1.5, zorder=10)
    ax.scatter(*coords[-1], c='red', s=150, marker='X',
               edgecolors='black', linewidths=1.5, zorder=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('Dim 1', fontsize=8)
    ax.set_ylabel('Dim 2', fontsize=8)
    ax.set_zlabel('Dim 3', fontsize=8)
    ax.tick_params(labelsize=7)


def plot_behavior_colored_3d(ax, coords, behav_dict, exc_mask, title):
    """3D scatter colored by active behavior."""
    n = len(coords)
    # Assign each time point to dominant behavior
    behav_labels = np.full(n, 'Other', dtype=object)
    for bname in ['Feeding', 'Digging', 'Grooming', 'Quick arena exploration',
                   'Arena wall exploration', 'Transition wall exploration',
                   'Hesitant exploration', 'Quick one loop at home']:
        if bname in behav_dict:
            bdata = behav_dict[bname][exc_mask]
            active = bdata > 0
            behav_labels[active] = bname

    # Plot
    for bname, color in BEHAVIOR_COLORS.items():
        mask = behav_labels == bname
        if mask.any():
            ax.scatter(coords[mask, 0], coords[mask, 1], coords[mask, 2],
                       c=color, s=30, alpha=0.8, label=bname, zorder=5)
    other = behav_labels == 'Other'
    if other.any():
        ax.scatter(coords[other, 0], coords[other, 1], coords[other, 2],
                   c='#BDBDBD', s=12, alpha=0.4, label='Other/unlabeled', zorder=2)

    # Trajectory line
    ax.plot(coords[:, 0], coords[:, 1], coords[:, 2],
            color='gray', alpha=0.15, linewidth=0.5, zorder=1)

    ax.scatter(*coords[0], c='lime', s=200, marker='*',
               edgecolors='black', linewidths=1.5, zorder=10)
    ax.scatter(*coords[-1], c='red', s=150, marker='X',
               edgecolors='black', linewidths=1.5, zorder=10)

    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('Dim 1', fontsize=8)
    ax.set_ylabel('Dim 2', fontsize=8)
    ax.set_zlabel('Dim 3', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6, loc='upper left', markerscale=1.5)


def plot_behavior_colored_2d(ax, coords_2d, behav_dict, exc_mask, title):
    """2D scatter colored by active behavior with trajectory."""
    n = len(coords_2d)
    behav_labels = np.full(n, 'Other', dtype=object)
    for bname in ['Feeding', 'Digging', 'Grooming', 'Quick arena exploration',
                   'Arena wall exploration', 'Transition wall exploration',
                   'Hesitant exploration', 'Quick one loop at home']:
        if bname in behav_dict:
            bdata = behav_dict[bname][exc_mask]
            behav_labels[bdata > 0] = bname

    # Trajectory line
    ax.plot(coords_2d[:, 0], coords_2d[:, 1],
            color='gray', alpha=0.2, linewidth=0.5, zorder=1)

    for bname, color in BEHAVIOR_COLORS.items():
        mask = behav_labels == bname
        if mask.any():
            ax.scatter(coords_2d[mask, 0], coords_2d[mask, 1],
                       c=color, s=40, alpha=0.8, label=bname, zorder=5,
                       edgecolors='black', linewidths=0.3)
    other = behav_labels == 'Other'
    if other.any():
        ax.scatter(coords_2d[other, 0], coords_2d[other, 1],
                   c='#BDBDBD', s=15, alpha=0.4, label='Other', zorder=2)

    ax.scatter(*coords_2d[0], c='lime', s=200, marker='*',
               edgecolors='black', linewidths=1.5, zorder=10)
    ax.scatter(*coords_2d[-1], c='red', s=150, marker='X',
               edgecolors='black', linewidths=1.5, zorder=10)

    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('Dim 1', fontsize=9)
    ax.set_ylabel('Dim 2', fontsize=9)
    ax.legend(fontsize=6, loc='best', markerscale=1.5)
    ax.set_aspect('equal', adjustable='datalim')


def plot_zone_colored_2d(ax, coords_2d, behav_dict, exc_mask, title):
    """2D scatter colored by zone."""
    n = len(coords_2d)
    zone_labels = np.full(n, 'Other', dtype=object)
    for zone in ['Home', 'Ladder', 'Transition zone', 'Foraging arena']:
        if zone in behav_dict:
            zdata = behav_dict[zone][exc_mask]
            zone_labels[zdata > 0] = zone

    ax.plot(coords_2d[:, 0], coords_2d[:, 1],
            color='gray', alpha=0.2, linewidth=0.5, zorder=1)

    for zone, color in ZONE_COLORS.items():
        mask = zone_labels == zone
        if mask.any():
            ax.scatter(coords_2d[mask, 0], coords_2d[mask, 1],
                       c=color, s=40, alpha=0.8, label=zone, zorder=5,
                       edgecolors='black', linewidths=0.3)

    ax.scatter(*coords_2d[0], c='lime', s=200, marker='*',
               edgecolors='black', linewidths=1.5, zorder=10)
    ax.scatter(*coords_2d[-1], c='red', s=150, marker='X',
               edgecolors='black', linewidths=1.5, zorder=10)

    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('Dim 1', fontsize=9)
    ax.set_ylabel('Dim 2', fontsize=9)
    ax.legend(fontsize=7, loc='best')
    ax.set_aspect('equal', adjustable='datalim')


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("Feeding & Digging Excursion Manifold Analysis")
    print("=" * 60)

    sp = paths_config['single_probe']['coordinates_1']['mouse01']['sessions']
    sc = sp['session_1']
    sorted_path = Path(sc['sorted'])
    sorting = se.read_kilosort(sorted_path)
    lha_ids, rsp_ids = get_good_units_by_region(sorted_path)
    print(f"LHA: {len(lha_ids)} neurons, RSP: {len(rsp_ids)} neurons")

    exc_df = pd.read_csv("data/excursions_session_1.csv")
    complete = exc_df[exc_df['label'] == 'complete']

    # Resolution configs to test
    resolutions = [
        {'bin_ms': 200, 'smooth_ms': 500, 'label': '200ms/500ms'},
        {'bin_ms': 500, 'smooth_ms': 1000, 'label': '500ms/1s'},
    ]

    all_excursions = (FEEDING_EXCURSIONS + DIGGING_EXCURSIONS +
                      TYPICAL_EXCURSIONS + LOOP_EXCURSIONS)

    for region, unit_ids in [('rsp', rsp_ids), ('lha', lha_ids)]:
        region_label = region.upper()
        if len(unit_ids) < 5:
            continue

        for res in resolutions:
            bin_ms = res['bin_ms']
            smooth_ms = res['smooth_ms']
            res_label = res['label']

            print(f"\n{'='*60}")
            print(f"  {region_label} @ {res_label}")
            print(f"{'='*60}")

            zscore, time_sec = bin_and_smooth(sorting, unit_ids, bin_ms, smooth_ms)
            behav_dict = load_behavior_timeseries(1, time_sec)
            print(f"  Data shape: {zscore.shape}, {len(behav_dict)} behavior vars")

            for exc_entry in all_excursions:
                eid = exc_entry['exc_id']
                exc_label = exc_entry['label']

                erow = complete[complete['excursion_id'] == eid]
                if len(erow) == 0:
                    continue
                erow = erow.iloc[0]

                mask = (time_sec >= erow['start_time']) & (time_sec <= erow['end_time'])
                exc_data = zscore[mask]
                exc_times = time_sec[mask]
                n_pts = len(exc_data)

                if n_pts < 10:
                    print(f"  Exc {eid}: {n_pts} pts, skip")
                    continue

                print(f"\n  Exc {eid} ({exc_label}): {n_pts} pts, {erow['duration']:.1f}s")

                time_norm = np.linspace(0, 1, n_pts)

                # Embeddings
                pca3d = PCA(n_components=3).fit_transform(exc_data)
                pca2d = PCA(n_components=2).fit_transform(exc_data)
                pca3d_var = PCA(n_components=3).fit(exc_data).explained_variance_ratio_.sum()

                n_nb = min(15, n_pts - 1)
                try:
                    iso3d = Isomap(n_components=3, n_neighbors=n_nb).fit_transform(exc_data)
                    iso2d = Isomap(n_components=2, n_neighbors=n_nb).fit_transform(exc_data)
                    has_iso = True
                except:
                    iso3d, iso2d, has_iso = pca3d, pca2d, False

                # Persistent homology
                print(f"    Running persistence...", end='', flush=True)
                diagrams, data_pca, n_keep, var_exp = run_persistence(
                    exc_data, max_pts=500, n_pca=15, max_dim=2)

                # Extract gap ratios
                h1_gap, h2_gap = 0, 0
                for dim, name in [(1, 'H1'), (2, 'H2')]:
                    dmask = diagrams[:, 2] == dim
                    feats = diagrams[dmask]
                    if len(feats) > 0:
                        lt = feats[:, 1] - feats[:, 0]
                        flt = np.sort(lt[np.isfinite(lt)])[::-1]
                        if len(flt) >= 2 and flt[1] > 0:
                            gap = flt[0] / flt[1]
                            if dim == 1:
                                h1_gap = gap
                            else:
                                h2_gap = gap
                print(f" H1 gap={h1_gap:.1f}x, H2 gap={h2_gap:.1f}x")

                # Recurrence matrix
                D = squareform(pdist(data_pca))
                D_norm = D / D.max() if D.max() > 0 else D

                # =================================================
                # FIGURE: 3 rows x 3 cols
                # Row 0: 3D PCA (time) | 3D Isomap (time) | 3D behavior-colored
                # Row 1: 2D behavior | 2D zone | Recurrence matrix
                # Row 2: Persistence diagram | Barcode | Behavior timeline
                # =================================================
                fig = plt.figure(figsize=(22, 20))

                # Determine excursion type for title coloring
                is_feeding = 'Feeding' in exc_label
                is_digging = 'Digging' in exc_label
                is_loop = 'loop' in exc_label.lower()
                is_typical = 'Typical' in exc_label

                type_color = '#D32F2F' if is_feeding else '#FF9800' if is_digging else \
                             '#1976D2' if is_loop else '#666666'
                type_tag = 'FEEDING' if is_feeding else 'DIGGING' if is_digging else \
                           'LOOP' if is_loop else 'TYPICAL'

                fig.suptitle(
                    f"[{type_tag}] Excursion {eid} — {region_label} — Session 1 (Fed)\n"
                    f"{exc_label} | {erow['duration']:.1f}s, {n_pts} bins ({res_label}) | "
                    f"{len(unit_ids)} neurons | H1={h1_gap:.1f}x, H2={h2_gap:.1f}x",
                    fontsize=14, fontweight='bold', color=type_color, y=0.98)

                # Row 0
                ax0 = fig.add_subplot(3, 3, 1, projection='3d')
                plot_tube_3d(ax0, pca3d, time_norm, 'plasma',
                             f'PCA 3D ({pca3d_var:.0%} var) — Time')

                ax1 = fig.add_subplot(3, 3, 2, projection='3d')
                emb3d = iso3d if has_iso else pca3d
                plot_tube_3d(ax1, emb3d, time_norm, 'plasma',
                             f'{"Isomap" if has_iso else "PCA"} 3D — Time')

                ax2 = fig.add_subplot(3, 3, 3, projection='3d')
                plot_behavior_colored_3d(ax2, emb3d, behav_dict, mask,
                                         f'3D — Behavior coloring')

                # Row 1
                ax3 = fig.add_subplot(3, 3, 4)
                emb2d = iso2d if has_iso else pca2d
                plot_behavior_colored_2d(ax3, emb2d, behav_dict, mask,
                                          f'2D — Behavior coloring')

                ax4 = fig.add_subplot(3, 3, 5)
                plot_zone_colored_2d(ax4, emb2d, behav_dict, mask,
                                      f'2D — Zone coloring')

                ax5 = fig.add_subplot(3, 3, 6)
                t_rel = exc_times - exc_times[0]
                im = ax5.imshow(D_norm, cmap='magma_r', origin='lower', aspect='equal',
                                extent=[t_rel[0], t_rel[-1], t_rel[0], t_rel[-1]])
                thresh = np.percentile(D_norm, 20)
                ax5.contour(D_norm, levels=[thresh], colors=['cyan'], linewidths=0.8,
                            extent=[t_rel[0], t_rel[-1], t_rel[0], t_rel[-1]], origin='lower')
                ax5.set_xlabel('Time (s)', fontsize=10)
                ax5.set_ylabel('Time (s)', fontsize=10)
                ax5.set_title('Recurrence Distance Matrix', fontsize=11, fontweight='bold')
                plt.colorbar(im, ax=ax5, shrink=0.8, label='Norm. distance')

                # Row 2
                # Persistence diagram
                ax6 = fig.add_subplot(3, 3, 7)
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
                        ax6.scatter(finite[:, 0], finite[:, 1],
                                    c=[DIM_COLORS[dim]] * len(finite), s=sizes, alpha=0.6,
                                    label=f'H{dim}', edgecolors='black' if dim > 0 else 'none',
                                    linewidths=np.where(sizes > 50, 2, 0))
                        max_val = max(max_val, finite.max())
                if max_val > 0:
                    ax6.plot([0, max_val * 1.1], [0, max_val * 1.1], 'k--', alpha=0.3)
                ax6.set_xlabel('Birth', fontsize=10)
                ax6.set_ylabel('Death', fontsize=10)
                ax6.set_title(f'Persistence Diagram\nH1={h1_gap:.1f}x H2={h2_gap:.1f}x',
                              fontsize=11, fontweight='bold')
                ax6.legend(fontsize=8)

                # Barcode (H1 + H2)
                ax7 = fig.add_subplot(3, 3, 8)
                y = 0
                for dim in [1, 2]:
                    dmask = diagrams[:, 2] == dim
                    features = diagrams[dmask]
                    if len(features) == 0:
                        continue
                    lt = features[:, 1] - features[:, 0]
                    fmask = np.isfinite(lt)
                    order = np.argsort(lt[fmask])[::-1]
                    ff = features[fmask][order]
                    for fi, feat in enumerate(ff):
                        lw = 3.5 if fi == 0 else 1.2
                        alpha = 1.0 if fi == 0 else 0.5
                        ax7.plot([feat[0], feat[1]], [y, y],
                                 color=DIM_COLORS[dim], linewidth=lw, alpha=alpha)
                        y += 1
                ax7.set_xlabel('Filtration value', fontsize=10)
                ax7.set_ylabel('Feature', fontsize=10)
                ax7.set_title('Barcode (H1+H2)', fontsize=11, fontweight='bold')
                handles = [Line2D([0], [0], color=DIM_COLORS[1], lw=3, label='H1 (loops)'),
                           Line2D([0], [0], color=DIM_COLORS[2], lw=3, label='H2 (voids)')]
                ax7.legend(handles=handles, fontsize=8)

                # Behavior timeline
                ax8 = fig.add_subplot(3, 3, 9)
                y_pos = 0
                yticks, ylabels = [], []

                # Zone timeline
                for zone, color in ZONE_COLORS.items():
                    if zone in behav_dict:
                        zdata = behav_dict[zone][mask]
                        active = zdata > 0
                        if active.any():
                            starts = np.where(np.diff(np.concatenate([[0], active.astype(int)])) == 1)[0]
                            ends = np.where(np.diff(np.concatenate([active.astype(int), [0]])) == -1)[0]
                            for s, e in zip(starts, ends):
                                t_s = t_rel[s] if s < len(t_rel) else t_rel[-1]
                                t_e = t_rel[min(e, len(t_rel)-1)]
                                ax8.barh(y_pos, t_e - t_s, left=t_s, height=0.6,
                                         color=color, alpha=0.7)
                yticks.append(y_pos)
                ylabels.append('Zones')
                y_pos += 1.0

                # Behavior timeline
                for bname, color in BEHAVIOR_COLORS.items():
                    if bname in behav_dict:
                        bdata = behav_dict[bname][mask]
                        active = bdata > 0
                        if active.any():
                            starts = np.where(np.diff(np.concatenate([[0], active.astype(int)])) == 1)[0]
                            ends = np.where(np.diff(np.concatenate([active.astype(int), [0]])) == -1)[0]
                            for s, e in zip(starts, ends):
                                t_s = t_rel[s] if s < len(t_rel) else t_rel[-1]
                                t_e = t_rel[min(e, len(t_rel)-1)]
                                ax8.barh(y_pos, t_e - t_s, left=t_s, height=0.6,
                                         color=color, alpha=0.8)
                            yticks.append(y_pos)
                            ylabels.append(bname)
                            y_pos += 0.8

                ax8.set_yticks(yticks)
                ax8.set_yticklabels(ylabels, fontsize=7)
                ax8.set_xlabel('Time within excursion (s)', fontsize=10)
                ax8.set_title('Behavior Timeline', fontsize=11, fontweight='bold')

                # Zone legend
                zone_handles = [Patch(facecolor=c, alpha=0.7, label=n)
                                for n, c in ZONE_COLORS.items()]
                ax8.legend(handles=zone_handles, fontsize=6, loc='upper right', ncol=2)

                plt.tight_layout(rect=[0, 0, 1, 0.94])
                tag = 'feeding' if is_feeding else 'digging' if is_digging else \
                      'loop' if is_loop else 'typical'
                outpath = Path("figures") / f"manifold_{tag}_exc{eid}_{region}_{bin_ms}ms.png"
                fig.savefig(outpath, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"    Saved: {outpath}")

    # =========================================================================
    # SUMMARY: Side-by-side comparison figure
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"  Creating comparison figure...")
    print(f"{'='*60}")

    # Compare at 500ms/1s for RSP
    bin_ms, smooth_ms = 500, 1000
    zscore, time_sec = bin_and_smooth(sorting, rsp_ids, bin_ms, smooth_ms)
    behav_dict = load_behavior_timeseries(1, time_sec)

    compare_ids = [81, 57, 90, 89, 80]
    compare_labels = ['FEEDING (Exc 81)', 'DIGGING (Exc 57)', 'LOOP (Exc 90)',
                      'LOOP (Exc 89)', 'TYPICAL (Exc 80)']
    compare_colors = ['#D32F2F', '#FF9800', '#1976D2', '#1976D2', '#666666']

    fig, axes = plt.subplots(len(compare_ids), 4, figsize=(24, 6 * len(compare_ids)))
    fig.suptitle("RSP Manifold Comparison — Feeding vs Digging vs Loop vs Typical\n"
                 "Session 1 (Fed) | 500ms/1s | 108 neurons",
                 fontsize=15, fontweight='bold', y=0.99)

    for ei, (eid, label, color) in enumerate(zip(compare_ids, compare_labels, compare_colors)):
        erow = complete[complete['excursion_id'] == eid]
        if len(erow) == 0:
            continue
        erow = erow.iloc[0]

        mask = (time_sec >= erow['start_time']) & (time_sec <= erow['end_time'])
        exc_data = zscore[mask]
        n_pts = len(exc_data)
        if n_pts < 10:
            continue

        time_norm = np.linspace(0, 1, n_pts)
        n_nb = min(15, n_pts - 1)

        iso2d = Isomap(n_components=2, n_neighbors=n_nb).fit_transform(exc_data)
        iso3d = Isomap(n_components=3, n_neighbors=n_nb).fit_transform(exc_data)

        diagrams, data_pca, n_keep, var_exp = run_persistence(
            exc_data, max_pts=500, n_pca=15, max_dim=2)
        h1_gap, h2_gap = 0, 0
        for dim in [1, 2]:
            dmask = diagrams[:, 2] == dim
            feats = diagrams[dmask]
            if len(feats) > 0:
                lt = feats[:, 1] - feats[:, 0]
                flt = np.sort(lt[np.isfinite(lt)])[::-1]
                if len(flt) >= 2 and flt[1] > 0:
                    if dim == 1:
                        h1_gap = flt[0] / flt[1]
                    else:
                        h2_gap = flt[0] / flt[1]

        # Col 0: 3D Isomap tube (time)
        ax3d = fig.add_subplot(len(compare_ids), 4, ei * 4 + 1, projection='3d')
        plot_tube_3d(ax3d, iso3d, time_norm, 'plasma',
                     f'{label}\n{erow["duration"]:.0f}s, H1={h1_gap:.1f}x H2={h2_gap:.1f}x')
        axes[ei, 0].set_visible(False)

        # Col 1: 2D behavior
        plot_behavior_colored_2d(axes[ei, 1], iso2d, behav_dict, mask,
                                  f'{label} — Behavior')

        # Col 2: 2D zone
        plot_zone_colored_2d(axes[ei, 2], iso2d, behav_dict, mask,
                              f'{label} — Zone')

        # Col 3: Persistence diagram
        max_val = 0
        for dim in range(3):
            dmask = diagrams[:, 2] == dim
            features = diagrams[dmask]
            if len(features) == 0:
                continue
            finite = features[np.isfinite(features[:, 1])]
            if len(finite) > 0:
                lt = finite[:, 1] - finite[:, 0]
                sizes = np.full(len(finite), 20)
                sizes[np.argmax(lt)] = 100
                axes[ei, 3].scatter(finite[:, 0], finite[:, 1],
                                     c=[DIM_COLORS[dim]] * len(finite), s=sizes, alpha=0.6,
                                     label=f'H{dim}',
                                     edgecolors='black' if dim > 0 else 'none',
                                     linewidths=np.where(sizes > 50, 1.5, 0))
                max_val = max(max_val, finite.max())
        if max_val > 0:
            axes[ei, 3].plot([0, max_val * 1.1], [0, max_val * 1.1], 'k--', alpha=0.3)
        axes[ei, 3].set_xlabel('Birth', fontsize=9)
        axes[ei, 3].set_ylabel('Death', fontsize=9)
        axes[ei, 3].set_title(f'{label}\nH1={h1_gap:.1f}x H2={h2_gap:.1f}x',
                               fontsize=10, fontweight='bold', color=color)
        axes[ei, 3].legend(fontsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    outpath = Path("figures") / "manifold_feeding_vs_loop_comparison.png"
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {outpath}")

    print("\nDone!")


if __name__ == "__main__":
    main()
