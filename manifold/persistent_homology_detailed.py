"""
Detailed Visualizations of Excursions with Prominent Topology
=============================================================
For each excursion with significant H1 loops or H2 voids:
  - 3D PCA scatter (colored by time)
  - 3D UMAP scatter (colored by time, min_dist=0)
  - 3D Isomap scatter (colored by time)
  - Persistence diagram + barcode
  - 2D UMAP with representative cycle highlighted
  - Zone-colored 3D embedding
"""

import yaml
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from gtda.homology import VietorisRipsPersistence
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from scipy.ndimage import uniform_filter1d
from scipy.spatial.distance import pdist, squareform
import spikeinterface.extractors as se
import warnings

warnings.filterwarnings('ignore')

# Try UMAP
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("WARNING: umap not available, skipping UMAP plots")

# =============================================================================
# CONFIG
# =============================================================================

FS = 30000
LHA_DEPTH_MAX = 1300
RSP_DEPTH_MIN = 1300

with open("paths.yaml") as f:
    paths_config = yaml.safe_load(f)

DIM_COLORS = {0: '#1976D2', 1: '#D32F2F', 2: '#4CAF50'}
DIM_LABELS = {0: 'H0 (components)', 1: 'H1 (loops)', 2: 'H2 (voids)'}

# Excursions with prominent topology from the scan
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
     'feature': 'H2 void', 'gap': 16.9, 'note': 'Strongest void in dataset'},
    {'exc_id': 58, 'region': 'rsp', 'bin_ms': 500, 'smooth_ms': 1000,
     'feature': 'H2 void', 'gap': 6.5, 'note': 'Strong void'},
    # LHA H1 loops
    {'exc_id': 11, 'region': 'lha', 'bin_ms': 500, 'smooth_ms': 1000,
     'feature': 'H1 loop', 'gap': 4.3, 'note': 'Strongest LHA loop'},
    {'exc_id': 89, 'region': 'lha', 'bin_ms': 500, 'smooth_ms': 1000,
     'feature': 'H1 loop', 'gap': 3.4, 'note': 'Loop in both LHA+RSP'},
]

ZONE_COLORS = {
    'Home': '#4CAF50',
    'Ladder': '#FF9800',
    'Transition Zone': '#9C27B0',
    'Foraging Arena': '#D32F2F',
}

# =============================================================================
# DATA LOADING (reused from persistent_homology script)
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
    """Load zone labels from behavior CSV aligned to neural time bins."""
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
                # Assume 100ms bins starting from t=0
                behav_time = np.arange(len(row_data)) * 0.1
                for ti, t in enumerate(time_sec):
                    bi = int(t / 0.1)
                    if 0 <= bi < len(row_data) and row_data[bi] == 1:
                        zones[ti] = zone_name
                break

    return zones


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
        max_edge_length=np.inf,
        n_jobs=-1,
    )
    diagrams = VR.fit_transform(data_pca[np.newaxis, :, :])[0]
    return diagrams, data_pca, pca, n_keep, var_exp[n_keep - 1]


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def make_3d_scatter(ax, coords_3d, color_vals, cmap, title, norm=None):
    """3D scatter plot with consistent styling."""
    if norm is None:
        norm = Normalize(vmin=np.min(color_vals), vmax=np.max(color_vals))
    sc = ax.scatter(coords_3d[:, 0], coords_3d[:, 1], coords_3d[:, 2],
                    c=color_vals, cmap=cmap, norm=norm, s=12, alpha=0.7)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('Dim 1', fontsize=8)
    ax.set_ylabel('Dim 2', fontsize=8)
    ax.set_zlabel('Dim 3', fontsize=8)
    ax.tick_params(labelsize=7)
    # Connect consecutive points with thin line
    ax.plot(coords_3d[:, 0], coords_3d[:, 1], coords_3d[:, 2],
            color='gray', alpha=0.15, linewidth=0.3)
    return sc


def make_zone_3d_scatter(ax, coords_3d, zones, title):
    """3D scatter colored by behavioral zone."""
    for zone, color in ZONE_COLORS.items():
        mask = zones == zone
        if mask.any():
            ax.scatter(coords_3d[mask, 0], coords_3d[mask, 1], coords_3d[mask, 2],
                       c=color, s=12, alpha=0.7, label=zone)
    unknown = ~np.isin(zones, list(ZONE_COLORS.keys()))
    if unknown.any():
        ax.scatter(coords_3d[unknown, 0], coords_3d[unknown, 1], coords_3d[unknown, 2],
                   c='#BDBDBD', s=8, alpha=0.3, label='Unknown')
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('Dim 1', fontsize=8)
    ax.set_ylabel('Dim 2', fontsize=8)
    ax.set_zlabel('Dim 3', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, loc='upper left', markerscale=1.5)


def plot_persistence_panel(diagrams, ax_diag, ax_bar, feature_type, gap_ratio):
    """Combined persistence diagram + barcode."""
    # Diagram
    max_val = 0
    for dim in range(3):
        mask = diagrams[:, 2] == dim
        features = diagrams[mask]
        if len(features) == 0:
            continue
        finite = features[np.isfinite(features[:, 1])]
        if len(finite) > 0:
            sizes = np.full(len(finite), 25)
            # Highlight the top feature
            lifetimes = finite[:, 1] - finite[:, 0]
            top_idx = np.argmax(lifetimes)
            sizes[top_idx] = 120
            ax_diag.scatter(finite[:, 0], finite[:, 1],
                           c=DIM_COLORS[dim], s=sizes, alpha=0.6,
                           label=DIM_LABELS[dim], zorder=2,
                           edgecolors='black' if dim > 0 else 'none',
                           linewidths=np.where(sizes > 50, 1.5, 0))
            max_val = max(max_val, finite.max())
        inf_feat = features[~np.isfinite(features[:, 1])]
        if len(inf_feat) > 0 and max_val > 0:
            ax_diag.scatter(inf_feat[:, 0], [max_val * 1.05] * len(inf_feat),
                           c=DIM_COLORS[dim], s=35, marker='^', alpha=0.8, zorder=2)

    if max_val > 0:
        ax_diag.plot([0, max_val * 1.1], [0, max_val * 1.1], 'k--', alpha=0.3, linewidth=0.8)
    ax_diag.set_xlabel('Birth', fontsize=9)
    ax_diag.set_ylabel('Death', fontsize=9)
    ax_diag.set_title(f'Persistence Diagram\n{feature_type} gap = {gap_ratio:.1f}x',
                      fontsize=10, fontweight='bold',
                      color='#D32F2F' if 'H1' in feature_type else '#4CAF50')
    ax_diag.legend(fontsize=7, loc='lower right')

    # Barcode — show only H1 and H2 for clarity
    y = 0
    for dim in [1, 2]:
        mask = diagrams[:, 2] == dim
        features = diagrams[mask]
        if len(features) == 0:
            continue
        lifetimes = features[:, 1] - features[:, 0]
        finite_mask = np.isfinite(lifetimes)
        order = np.argsort(lifetimes[finite_mask])[::-1]
        finite_features = features[finite_mask][order]
        finite_lt = lifetimes[finite_mask][order]

        for fi, feat in enumerate(finite_features):
            lw = 3.5 if fi == 0 else 1.2
            alpha = 1.0 if fi == 0 else 0.5
            ax_bar.plot([feat[0], feat[1]], [y, y],
                       color=DIM_COLORS[dim], linewidth=lw, alpha=alpha)
            y += 1

    ax_bar.set_xlabel('Filtration value', fontsize=9)
    ax_bar.set_ylabel('Feature', fontsize=9)
    ax_bar.set_title('Barcode (H1 + H2 only)\nTop feature highlighted',
                     fontsize=10)
    # Add legend
    handles = [Line2D([0], [0], color=DIM_COLORS[1], lw=3, label='H1 (loops)'),
               Line2D([0], [0], color=DIM_COLORS[2], lw=3, label='H2 (voids)')]
    ax_bar.legend(handles=handles, fontsize=7)


def make_2d_with_trajectory(ax, coords_2d, time_norm, title, connect=True):
    """2D scatter with trajectory line and time coloring."""
    sc = ax.scatter(coords_2d[:, 0], coords_2d[:, 1],
                    c=time_norm, cmap='plasma', s=15, alpha=0.8, zorder=2)
    if connect:
        ax.plot(coords_2d[:, 0], coords_2d[:, 1],
                color='gray', alpha=0.2, linewidth=0.5, zorder=1)
    # Mark start and end
    ax.scatter(coords_2d[0, 0], coords_2d[0, 1],
               c='lime', s=100, marker='*', edgecolors='black',
               linewidths=1, zorder=5, label='Start')
    ax.scatter(coords_2d[-1, 0], coords_2d[-1, 1],
               c='red', s=100, marker='X', edgecolors='black',
               linewidths=1, zorder=5, label='End')
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('Dim 1', fontsize=9)
    ax.set_ylabel('Dim 2', fontsize=9)
    ax.legend(fontsize=7)
    return sc


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("Detailed Topology Visualizations")
    print("=" * 50)

    sp = paths_config['single_probe']['coordinates_1']['mouse01']['sessions']
    sc = sp['session_1']
    sorted_path = Path(sc['sorted'])

    sorting = se.read_kilosort(sorted_path)
    lha_ids, rsp_ids = get_good_units_by_region(sorted_path)
    print(f"LHA: {len(lha_ids)} neurons, RSP: {len(rsp_ids)} neurons")

    exc_df = pd.read_csv("data/excursions_session_1.csv")
    complete = exc_df[exc_df['label'] == 'complete']

    # Cache binned data per region+resolution
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
        print(f"  Exc {eid} | {region_label} | {feature_type} (gap={gap_val}x)")
        print(f"  {note}")
        print(f"  Resolution: {bin_ms}ms bins / {smooth_ms}ms smooth")
        print(f"{'='*60}")

        # Get excursion info
        erow = complete[complete['excursion_id'] == eid]
        if len(erow) == 0:
            print(f"  WARNING: Excursion {eid} not found in complete list, skipping")
            continue
        erow = erow.iloc[0]

        # Load binned data (cached)
        cache_key = (region, bin_ms, smooth_ms)
        if cache_key not in binned_cache:
            print(f"  Binning {region_label} at {bin_ms}ms...", end='', flush=True)
            zscore, time_sec = bin_and_smooth(sorting, unit_ids, bin_ms, smooth_ms)
            binned_cache[cache_key] = (zscore, time_sec)
            print(f" done ({zscore.shape})")
        else:
            zscore, time_sec = binned_cache[cache_key]

        # Extract excursion data
        mask = (time_sec >= erow['start_time']) & (time_sec <= erow['end_time'])
        exc_data = zscore[mask]
        exc_times = time_sec[mask]

        if len(exc_data) < 10:
            print(f"  Only {len(exc_data)} time points, skipping")
            continue

        print(f"  {len(exc_data)} time points, {erow['duration']:.1f}s duration")

        # Load zone labels
        zones = load_behavior_zones(1, exc_times)

        # Normalized time for coloring (0=start, 1=end)
        time_norm = (exc_times - exc_times.min()) / max(exc_times.max() - exc_times.min(), 1e-8)

        # =====================================================
        # Run persistent homology
        # =====================================================
        print("  Running persistent homology...", end='', flush=True)
        diagrams, data_pca, pca_model, n_keep, var_exp = run_persistence(
            exc_data, max_pts=500, n_pca=15, max_dim=2)
        print(f" done (PCA: {n_keep}D, {var_exp:.0%} var)")

        # =====================================================
        # Compute 3D embeddings
        # =====================================================

        # PCA 3D (from the full excursion, not subsampled)
        pca_3d = PCA(n_components=3)
        coords_pca3d = pca_3d.fit_transform(exc_data)
        pca3d_var = pca_3d.explained_variance_ratio_.sum()

        # Isomap 3D
        n_neighbors_iso = min(15, len(exc_data) - 1)
        try:
            iso = Isomap(n_components=3, n_neighbors=n_neighbors_iso)
            coords_iso3d = iso.fit_transform(exc_data)
            has_isomap = True
        except Exception as e:
            print(f"  Isomap failed: {e}")
            has_isomap = False

        # UMAP 3D
        has_umap = False
        if HAS_UMAP and len(exc_data) > 15:
            try:
                reducer = umap.UMAP(n_components=3, n_neighbors=min(15, len(exc_data)-1),
                                     min_dist=0.0, spread=0.5, random_state=42)
                coords_umap3d = reducer.fit_transform(exc_data)
                has_umap = True
            except Exception as e:
                print(f"  UMAP 3D failed: {e}")

        # UMAP 2D
        has_umap2d = False
        if HAS_UMAP and len(exc_data) > 15:
            try:
                reducer2d = umap.UMAP(n_components=2, n_neighbors=min(15, len(exc_data)-1),
                                       min_dist=0.0, spread=0.5, random_state=42)
                coords_umap2d = reducer2d.fit_transform(exc_data)
                has_umap2d = True
            except Exception as e:
                print(f"  UMAP 2D failed: {e}")

        # Isomap 2D
        has_iso2d = False
        try:
            iso2d = Isomap(n_components=2, n_neighbors=n_neighbors_iso)
            coords_iso2d = iso2d.fit_transform(exc_data)
            has_iso2d = True
        except Exception:
            pass

        # =====================================================
        # Create figure: 3 rows x 3 cols
        # Row 0: 3D PCA (time) | 3D UMAP (time) | 3D Isomap (time)
        # Row 1: 3D PCA (zone) | 3D UMAP (zone) | 3D Isomap (zone)
        # Row 2: 2D UMAP (time+traj) | Persistence diagram | Barcode
        # =====================================================
        fig = plt.figure(figsize=(20, 18))
        fig.suptitle(
            f"Excursion {eid} — {region_label} — Session 1 (Fed)\n"
            f"{feature_type} (gap ratio = {gap_val:.1f}x) | "
            f"Duration: {erow['duration']:.1f}s | "
            f"{len(exc_data)} time bins ({bin_ms}ms/{smooth_ms}ms) | "
            f"{len(unit_ids)} neurons\n"
            f"{note}",
            fontsize=14, fontweight='bold', y=0.98)

        # Row 0: 3D embeddings colored by TIME
        ax0 = fig.add_subplot(3, 3, 1, projection='3d')
        sc0 = make_3d_scatter(ax0, coords_pca3d, time_norm, 'plasma',
                              f'PCA 3D ({pca3d_var:.0%} var)\nTime coloring')

        if has_umap:
            ax1 = fig.add_subplot(3, 3, 2, projection='3d')
            sc1 = make_3d_scatter(ax1, coords_umap3d, time_norm, 'plasma',
                                  'UMAP 3D (min_dist=0)\nTime coloring')
        else:
            ax1 = fig.add_subplot(3, 3, 2)
            ax1.text(0.5, 0.5, 'UMAP not available', ha='center', va='center', fontsize=12)
            ax1.set_title('UMAP 3D')

        if has_isomap:
            ax2 = fig.add_subplot(3, 3, 3, projection='3d')
            sc2 = make_3d_scatter(ax2, coords_iso3d, time_norm, 'plasma',
                                  'Isomap 3D\nTime coloring')
        else:
            ax2 = fig.add_subplot(3, 3, 3)
            ax2.text(0.5, 0.5, 'Isomap failed', ha='center', va='center', fontsize=12)
            ax2.set_title('Isomap 3D')

        # Add colorbar for time
        sm = ScalarMappable(cmap='plasma', norm=Normalize(0, 1))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=[ax0, ax1, ax2] if has_umap and has_isomap else [ax0],
                           shrink=0.4, pad=0.08, label='Normalized time (0→1)')

        # Row 1: 3D embeddings colored by ZONE
        ax3 = fig.add_subplot(3, 3, 4, projection='3d')
        make_zone_3d_scatter(ax3, coords_pca3d, zones, 'PCA 3D — Zone coloring')

        if has_umap:
            ax4 = fig.add_subplot(3, 3, 5, projection='3d')
            make_zone_3d_scatter(ax4, coords_umap3d, zones, 'UMAP 3D — Zone coloring')
        else:
            ax4 = fig.add_subplot(3, 3, 5)
            ax4.text(0.5, 0.5, 'UMAP not available', ha='center', va='center', fontsize=12)

        if has_isomap:
            ax5 = fig.add_subplot(3, 3, 6, projection='3d')
            make_zone_3d_scatter(ax5, coords_iso3d, zones, 'Isomap 3D — Zone coloring')
        else:
            ax5 = fig.add_subplot(3, 3, 6)
            ax5.text(0.5, 0.5, 'Isomap failed', ha='center', va='center', fontsize=12)

        # Row 2: 2D trajectory | Persistence diagram | Barcode
        ax6 = fig.add_subplot(3, 3, 7)
        if has_umap2d:
            sc6 = make_2d_with_trajectory(ax6, coords_umap2d, time_norm,
                                           'UMAP 2D — Trajectory')
        elif has_iso2d:
            sc6 = make_2d_with_trajectory(ax6, coords_iso2d, time_norm,
                                           'Isomap 2D — Trajectory')
        else:
            pca_2d_obj = PCA(n_components=2)
            coords_pca2d = pca_2d_obj.fit_transform(exc_data)
            sc6 = make_2d_with_trajectory(ax6, coords_pca2d, time_norm,
                                           'PCA 2D — Trajectory')

        ax7 = fig.add_subplot(3, 3, 8)
        ax8 = fig.add_subplot(3, 3, 9)
        plot_persistence_panel(diagrams, ax7, ax8, feature_type, gap_val)

        plt.tight_layout(rect=[0, 0, 1, 0.93])
        outname = f"persistent_homology_detail_exc{eid}_{region}_s1.png"
        outpath = Path("figures") / outname
        fig.savefig(outpath, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {outpath}")

    # =====================================================
    # Summary figure: all prominent excursions side by side
    # =====================================================
    print("\n\nCreating summary comparison figure...")

    # Group by region
    for region in ['rsp', 'lha']:
        entries = [e for e in PROMINENT_EXCURSIONS if e['region'] == region]
        if not entries:
            continue

        region_label = region.upper()
        n_exc = len(entries)
        fig, axes = plt.subplots(n_exc, 4, figsize=(22, 5 * n_exc))
        if n_exc == 1:
            axes = axes[np.newaxis, :]

        fig.suptitle(
            f"{region_label} — Excursions with Prominent Topology (Session 1, Fed)\n"
            f"3D PCA (time) | 3D UMAP (time) | 2D UMAP (trajectory) | Persistence Diagram",
            fontsize=14, fontweight='bold', y=0.99)

        for ei, entry in enumerate(entries):
            eid = entry['exc_id']
            bin_ms = entry['bin_ms']
            smooth_ms = entry['smooth_ms']
            feature_type = entry['feature']
            gap_val = entry['gap']

            unit_ids = lha_ids if region == 'lha' else rsp_ids
            cache_key = (region, bin_ms, smooth_ms)
            zscore, time_sec = binned_cache[cache_key]

            erow = complete[complete['excursion_id'] == eid]
            if len(erow) == 0:
                for c in range(4):
                    axes[ei, c].set_visible(False)
                continue
            erow = erow.iloc[0]

            mask = (time_sec >= erow['start_time']) & (time_sec <= erow['end_time'])
            exc_data = zscore[mask]
            exc_times = time_sec[mask]
            if len(exc_data) < 10:
                for c in range(4):
                    axes[ei, c].set_visible(False)
                continue

            time_norm = (exc_times - exc_times.min()) / max(exc_times.max() - exc_times.min(), 1e-8)

            # PCA 3D
            pca_3d = PCA(n_components=3)
            coords_pca3d = pca_3d.fit_transform(exc_data)

            ax_pca = fig.add_subplot(n_exc, 4, ei * 4 + 1, projection='3d')
            make_3d_scatter(ax_pca, coords_pca3d, time_norm, 'plasma',
                           f'Exc {eid} — PCA 3D\n{feature_type} gap={gap_val:.1f}x')
            # Remove the flat placeholder
            axes[ei, 0].set_visible(False)

            # UMAP 3D
            if HAS_UMAP and len(exc_data) > 15:
                try:
                    reducer3d = umap.UMAP(n_components=3, n_neighbors=min(15, len(exc_data)-1),
                                           min_dist=0.0, spread=0.5, random_state=42)
                    coords_umap3d = reducer3d.fit_transform(exc_data)
                    ax_umap = fig.add_subplot(n_exc, 4, ei * 4 + 2, projection='3d')
                    make_3d_scatter(ax_umap, coords_umap3d, time_norm, 'plasma',
                                   f'Exc {eid} — UMAP 3D')
                    axes[ei, 1].set_visible(False)
                except:
                    axes[ei, 1].text(0.5, 0.5, 'UMAP failed', ha='center', va='center')

            # UMAP 2D trajectory
            if HAS_UMAP and len(exc_data) > 15:
                try:
                    reducer2d = umap.UMAP(n_components=2, n_neighbors=min(15, len(exc_data)-1),
                                           min_dist=0.0, spread=0.5, random_state=42)
                    coords_umap2d = reducer2d.fit_transform(exc_data)
                    make_2d_with_trajectory(axes[ei, 2], coords_umap2d, time_norm,
                                            f'Exc {eid} — UMAP 2D trajectory')
                except:
                    axes[ei, 2].text(0.5, 0.5, 'UMAP failed', ha='center', va='center')

            # Persistence diagram
            diagrams, _, _, n_keep, var_exp = run_persistence(
                exc_data, max_pts=500, n_pca=15, max_dim=2)
            max_val = 0
            for dim in range(3):
                dmask = diagrams[:, 2] == dim
                features = diagrams[dmask]
                if len(features) == 0:
                    continue
                finite = features[np.isfinite(features[:, 1])]
                if len(finite) > 0:
                    sizes = np.full(len(finite), 20)
                    lt = finite[:, 1] - finite[:, 0]
                    sizes[np.argmax(lt)] = 100
                    axes[ei, 3].scatter(finite[:, 0], finite[:, 1],
                                        c=DIM_COLORS[dim], s=sizes, alpha=0.6,
                                        label=DIM_LABELS[dim],
                                        edgecolors='black' if dim > 0 else 'none',
                                        linewidths=np.where(sizes > 50, 1.5, 0))
                    max_val = max(max_val, finite.max())
            if max_val > 0:
                axes[ei, 3].plot([0, max_val * 1.1], [0, max_val * 1.1],
                                'k--', alpha=0.3, linewidth=0.8)
            axes[ei, 3].set_xlabel('Birth', fontsize=9)
            axes[ei, 3].set_ylabel('Death', fontsize=9)
            axes[ei, 3].set_title(f'Exc {eid} — {feature_type} gap={gap_val:.1f}x\n'
                                   f'{erow["duration"]:.0f}s, {len(exc_data)} pts',
                                  fontsize=10, fontweight='bold',
                                  color='#D32F2F' if 'H1' in feature_type else '#4CAF50')
            axes[ei, 3].legend(fontsize=7)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        outpath = Path("figures") / f"persistent_homology_summary_{region}_s1.png"
        fig.savefig(outpath, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {outpath}")

    print("\nDone!")


if __name__ == "__main__":
    main()
