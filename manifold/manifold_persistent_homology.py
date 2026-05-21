"""
Persistent Homology on Neural Excursion Data
=============================================
Uses giotto-tda to compute Vietoris-Rips persistence on neural
population activity during individual excursions.

Detects:
  - H0: connected components (should merge to 1)
  - H1: loops/rings (prominent = circular manifold)
  - H2: voids/cavities (prominent = toroidal or spherical structure)

For each excursion: persistence diagram, barcode, and gap ratio.
Runs on all complete excursions at multiple time resolutions.
"""

import yaml
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from gtda.homology import VietorisRipsPersistence
from sklearn.decomposition import PCA
from scipy.ndimage import uniform_filter1d
import spikeinterface.extractors as se
import warnings
import time as timer

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIG
# =============================================================================

FS = 30000
LHA_DEPTH_MAX = 1300
RSP_DEPTH_MIN = 1300

with open("paths.yaml") as f:
    paths_config = yaml.safe_load(f)

SESSION_INFO = {
    1: {'state': 'Fed', 'phase': 'Exploration'},
}

DIM_COLORS = {0: '#1976D2', 1: '#D32F2F', 2: '#4CAF50'}
DIM_LABELS = {0: 'H0 (components)', 1: 'H1 (loops)', 2: 'H2 (voids)'}


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


def bin_and_smooth_flexible(sorting, unit_ids, bin_ms=10, smooth_ms=100):
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


def extract_excursion_data(zscore, time_sec, exc_start, exc_end):
    mask = (time_sec >= exc_start) & (time_sec <= exc_end)
    return zscore[mask], time_sec[mask]


# =============================================================================
# PERSISTENT HOMOLOGY
# =============================================================================

def run_persistence(data, max_pts=500, n_pca=15, max_dim=2):
    """Run Vietoris-Rips persistent homology via giotto-tda.

    PCA-reduce first to remove noise dimensions, then subsample
    if needed for computational tractability.
    """
    # PCA
    n_comp = min(n_pca, data.shape[1], len(data) - 1)
    pca = PCA(n_components=n_comp)
    data_pca = pca.fit_transform(data)
    var_exp = pca.explained_variance_ratio_.cumsum()
    n_keep = min(np.searchsorted(var_exp, 0.95) + 1, n_comp)
    n_keep = max(n_keep, 3)
    data_pca = data_pca[:, :n_keep]

    # Subsample if too many points
    if len(data_pca) > max_pts:
        idx = np.random.choice(len(data_pca), max_pts, replace=False)
        idx.sort()
        data_pca = data_pca[idx]

    # Run VR persistence
    VR = VietorisRipsPersistence(
        homology_dimensions=list(range(max_dim + 1)),
        max_edge_length=np.inf,
        n_jobs=-1,
    )
    diagrams = VR.fit_transform(data_pca[np.newaxis, :, :])[0]

    return diagrams, data_pca, n_keep, var_exp[n_keep - 1]


def analyze_persistence(diagrams):
    """Extract key topological features from persistence diagram."""
    results = {}
    for dim in range(3):
        mask = diagrams[:, 2] == dim
        features = diagrams[mask]
        if len(features) == 0:
            results[f'h{dim}_count'] = 0
            results[f'h{dim}_top_lifetime'] = 0
            results[f'h{dim}_gap_ratio'] = 0
            continue

        lifetimes = features[:, 1] - features[:, 0]
        # Remove infinite lifetimes for gap analysis
        finite_lt = lifetimes[np.isfinite(lifetimes)]
        if len(finite_lt) == 0:
            results[f'h{dim}_count'] = len(lifetimes)
            results[f'h{dim}_top_lifetime'] = 0
            results[f'h{dim}_gap_ratio'] = 0
            continue

        sorted_lt = np.sort(finite_lt)[::-1]
        results[f'h{dim}_count'] = len(finite_lt)
        results[f'h{dim}_top_lifetime'] = sorted_lt[0]
        results[f'h{dim}_gap_ratio'] = (
            sorted_lt[0] / sorted_lt[1] if len(sorted_lt) > 1 and sorted_lt[1] > 0
            else 0
        )

    return results


def plot_persistence_diagram(diagrams, ax, title=''):
    """Plot persistence diagram."""
    max_val = 0
    for dim in range(3):
        mask = diagrams[:, 2] == dim
        features = diagrams[mask]
        if len(features) == 0:
            continue
        finite = features[np.isfinite(features[:, 1])]
        if len(finite) > 0:
            ax.scatter(finite[:, 0], finite[:, 1],
                       c=DIM_COLORS[dim], s=25, alpha=0.6,
                       label=DIM_LABELS[dim], zorder=2)
            max_val = max(max_val, finite.max())
        # Infinite features
        inf_feat = features[~np.isfinite(features[:, 1])]
        if len(inf_feat) > 0 and max_val > 0:
            ax.scatter(inf_feat[:, 0], [max_val * 1.05] * len(inf_feat),
                       c=DIM_COLORS[dim], s=35, marker='^', alpha=0.8,
                       zorder=2)

    if max_val > 0:
        ax.plot([0, max_val * 1.1], [0, max_val * 1.1], 'k--',
                alpha=0.3, linewidth=0.8, zorder=1)
    ax.set_xlabel('Birth', fontsize=9)
    ax.set_ylabel('Death', fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7, loc='lower right')


def plot_persistence_barcode(diagrams, ax, title=''):
    """Plot persistence barcode."""
    y = 0
    for dim in [0, 1, 2]:
        mask = diagrams[:, 2] == dim
        features = diagrams[mask]
        if len(features) == 0:
            continue
        lifetimes = features[:, 1] - features[:, 0]
        finite_mask = np.isfinite(lifetimes)
        # Sort by lifetime (longest first)
        order = np.argsort(lifetimes[finite_mask])[::-1]
        finite_features = features[finite_mask][order]

        for feat in finite_features:
            ax.plot([feat[0], feat[1]], [y, y],
                    color=DIM_COLORS[dim], linewidth=1.5, alpha=0.7)
            y += 1

        # Infinite bars
        inf_features = features[~finite_mask]
        max_death = finite_features[:, 1].max() if len(finite_features) > 0 else 1
        for feat in inf_features:
            ax.plot([feat[0], max_death * 1.15], [y, y],
                    color=DIM_COLORS[dim], linewidth=2, alpha=0.9)
            ax.plot(max_death * 1.15, y, '>', color=DIM_COLORS[dim], ms=3)
            y += 1

    ax.set_xlabel('Filtration value', fontsize=9)
    ax.set_ylabel('Feature', fontsize=9)
    ax.set_title(title, fontsize=10)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("Persistent Homology on Neural Excursion Data")
    print("=" * 50)

    sp = paths_config['single_probe']['coordinates_1']['mouse01']['sessions']
    sc = sp['session_1']
    sorted_path = Path(sc['sorted'])

    sorting = se.read_kilosort(sorted_path)
    lha_ids, rsp_ids = get_good_units_by_region(sorted_path)

    exc_df = pd.read_csv("data/excursions_session_1.csv")
    complete = exc_df[exc_df['label'] == 'complete'].sort_values('duration', ascending=False)
    print(f"  {len(complete)} complete excursions in S1")

    resolutions = [
        {'bin_ms': 50,  'smooth_ms': 200,  'label': '50ms/200ms'},
        {'bin_ms': 200, 'smooth_ms': 500,  'label': '200ms/500ms'},
        {'bin_ms': 500, 'smooth_ms': 1000, 'label': '500ms/1s'},
    ]

    # Top 6 longest for detailed figures
    top_exc = complete.head(6)

    all_results = []

    for region, region_label, unit_ids in [('lha', 'LHA', lha_ids),
                                            ('rsp', 'RSP', rsp_ids)]:
        if len(unit_ids) < 5:
            continue

        print(f"\n{'='*60}")
        print(f"  {region_label}: {len(unit_ids)} neurons")
        print(f"{'='*60}")

        for res in resolutions:
            print(f"\n  Resolution: {res['label']}")

            zscore, time_sec = bin_and_smooth_flexible(
                sorting, unit_ids,
                bin_ms=res['bin_ms'], smooth_ms=res['smooth_ms'])

            # ============================================
            # Figure: Top 6 excursions, 3 cols each
            # Col 0: Persistence diagram
            # Col 1: Persistence barcode
            # Col 2: H1 lifetime gap analysis
            # ============================================
            n_show = len(top_exc)
            fig, axes = plt.subplots(n_show, 3, figsize=(18, 4 * n_show))
            fig.suptitle(f"{region_label} Session 1 — Persistent Homology\n"
                         f"Resolution: {res['label']}, {len(unit_ids)} neurons, "
                         f"PCA→95% var, max 500 pts",
                         fontsize=14, fontweight='bold')

            for ei, (_, erow) in enumerate(top_exc.iterrows()):
                eid = int(erow['excursion_id'])
                exc_data, exc_times = extract_excursion_data(
                    zscore, time_sec, erow['start_time'], erow['end_time'])

                if len(exc_data) < 10:
                    print(f"    Exc {eid}: {len(exc_data)} pts, skip")
                    for c in range(3):
                        axes[ei, c].set_visible(False)
                    continue

                print(f"    Exc {eid}: {len(exc_data)} pts, {erow['duration']:.0f}s...",
                      end='', flush=True)
                t0 = timer.time()

                diagrams, data_pca, n_pca, var_exp = run_persistence(
                    exc_data, max_pts=500, n_pca=15, max_dim=2)
                elapsed = timer.time() - t0

                stats = analyze_persistence(diagrams)
                stats['excursion_id'] = eid
                stats['duration'] = erow['duration']
                stats['n_pts'] = len(exc_data)
                stats['n_pca_dims'] = n_pca
                stats['var_explained'] = var_exp
                stats['region'] = region_label
                stats['resolution'] = res['label']
                all_results.append(stats)

                h1_gap = stats['h1_gap_ratio']
                h1_life = stats['h1_top_lifetime']
                h2_gap = stats['h2_gap_ratio']

                loop_flag = ' ** LOOP **' if h1_gap > 3 else ''
                void_flag = ' ** VOID **' if h2_gap > 3 else ''
                print(f" {elapsed:.1f}s | H1: life={h1_life:.3f} gap={h1_gap:.1f}x{loop_flag}"
                      f" | H2: gap={h2_gap:.1f}x{void_flag}")

                # Persistence diagram
                plot_persistence_diagram(
                    diagrams, axes[ei, 0],
                    title=f'Exc {eid} ({erow["duration"]:.0f}s, {len(exc_data)} pts)\n'
                          f'PCA: {n_pca}D ({var_exp:.0%} var)')

                # Barcode
                plot_persistence_barcode(
                    diagrams, axes[ei, 1],
                    title=f'Barcode')

                # H1/H2 gap analysis
                ax = axes[ei, 2]
                for dim in [1, 2]:
                    mask = diagrams[:, 2] == dim
                    features = diagrams[mask]
                    if len(features) == 0:
                        continue
                    lifetimes = features[:, 1] - features[:, 0]
                    finite_lt = np.sort(lifetimes[np.isfinite(lifetimes)])[::-1]
                    if len(finite_lt) > 0:
                        x = np.arange(1, len(finite_lt) + 1)
                        ax.bar(x + (0.15 if dim == 2 else -0.15), finite_lt,
                               width=0.3, color=DIM_COLORS[dim], alpha=0.7,
                               label=DIM_LABELS[dim])

                ax.set_xlabel('Feature rank', fontsize=9)
                ax.set_ylabel('Lifetime', fontsize=9)
                gap_str = f'H1 gap: {h1_gap:.1f}x'
                if h1_gap > 3:
                    gap_str += ' (PROMINENT LOOP)'
                elif h1_gap > 2:
                    gap_str += ' (possible loop)'
                gap_str += f'\nH2 gap: {h2_gap:.1f}x'
                if h2_gap > 3:
                    gap_str += ' (PROMINENT VOID)'
                ax.set_title(gap_str, fontsize=10,
                             color='#D32F2F' if h1_gap > 3 else 'black')
                ax.legend(fontsize=8)

            plt.tight_layout(rect=[0, 0, 1, 0.94])
            res_tag = f"{res['bin_ms']}ms"
            outpath = Path("figures") / f"persistent_homology_{region}_s1_{res_tag}.png"
            fig.savefig(outpath, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"    Saved: {outpath}")

            # ============================================
            # Run on ALL complete excursions (quick scan)
            # ============================================
            print(f"\n    Scanning all {len(complete)} complete excursions...")
            for _, erow in complete.iterrows():
                eid = int(erow['excursion_id'])
                if eid in [int(r['excursion_id']) for r in all_results
                           if r['region'] == region_label
                           and r['resolution'] == res['label']]:
                    continue  # already done

                exc_data, _ = extract_excursion_data(
                    zscore, time_sec, erow['start_time'], erow['end_time'])
                if len(exc_data) < 10:
                    continue

                try:
                    diagrams, _, n_pca, var_exp = run_persistence(
                        exc_data, max_pts=300, n_pca=10, max_dim=1)
                    stats = analyze_persistence(diagrams)
                except Exception:
                    continue

                stats['excursion_id'] = eid
                stats['duration'] = erow['duration']
                stats['n_pts'] = len(exc_data)
                stats['n_pca_dims'] = n_pca
                stats['var_explained'] = var_exp
                stats['region'] = region_label
                stats['resolution'] = res['label']
                all_results.append(stats)

            # Report loops found
            res_df = pd.DataFrame([r for r in all_results
                                    if r['region'] == region_label
                                    and r['resolution'] == res['label']])
            n_loops = (res_df['h1_gap_ratio'] > 3).sum()
            print(f"    Excursions with H1 gap > 3: {n_loops}/{len(res_df)}")
            if n_loops > 0:
                loops = res_df[res_df['h1_gap_ratio'] > 3].sort_values(
                    'h1_gap_ratio', ascending=False)
                for _, r in loops.head(5).iterrows():
                    print(f"      Exc {int(r['excursion_id'])}: "
                          f"gap={r['h1_gap_ratio']:.1f}x, "
                          f"lifetime={r['h1_top_lifetime']:.3f}, "
                          f"dur={r['duration']:.0f}s")

    # Save all results
    results_df = pd.DataFrame(all_results)
    outpath = Path("data") / "persistent_homology_s1.csv"
    results_df.to_csv(outpath, index=False)
    print(f"\nAll results saved: {outpath}")
    print(f"Total: {len(results_df)} excursion-resolution combos")

    # Summary
    print("\n" + "=" * 60)
    print("  SUMMARY: Excursions with prominent H1 loops (gap > 3)")
    print("=" * 60)
    for region_label in ['LHA', 'RSP']:
        for res_label in [r['label'] for r in resolutions]:
            sub = results_df[(results_df['region'] == region_label) &
                             (results_df['resolution'] == res_label)]
            n_loops = (sub['h1_gap_ratio'] > 3).sum()
            print(f"  {region_label} {res_label}: {n_loops}/{len(sub)} excursions")

    print("\nDone!")


if __name__ == "__main__":
    main()
