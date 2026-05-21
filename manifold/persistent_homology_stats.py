"""
Statistical Assessment of Persistent Homology Results
=====================================================
Are the excursions with prominent topology (H1 loops, H2 voids)
significantly different from other excursions?

Tests:
  1. Distribution of H1/H2 gap ratios across ALL excursions — where do
     the "prominent" ones fall?
  2. Null model: time-shuffled spike trains → persistence on shuffled data
     to build a null distribution of gap ratios
  3. Percentile rank and z-score of each prominent excursion vs null
  4. Multiple comparisons correction (Bonferroni / FDR)
  5. Effect of excursion duration on gap ratio (confound check)
"""

import yaml
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from scipy.ndimage import uniform_filter1d
from scipy.spatial.distance import pdist, squareform
from scipy import stats
from gtda.homology import VietorisRipsPersistence
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
N_SHUFFLES = 100  # number of null permutations

with open("paths.yaml") as f:
    paths_config = yaml.safe_load(f)

PROMINENT_IDS = {
    'rsp': {
        '200ms/500ms': [90, 71, 89, 88],
        '500ms/1s': [89, 5, 90, 13, 4],
        '50ms/200ms': [35, 89],
    },
    'lha': {
        '500ms/1s': [11, 89, 16],
    }
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


def run_persistence_quick(data, max_pts=300, n_pca=10, max_dim=1):
    """Fast persistence for null model (H0 + H1 only)."""
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

    # Extract H1 gap ratio
    h1_mask = diagrams[:, 2] == 1
    h1_features = diagrams[h1_mask]
    if len(h1_features) == 0:
        return 0.0, 0.0

    lifetimes = h1_features[:, 1] - h1_features[:, 0]
    finite_lt = np.sort(lifetimes[np.isfinite(lifetimes)])[::-1]
    if len(finite_lt) < 2 or finite_lt[1] <= 0:
        return finite_lt[0] if len(finite_lt) > 0 else 0.0, 0.0

    return finite_lt[0], finite_lt[0] / finite_lt[1]


def shuffle_excursion(exc_data):
    """Time-shuffle each neuron independently (destroys temporal structure)."""
    shuffled = np.copy(exc_data)
    for j in range(shuffled.shape[1]):
        np.random.shuffle(shuffled[:, j])
    return shuffled


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("Statistical Assessment of Persistent Homology")
    print("=" * 60)

    # Load observed results
    df = pd.read_csv("data/persistent_homology_s1.csv")
    print(f"Loaded {len(df)} excursion-resolution results")

    # Load neural data for null model
    sp = paths_config['single_probe']['coordinates_1']['mouse01']['sessions']
    sc = sp['session_1']
    sorted_path = Path(sc['sorted'])
    sorting = se.read_kilosort(sorted_path)
    lha_ids, rsp_ids = get_good_units_by_region(sorted_path)

    exc_df = pd.read_csv("data/excursions_session_1.csv")
    complete = exc_df[exc_df['label'] == 'complete']

    # =========================================================================
    # 1. Distribution of gap ratios across all excursions
    # =========================================================================
    print("\n" + "=" * 60)
    print("  1. H1 GAP RATIO DISTRIBUTIONS")
    print("=" * 60)

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle("H1 Gap Ratio Distribution — All Complete Excursions (Session 1)\n"
                 "Red dashed = gap > 3 threshold | Orange triangles = prominent excursions",
                 fontsize=14, fontweight='bold')

    for ri, region in enumerate(['LHA', 'RSP']):
        for ci, res_label in enumerate(['50ms/200ms', '200ms/500ms', '500ms/1s']):
            ax = axes[ri, ci]
            sub = df[(df['region'] == region) & (df['resolution'] == res_label)]
            gaps = sub['h1_gap_ratio'].values
            gaps = gaps[gaps > 0]  # remove zeros (no features)

            if len(gaps) == 0:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center')
                continue

            # Histogram
            bins = np.linspace(0.8, max(gaps.max(), 5), 30)
            ax.hist(gaps, bins=bins, color='#90CAF9', edgecolor='#1565C0',
                    alpha=0.8, label=f'All excursions (n={len(gaps)})')

            # Mark threshold
            ax.axvline(3.0, color='red', linestyle='--', linewidth=2, label='Gap > 3 threshold')

            # Mark prominent excursions
            prom_key = region.lower()
            if prom_key in PROMINENT_IDS and res_label in PROMINENT_IDS[prom_key]:
                prom_ids = PROMINENT_IDS[prom_key][res_label]
                prom_gaps = sub[sub['excursion_id'].isin(prom_ids)]['h1_gap_ratio'].values
                for pg in prom_gaps:
                    ax.axvline(pg, color='#FF6F00', linewidth=2, alpha=0.8)
                ax.scatter(prom_gaps, [0.5] * len(prom_gaps), marker='^',
                           color='#FF6F00', s=100, zorder=10, label='Prominent')

            # Stats
            n_above = (gaps > 3).sum()
            pct_above = 100 * n_above / len(gaps)
            ax.set_title(f'{region} — {res_label}\n'
                         f'Median={np.median(gaps):.2f}, Mean={np.mean(gaps):.2f}\n'
                         f'{n_above}/{len(gaps)} above threshold ({pct_above:.0f}%)',
                         fontsize=10)
            ax.set_xlabel('H1 Gap Ratio', fontsize=10)
            ax.set_ylabel('Count', fontsize=10)
            ax.legend(fontsize=7)

            # Print
            print(f"  {region} {res_label}: median={np.median(gaps):.2f}, "
                  f"mean={np.mean(gaps):.2f}, max={gaps.max():.1f}, "
                  f">{3}: {n_above}/{len(gaps)} ({pct_above:.0f}%)")

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig("figures/persistent_homology_gap_distributions.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: figures/persistent_homology_gap_distributions.png")

    # =========================================================================
    # 2. Null model: shuffled spike trains
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"  2. NULL MODEL ({N_SHUFFLES} shuffles per excursion)")
    print(f"{'='*60}")

    # Pick representative excursions for null model (mix of durations)
    # Use all complete excursions but only at the resolutions where we found loops
    test_configs = [
        ('rsp', rsp_ids, 200, 500, '200ms/500ms'),
        ('rsp', rsp_ids, 500, 1000, '500ms/1s'),
        ('lha', lha_ids, 500, 1000, '500ms/1s'),
    ]

    null_results = []
    observed_results = []

    for region, unit_ids, bin_ms, smooth_ms, res_label in test_configs:
        region_label = region.upper()
        print(f"\n  {region_label} @ {res_label}:")
        print(f"  Binning...", end='', flush=True)
        zscore, time_sec = bin_and_smooth(sorting, unit_ids, bin_ms, smooth_ms)
        print(f" done")

        # Get all complete excursions for this config
        sub = df[(df['region'] == region_label) & (df['resolution'] == res_label)]

        exc_count = 0
        for _, erow in complete.iterrows():
            eid = int(erow['excursion_id'])
            mask = (time_sec >= erow['start_time']) & (time_sec <= erow['end_time'])
            exc_data = zscore[mask]

            if len(exc_data) < 10:
                continue

            # Get observed gap ratio
            obs_row = sub[sub['excursion_id'] == eid]
            if len(obs_row) == 0:
                continue
            obs_gap = obs_row.iloc[0]['h1_gap_ratio']
            obs_lifetime = obs_row.iloc[0]['h1_top_lifetime']

            # Run null shuffles
            null_gaps = []
            null_lifetimes = []
            for s in range(N_SHUFFLES):
                shuffled = shuffle_excursion(exc_data)
                lt, gap = run_persistence_quick(shuffled, max_pts=200, n_pca=8)
                null_gaps.append(gap)
                null_lifetimes.append(lt)

            null_gaps = np.array(null_gaps)
            null_lifetimes = np.array(null_lifetimes)

            # Percentile rank of observed gap in null distribution
            percentile = 100 * np.mean(null_gaps < obs_gap)
            # p-value: fraction of null >= observed
            p_value = np.mean(null_gaps >= obs_gap)
            # z-score
            null_mean = np.mean(null_gaps)
            null_std = np.std(null_gaps)
            z_score = (obs_gap - null_mean) / null_std if null_std > 0 else 0

            is_prominent = obs_gap > 3
            observed_results.append({
                'excursion_id': eid,
                'region': region_label,
                'resolution': res_label,
                'duration': erow['duration'],
                'n_pts': len(exc_data),
                'obs_h1_gap': obs_gap,
                'obs_h1_lifetime': obs_lifetime,
                'null_mean_gap': null_mean,
                'null_std_gap': null_std,
                'null_max_gap': null_gaps.max(),
                'percentile': percentile,
                'p_value': p_value,
                'z_score': z_score,
                'is_prominent': is_prominent,
            })

            null_results.append({
                'region': region_label,
                'resolution': res_label,
                'excursion_id': eid,
                'null_gaps': null_gaps.tolist(),
            })

            exc_count += 1
            if exc_count % 10 == 0:
                print(f"    {exc_count} excursions done...", flush=True)

        print(f"    {exc_count} excursions completed")

    # Save results
    results_df = pd.DataFrame(observed_results)
    results_df.to_csv("data/persistent_homology_null_test.csv", index=False)
    print(f"\n  Saved: data/persistent_homology_null_test.csv")

    # =========================================================================
    # 3. Statistical summary
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"  3. STATISTICAL RESULTS")
    print(f"{'='*60}")

    for region_label in ['LHA', 'RSP']:
        for res_label in ['200ms/500ms', '500ms/1s']:
            sub = results_df[(results_df['region'] == region_label) &
                             (results_df['resolution'] == res_label)]
            if len(sub) == 0:
                continue

            prom = sub[sub['is_prominent']]
            non_prom = sub[~sub['is_prominent']]

            print(f"\n  {region_label} @ {res_label}:")
            print(f"    Total excursions: {len(sub)}")
            print(f"    Prominent (gap>3): {len(prom)}")

            if len(prom) > 0:
                print(f"    --- Prominent excursions ---")
                for _, r in prom.iterrows():
                    sig = '*' if r['p_value'] < 0.05 else ''
                    sig2 = '**' if r['p_value'] < 0.01 else sig
                    sig3 = '***' if r['p_value'] < 0.001 else sig2
                    print(f"      Exc {int(r['excursion_id'])}: "
                          f"gap={r['obs_h1_gap']:.1f}x, "
                          f"p={r['p_value']:.3f}{sig3}, "
                          f"z={r['z_score']:.1f}, "
                          f"percentile={r['percentile']:.0f}%, "
                          f"null_mean={r['null_mean_gap']:.2f}±{r['null_std_gap']:.2f}, "
                          f"dur={r['duration']:.0f}s")

            # Mann-Whitney: prominent vs non-prominent gap ratios
            if len(prom) >= 2 and len(non_prom) >= 2:
                U, mw_p = stats.mannwhitneyu(prom['obs_h1_gap'], non_prom['obs_h1_gap'],
                                              alternative='greater')
                print(f"    Mann-Whitney (prominent > others): U={U:.0f}, p={mw_p:.4f}")

            # Bonferroni correction
            if len(prom) > 0:
                n_tests = len(sub)
                bonf_threshold = 0.05 / n_tests
                n_survive = (prom['p_value'] < bonf_threshold).sum()
                print(f"    Bonferroni correction (alpha/n = 0.05/{n_tests} = {bonf_threshold:.4f}): "
                      f"{n_survive}/{len(prom)} survive")

    # =========================================================================
    # 4. Confound check: duration vs gap ratio
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"  4. CONFOUND CHECK: Duration vs Gap Ratio")
    print(f"{'='*60}")

    for region_label in ['LHA', 'RSP']:
        for res_label in ['200ms/500ms', '500ms/1s']:
            sub = results_df[(results_df['region'] == region_label) &
                             (results_df['resolution'] == res_label)]
            if len(sub) < 5:
                continue
            r_corr, p_corr = stats.spearmanr(sub['duration'], sub['obs_h1_gap'])
            print(f"  {region_label} {res_label}: Spearman(duration, gap) = "
                  f"{r_corr:.3f}, p={p_corr:.3f}")

    # =========================================================================
    # 5. Visualization
    # =========================================================================

    fig, axes = plt.subplots(2, 3, figsize=(22, 14))
    fig.suptitle("Persistent Homology — Statistical Validation\n"
                 "Observed H1 gap ratios vs null distribution (time-shuffled spike trains)",
                 fontsize=15, fontweight='bold')

    plot_idx = 0
    for ri, region_label in enumerate(['RSP', 'LHA']):
        for ci, res_label in enumerate(['200ms/500ms', '500ms/1s']):
            sub = results_df[(results_df['region'] == region_label) &
                             (results_df['resolution'] == res_label)]
            if len(sub) == 0:
                axes[ri, ci].set_visible(False)
                continue

            ax = axes[ri, ci]

            # Plot observed gap ratios sorted
            sub_sorted = sub.sort_values('obs_h1_gap', ascending=False)
            x = np.arange(len(sub_sorted))
            colors = ['#D32F2F' if p else '#90CAF9'
                      for p in sub_sorted['is_prominent'].values]
            edgecolors = ['black' if p else '#1565C0'
                          for p in sub_sorted['is_prominent'].values]

            bars = ax.bar(x, sub_sorted['obs_h1_gap'].values,
                         color=colors, edgecolor=edgecolors, alpha=0.8, linewidth=0.5)

            # Null mean + std band
            null_mean = sub_sorted['null_mean_gap'].values
            null_std = sub_sorted['null_std_gap'].values
            ax.fill_between(x, null_mean - null_std, null_mean + null_std,
                           alpha=0.3, color='gray', label='Null mean ± 1 SD')
            ax.plot(x, null_mean, 'k--', linewidth=1, alpha=0.5, label='Null mean')

            # Threshold line
            ax.axhline(3.0, color='green', linestyle=':', linewidth=2,
                       alpha=0.7, label='Gap = 3 threshold')

            # Label prominent bars
            for i, (_, row) in enumerate(sub_sorted.iterrows()):
                if row['is_prominent']:
                    p_str = f"p={row['p_value']:.3f}"
                    if row['p_value'] < 0.001:
                        p_str = f"p<0.001"
                    ax.text(i, row['obs_h1_gap'] + 0.15,
                            f"Exc {int(row['excursion_id'])}\n{p_str}",
                            ha='center', va='bottom', fontsize=7, fontweight='bold',
                            color='#D32F2F')

            ax.set_xlabel('Excursions (ranked by gap ratio)', fontsize=10)
            ax.set_ylabel('H1 Gap Ratio', fontsize=10)
            ax.set_title(f'{region_label} — {res_label}\n'
                         f'n={len(sub)} excursions, {N_SHUFFLES} shuffles each',
                         fontsize=11, fontweight='bold')
            ax.legend(fontsize=8)
            ax.set_xticks([])

    # Third column: p-value volcano plot
    for ri, region_label in enumerate(['RSP', 'LHA']):
        ax = axes[ri, 2]
        sub = results_df[results_df['region'] == region_label]
        if len(sub) == 0:
            ax.set_visible(False)
            continue

        # Plot gap ratio vs -log10(p)
        gap_vals = sub['obs_h1_gap'].values
        p_vals = sub['p_value'].values
        p_vals_adj = np.clip(p_vals, 1e-10, 1)  # avoid log(0)
        neg_log_p = -np.log10(p_vals_adj)
        is_sig = p_vals < 0.05
        is_prom = sub['is_prominent'].values

        ax.scatter(gap_vals[~is_prom], neg_log_p[~is_prom],
                   c='#90CAF9', s=40, alpha=0.7, edgecolors='#1565C0',
                   linewidths=0.5, label='Non-prominent')
        ax.scatter(gap_vals[is_prom], neg_log_p[is_prom],
                   c='#D32F2F', s=100, alpha=0.9, edgecolors='black',
                   linewidths=1, marker='*', label='Prominent (gap>3)')

        # Label prominent points
        for _, row in sub[sub['is_prominent']].iterrows():
            p_adj = max(row['p_value'], 1e-10)
            ax.annotate(f"Exc {int(row['excursion_id'])}\n({row['resolution']})",
                        (row['obs_h1_gap'], -np.log10(p_adj)),
                        fontsize=7, fontweight='bold',
                        xytext=(5, 5), textcoords='offset points')

        # Significance lines
        ax.axhline(-np.log10(0.05), color='orange', linestyle='--',
                   linewidth=1.5, label='p=0.05')
        ax.axhline(-np.log10(0.01), color='red', linestyle='--',
                   linewidth=1.5, label='p=0.01')
        ax.axvline(3.0, color='green', linestyle=':', linewidth=1.5,
                   alpha=0.7, label='Gap=3')

        ax.set_xlabel('H1 Gap Ratio', fontsize=10)
        ax.set_ylabel('-log10(p-value)', fontsize=10)
        ax.set_title(f'{region_label} — Volcano Plot\n'
                     f'Gap ratio vs significance (all resolutions)',
                     fontsize=11, fontweight='bold')
        ax.legend(fontsize=7, loc='upper left')

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig("figures/persistent_homology_stats.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("\n  Saved: figures/persistent_homology_stats.png")

    # =========================================================================
    # 6. Duration confound figure
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Confound Check: Excursion Duration vs H1 Gap Ratio",
                 fontsize=14, fontweight='bold')

    for ri, region_label in enumerate(['RSP', 'LHA']):
        ax = axes[ri]
        sub = results_df[results_df['region'] == region_label]
        if len(sub) == 0:
            continue

        for res_label, marker, color in [('200ms/500ms', 'o', '#1976D2'),
                                          ('500ms/1s', 's', '#D32F2F')]:
            rsub = sub[sub['resolution'] == res_label]
            if len(rsub) == 0:
                continue
            prom = rsub[rsub['is_prominent']]
            non_prom = rsub[~rsub['is_prominent']]

            ax.scatter(non_prom['duration'], non_prom['obs_h1_gap'],
                       c=color, marker=marker, s=40, alpha=0.5,
                       label=f'{res_label}')
            ax.scatter(prom['duration'], prom['obs_h1_gap'],
                       c=color, marker=marker, s=120, alpha=0.9,
                       edgecolors='black', linewidths=2,
                       label=f'{res_label} prominent')

            r_corr, p_corr = stats.spearmanr(rsub['duration'], rsub['obs_h1_gap'])
            ax.text(0.98, 0.98 - 0.06 * (['200ms/500ms', '500ms/1s'].index(res_label)),
                    f'{res_label}: rho={r_corr:.2f}, p={p_corr:.2f}',
                    transform=ax.transAxes, ha='right', va='top', fontsize=9,
                    color=color, fontweight='bold')

        ax.axhline(3.0, color='green', linestyle=':', linewidth=1.5, alpha=0.7)
        ax.set_xlabel('Excursion Duration (s)', fontsize=10)
        ax.set_ylabel('H1 Gap Ratio', fontsize=10)
        ax.set_title(f'{region_label}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig("figures/persistent_homology_duration_confound.png",
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: figures/persistent_homology_duration_confound.png")

    # =========================================================================
    # Summary table
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"  SUMMARY TABLE — Prominent Excursions vs Null")
    print(f"{'='*60}")
    prom = results_df[results_df['is_prominent']].sort_values(
        ['region', 'resolution', 'obs_h1_gap'], ascending=[True, True, False])
    if len(prom) > 0:
        print(f"\n  {'Exc':>4} {'Region':>6} {'Resolution':>12} {'Gap':>5} "
              f"{'p-val':>7} {'z':>5} {'%ile':>5} "
              f"{'Null mean':>10} {'Dur(s)':>7}")
        print(f"  {'-'*72}")
        for _, r in prom.iterrows():
            sig = ''
            if r['p_value'] < 0.001:
                sig = '***'
            elif r['p_value'] < 0.01:
                sig = '**'
            elif r['p_value'] < 0.05:
                sig = '*'
            print(f"  {int(r['excursion_id']):>4} {r['region']:>6} {r['resolution']:>12} "
                  f"{r['obs_h1_gap']:>5.1f} {r['p_value']:>6.3f}{sig} "
                  f"{r['z_score']:>5.1f} {r['percentile']:>4.0f}% "
                  f"{r['null_mean_gap']:>5.2f}±{r['null_std_gap']:.2f} "
                  f"{r['duration']:>6.0f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
