"""
HFD LHA-ACA Cross-Probe Analysis -- Good Units Only
LHA from Probe-1 (depth 0-345 um) x ACA from Probe-0.
Fed-HFD sessions: 17-22
Includes depth fallback for sessions missing cluster_info.tsv.
Compares: HFD Exploration vs Foraging (phase), and Fed vs Fasted vs HFD (state).
"""

import yaml
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as sp_stats
import spikeinterface.extractors as se
import warnings
import time

warnings.filterwarnings('ignore')

with open("paths.yaml") as f:
    paths_config = yaml.safe_load(f)

LHA_DEPTH_MIN = 0
LHA_DEPTH_MAX = 345  # um

LAGS_MS = [2, 5, 10, 50, 100]
BIN_SIZE_MS = 1
FS = 30000
BIN_SAMPLES = int(BIN_SIZE_MS * FS / 1000)
LAG_BINS = {lag: int(lag * FS / (1000 * BIN_SAMPLES)) for lag in LAGS_MS}


def compute_cluster_depths(sorted_path_obj):
    """Compute depth per cluster from raw KS3 output files."""
    templates_path = sorted_path_obj / "templates.npy"
    chan_pos_path = sorted_path_obj / "channel_positions.npy"

    if not templates_path.exists() or not chan_pos_path.exists():
        return {}

    templates = np.load(templates_path)
    channel_positions = np.load(chan_pos_path)

    peak_channels = np.argmax(np.max(np.abs(templates), axis=1), axis=1)
    depths = channel_positions[peak_channels, 1]

    spike_clusters_path = sorted_path_obj / "spike_clusters.npy"
    spike_templates_path = sorted_path_obj / "spike_templates.npy"

    if spike_clusters_path.exists() and spike_templates_path.exists():
        spike_clusters = np.load(spike_clusters_path).flatten()
        spike_templates = np.load(spike_templates_path).flatten()
        unique_clusters = np.unique(spike_clusters)
        cluster_depths = {}
        for cid in unique_clusters:
            mask = spike_clusters == cid
            templates_for_cluster = spike_templates[mask]
            if len(templates_for_cluster) > 0:
                most_common_template = np.bincount(templates_for_cluster).argmax()
                if most_common_template < len(depths):
                    cluster_depths[cid] = depths[most_common_template]
        return cluster_depths
    else:
        return {i: depths[i] for i in range(len(depths))}


def get_good_unit_ids(sorted_path_obj):
    """Get unit IDs labeled 'good' from cluster_info.tsv or cluster_group.tsv."""
    ci = sorted_path_obj / "cluster_info.tsv"
    if ci.exists():
        df = pd.read_csv(ci, sep='\t')
        if 'group' in df.columns and df['group'].eq('good').any():
            return df[df['group'] == 'good']['cluster_id'].values
        if 'KSLabel' in df.columns:
            return df[df['KSLabel'] == 'good']['cluster_id'].values

    cg = sorted_path_obj / "cluster_group.tsv"
    if cg.exists():
        df = pd.read_csv(cg, sep='\t')
        col = df.columns[1]
        return df[df[col].str.strip() == 'good'].iloc[:, 0].values

    return np.array([])


def get_good_lha_unit_ids(sorted_path_obj):
    """Get unit IDs labeled 'good' AND in LHA depth range (0-345 um).
    Falls back to computing depth from templates if cluster_info.tsv is missing.
    """
    ci = sorted_path_obj / "cluster_info.tsv"
    if ci.exists():
        df = pd.read_csv(ci, sep='\t')
        if 'depth' in df.columns:
            label_col = None
            if 'group' in df.columns and df['group'].eq('good').any():
                label_col = 'group'
            elif 'KSLabel' in df.columns:
                label_col = 'KSLabel'
            if label_col is not None:
                good_lha = df[(df[label_col] == 'good') &
                              (df['depth'] >= LHA_DEPTH_MIN) &
                              (df['depth'] <= LHA_DEPTH_MAX)]
                return good_lha['cluster_id'].values
            print(f"    [WARN] No label column in cluster_info.tsv")
            return np.array([])

    # Fallback: cluster_group.tsv for labels + compute depth from templates
    print(f"    [INFO] No cluster_info.tsv, computing depths from templates...")
    cluster_depths = compute_cluster_depths(sorted_path_obj)
    if not cluster_depths:
        print(f"    [WARN] Cannot compute depths from templates")
        return np.array([])

    cg = sorted_path_obj / "cluster_group.tsv"
    if not cg.exists():
        cg = sorted_path_obj / "cluster_KSLabel.tsv"
    if not cg.exists():
        print(f"    [WARN] No cluster_group.tsv or cluster_KSLabel.tsv found")
        return np.array([])

    df = pd.read_csv(cg, sep='\t')
    col = df.columns[1]
    good_ids = df[df[col].str.strip() == 'good'].iloc[:, 0].values

    lha_ids = []
    for cid in good_ids:
        if cid in cluster_depths:
            d = cluster_depths[cid]
            if LHA_DEPTH_MIN <= d <= LHA_DEPTH_MAX:
                lha_ids.append(cid)

    if len(good_ids) > 0:
        print(f"    [INFO] {len(good_ids)} good units, {len(lha_ids)} in LHA depth range (depths from templates)")

    return np.array(lha_ids)


def prebin_spike_trains(sorting, unit_ids, global_min, global_max):
    """Bin and z-score spike trains using a shared time axis."""
    n_bins = int((global_max - global_min) / BIN_SAMPLES) + 1
    binned = {}
    for uid in unit_ids:
        st = sorting.get_unit_spike_train(uid)
        t = np.zeros(n_bins)
        if len(st) > 0:
            b = ((st - global_min) // BIN_SAMPLES).astype(int)
            b = b[(b >= 0) & (b < n_bins)]
            np.add.at(t, b, 1)
        std_val = np.std(t)
        if std_val > 1e-8:
            t = (t - np.mean(t)) / std_val
        else:
            t = t - np.mean(t)
        binned[uid] = t
    return binned, n_bins


def get_global_time_range(sorting, unit_ids):
    """Get min/max spike times across all units."""
    all_min, all_max = np.inf, 0
    for uid in unit_ids:
        st = sorting.get_unit_spike_train(uid)
        if len(st) > 0:
            all_min = min(all_min, np.min(st))
            all_max = max(all_max, np.max(st))
    return all_min, all_max


def cross_corr_fast(t1_z, t2_z, n_bins):
    """Compute cross-correlation at all lags using fast dot products."""
    results = {}
    for lag_ms in LAGS_MS:
        lb = LAG_BINS[lag_ms]
        if lb >= n_bins:
            results[lag_ms] = np.nan
        elif lb == 0:
            results[lag_ms] = np.dot(t1_z, t2_z) / n_bins
        else:
            results[lag_ms] = np.dot(t1_z[lb:], t2_z[:-lb]) / (n_bins - lb)
    return results


def compute_cohens_d(g1, g2):
    n1, n2 = len(g1), len(g2)
    v1, v2 = np.var(g1, ddof=1), np.var(g2, ddof=1)
    ps = np.sqrt(((n1-1)*v1 + (n2-1)*v2) / (n1+n2-2))
    return (np.mean(g1) - np.mean(g2)) / ps if ps > 0 else 0

def fmt_p(p):
    if p < 0.0001: return "p < 0.0001***"
    elif p < 0.001: return f"p = {p:.2e}***"
    elif p < 0.01: return f"p = {p:.4f}**"
    elif p < 0.05: return f"p = {p:.4f}*"
    else: return f"p = {p:.4f} ns"


# =============================================================================
# HFD SESSION GROUPS
# =============================================================================

session_groups = {
    'hfd_exploration': [('session_17', 17), ('session_19', 19), ('session_21', 21)],
    'hfd_foraging':    [('session_18', 18), ('session_20', 20), ('session_22', 22)]
}

session_meta = {
    17: ('fed-HFD', 'exploration'), 18: ('fed-HFD', 'foraging'),
    19: ('fed-HFD', 'exploration'), 20: ('fed-HFD', 'foraging'),
    21: ('fed-HFD', 'exploration'), 22: ('fed-HFD', 'foraging'),
}

print("=" * 70)
print("HFD LHA-ACA CROSS-PROBE ANALYSIS -- GOOD UNITS ONLY")
print("LHA from Probe-1 (0-345um) x ACA from Probe-0")
print("Fed-HFD Sessions: 17-22")
print("=" * 70)

# =============================================================================
# STEP 1: CROSS-CORRELATIONS
# =============================================================================

all_connectivity = {}
overall_start = time.time()

for group_name, session_list in session_groups.items():
    print(f"\n{'='*70}")
    print(f"GROUP: {group_name.upper()}")
    print(f"{'='*70}")

    group_pairs = []

    for session_name, session_num in session_list:
        session_config = paths_config["double_probe"]["coordinates_1"]["mouse01"]["sessions"][session_name]

        p0 = session_config.get("probe_0_aca", {})
        p0_sorted = p0.get("sorted") if p0 else None

        p1 = session_config.get("probe_1_lha_rsp", {})
        p1_sorted = p1.get("sorted") if p1 else None

        if p0_sorted is None or p0_sorted == '' or p1_sorted is None or p1_sorted == '':
            print(f"  Session {session_num}: [SKIP] Missing sorted path (p0={p0_sorted is not None and p0_sorted != ''}, p1={p1_sorted is not None and p1_sorted != ''})")
            continue

        p0_path = Path(p0_sorted)
        p1_path = Path(p1_sorted)

        if not p0_path.exists() or not p1_path.exists():
            print(f"  Session {session_num}: [SKIP] Path does not exist")
            continue

        aca_ids = get_good_unit_ids(p0_path)
        lha_ids = get_good_lha_unit_ids(p1_path)

        print(f"  Session {session_num}: ACA={len(aca_ids)} good, LHA={len(lha_ids)} good (0-345um)")

        if len(aca_ids) < 1 or len(lha_ids) < 1:
            print(f"    [SKIP] Need at least 1 unit from each region")
            continue

        try:
            sorting_aca = se.read_kilosort(p0_path)
            sorting_lha = se.read_kilosort(p1_path)
        except Exception as e:
            print(f"    [ERROR] {e}")
            continue

        avail_aca = set(sorting_aca.get_unit_ids())
        avail_lha = set(sorting_lha.get_unit_ids())
        aca_ids = np.array([u for u in aca_ids if u in avail_aca])
        lha_ids = np.array([u for u in lha_ids if u in avail_lha])

        if len(aca_ids) < 1 or len(lha_ids) < 1:
            print(f"    [SKIP] After filtering: ACA={len(aca_ids)}, LHA={len(lha_ids)}")
            continue

        t0 = time.time()
        print(f"    Pre-binning spike trains (1ms bins)...")
        aca_min, aca_max = get_global_time_range(sorting_aca, aca_ids)
        lha_min, lha_max = get_global_time_range(sorting_lha, lha_ids)
        global_min = min(aca_min, lha_min)
        global_max = max(aca_max, lha_max)

        binned_aca, n_bins = prebin_spike_trains(sorting_aca, aca_ids, global_min, global_max)
        binned_lha, _ = prebin_spike_trains(sorting_lha, lha_ids, global_min, global_max)
        print(f"    Binned ACA={len(aca_ids)}, LHA={len(lha_ids)} into {n_bins} bins ({n_bins/1000:.1f}s) in {time.time()-t0:.1f}s")

        total_pairs = len(lha_ids) * len(aca_ids)
        pair_count = 0
        t1_time = time.time()

        for lha_uid in lha_ids:
            t1_z = binned_lha[lha_uid]
            for aca_uid in aca_ids:
                t2_z = binned_aca[aca_uid]
                cc = cross_corr_fast(t1_z, t2_z, n_bins)
                group_pairs.append({
                    'lha_unit': lha_uid, 'aca_unit': aca_uid,
                    'session': session_num,
                    'correlation_2ms': cc[2], 'correlation_5ms': cc[5],
                    'correlation_10ms': cc[10], 'correlation_50ms': cc[50], 'correlation_100ms': cc[100]
                })
                pair_count += 1
                if pair_count % 10000 == 0:
                    elapsed = time.time() - t1_time
                    rate = pair_count / elapsed
                    remaining = (total_pairs - pair_count) / rate if rate > 0 else 0
                    print(f"      {pair_count}/{total_pairs} ({pair_count/total_pairs*100:.1f}%) - {remaining:.0f}s left")

        elapsed = time.time() - t0
        print(f"    Session {session_num}: {pair_count} LHA x ACA pairs in {elapsed:.1f}s")

    if group_pairs:
        all_connectivity[group_name] = pd.DataFrame(group_pairs)
        print(f"\n  {group_name}: {len(group_pairs)} total pairs")
        out_file = f"data/hfd_lha_aca_good_connectivity_{group_name}.csv"
        all_connectivity[group_name].to_csv(out_file, index=False)
        print(f"  [OK] Saved {out_file}")

total_time = time.time() - overall_start
print(f"\n[CROSS-CORRELATIONS DONE] {total_time:.1f}s ({total_time/60:.1f} min)")


# =============================================================================
# STEP 2: SESSION-LEVEL STATISTICAL ANALYSIS
# =============================================================================

print(f"\n{'='*70}")
print("SESSION-LEVEL STATISTICAL ANALYSIS")
print("=" * 70)

all_dfs = []
for group_name, df in all_connectivity.items():
    all_dfs.append(df)

if not all_dfs:
    print("[ERROR] No data to analyze - no sessions had valid LHA+ACA pairs")
    print("[DONE] HFD LHA-ACA cross-probe analysis ended (no data)")
    exit()

full_df = pd.concat(all_dfs, ignore_index=True)

lag_cols = [f'correlation_{lag}ms' for lag in LAGS_MS]
session_means = full_df.groupby('session')[lag_cols].mean().reset_index()
session_means['state'] = session_means['session'].map(lambda s: session_meta[s][0])
session_means['phase'] = session_means['session'].map(lambda s: session_meta[s][1])
session_means['n_pairs'] = full_df.groupby('session').size().values

print(f"\nPer-session mean correlations (N pairs in parentheses):")
for _, row in session_means.iterrows():
    s = int(row['session'])
    print(f"  Session {s} [{row['state']}/{row['phase']}] (n={int(row['n_pairs'])}): " +
          ", ".join([f"{lag}ms={row[f'correlation_{lag}ms']:.6f}" for lag in LAGS_MS]))

n_hfd = len(session_means)
n_exp = len(session_means[session_means['phase'] == 'exploration'])
n_for = len(session_means[session_means['phase'] == 'foraging'])
print(f"\nTotal pairs: {len(full_df)} across {n_hfd} HFD sessions")
print(f"  Exploration sessions: {n_exp}, Foraging sessions: {n_for}")

session_means.to_csv("data/hfd_lha_aca_good_session_means.csv", index=False)

# --- PHASE COMPARISON (within HFD) ---
print(f"\n--- PHASE COMPARISON (HFD Exploration vs Foraging, session-level) ---")
phase_results = []
for lag in LAGS_MS:
    col = f'correlation_{lag}ms'
    exp_vals = session_means[session_means['phase'] == 'exploration'][col].values
    for_vals = session_means[session_means['phase'] == 'foraging'][col].values
    if len(exp_vals) < 2 or len(for_vals) < 2:
        print(f"  Lag {lag}ms: [SKIP] Not enough sessions (exp={len(exp_vals)}, for={len(for_vals)})")
        continue
    mw_s, mw_p = sp_stats.mannwhitneyu(exp_vals, for_vals, alternative='two-sided')
    d = compute_cohens_d(exp_vals, for_vals)
    em = np.mean(exp_vals); e_sem = sp_stats.sem(exp_vals)
    fmm = np.mean(for_vals); f_sem = sp_stats.sem(for_vals)
    fold = ((fmm - em) / abs(em)) * 100 if em != 0 else 0
    phase_results.append({
        'Lag_ms': lag, 'Exploration_Mean': em, 'Exploration_SEM': e_sem, 'Exploration_Sessions': exp_vals.tolist(),
        'Foraging_Mean': fmm, 'Foraging_SEM': f_sem, 'Foraging_Sessions': for_vals.tolist(),
        'MannWhitney_Statistic': mw_s, 'MannWhitney_Pvalue': mw_p,
        'Cohens_d': d, 'N_Exploration': len(exp_vals), 'N_Foraging': len(for_vals)
    })
    print(f"  Lag {lag}ms: Exp={em:.6f}+/-{e_sem:.6f} (N={len(exp_vals)}), "
          f"For={fmm:.6f}+/-{f_sem:.6f} (N={len(for_vals)}), "
          f"Change={fold:+.1f}%, {fmt_p(mw_p)}, d={d:.4f}")

phase_df = pd.DataFrame(phase_results)
if len(phase_df) > 0:
    save_cols = ['Lag_ms', 'Exploration_Mean', 'Exploration_SEM', 'Foraging_Mean', 'Foraging_SEM',
                 'MannWhitney_Statistic', 'MannWhitney_Pvalue', 'Cohens_d', 'N_Exploration', 'N_Foraging']
    phase_df[save_cols].to_csv("data/hfd_lha_aca_good_stats_phase_comparison.csv", index=False)


# =============================================================================
# STEP 3: 3-WAY STATE COMPARISON (Fed vs Fasted vs HFD)
# =============================================================================

print(f"\n{'='*70}")
print("3-WAY STATE COMPARISON: Fed vs Fasted vs HFD (session-level)")
print("=" * 70)

existing_means_path = Path("data/double_probe_lha_aca_good_session_means.csv")
state_comparison_results = []

if existing_means_path.exists():
    existing_means = pd.read_csv(existing_means_path)
    fed_means = existing_means[existing_means['state'] == 'fed']
    fasted_means = existing_means[existing_means['state'] == 'fasted']
    hfd_means = session_means.copy()

    n_fed = len(fed_means)
    n_fas = len(fasted_means)

    print(f"  Fed sessions: {n_fed}, Fasted sessions: {n_fas}, HFD sessions: {n_hfd}")

    for lag in LAGS_MS:
        col = f'correlation_{lag}ms'
        fed_v = fed_means[col].values
        fas_v = fasted_means[col].values
        hfd_v = hfd_means[col].values

        print(f"\n  Lag {lag}ms:")

        if len(fed_v) >= 2 and len(fas_v) >= 2 and len(hfd_v) >= 2:
            kw_s, kw_p = sp_stats.kruskal(fed_v, fas_v, hfd_v)
            print(f"    Kruskal-Wallis: H={kw_s:.3f}, {fmt_p(kw_p)}")
        else:
            kw_s, kw_p = np.nan, np.nan

        comparisons = [
            ('Fed vs HFD', fed_v, hfd_v, n_fed, n_hfd),
            ('Fasted vs HFD', fas_v, hfd_v, n_fas, n_hfd),
            ('Fed vs Fasted', fed_v, fas_v, n_fed, n_fas)
        ]

        row_data = {'Lag_ms': lag, 'KW_Statistic': kw_s, 'KW_Pvalue': kw_p}

        for label, g1, g2, n1, n2 in comparisons:
            if len(g1) >= 2 and len(g2) >= 2:
                mw_s, mw_p = sp_stats.mannwhitneyu(g1, g2, alternative='two-sided')
                d = compute_cohens_d(g1, g2)
                m1, m2 = np.mean(g1), np.mean(g2)
                fold = ((m2 - m1) / abs(m1)) * 100 if m1 != 0 else 0
                print(f"    {label}: {m1:.6f} vs {m2:.6f} ({fold:+.1f}%), {fmt_p(mw_p)}, d={d:.4f}")
                prefix = label.replace(' vs ', '_vs_').replace(' ', '_')
                row_data[f'{prefix}_p'] = mw_p
                row_data[f'{prefix}_d'] = d
                row_data[f'{prefix}_change_pct'] = fold

        row_data['Fed_Mean'] = np.mean(fed_v) if len(fed_v) > 0 else np.nan
        row_data['Fed_SEM'] = sp_stats.sem(fed_v) if len(fed_v) > 1 else np.nan
        row_data['Fasted_Mean'] = np.mean(fas_v) if len(fas_v) > 0 else np.nan
        row_data['Fasted_SEM'] = sp_stats.sem(fas_v) if len(fas_v) > 1 else np.nan
        row_data['HFD_Mean'] = np.mean(hfd_v) if len(hfd_v) > 0 else np.nan
        row_data['HFD_SEM'] = sp_stats.sem(hfd_v) if len(hfd_v) > 1 else np.nan
        row_data['Fed_Sessions'] = fed_v.tolist()
        row_data['Fasted_Sessions'] = fas_v.tolist()
        row_data['HFD_Sessions'] = hfd_v.tolist()

        state_comparison_results.append(row_data)

    state_3way_df = pd.DataFrame(state_comparison_results)
    save_cols = [c for c in state_3way_df.columns if c not in ['Fed_Sessions', 'Fasted_Sessions', 'HFD_Sessions']]
    state_3way_df[save_cols].to_csv("data/hfd_lha_aca_good_stats_3way_state_comparison.csv", index=False)
else:
    print("  [WARN] No existing session means at data/double_probe_lha_aca_good_session_means.csv")
    print("  Skipping 3-way state comparison")
    fed_means = pd.DataFrame()
    fasted_means = pd.DataFrame()
    n_fed = 0
    n_fas = 0


# =============================================================================
# STEP 4: FIGURES
# =============================================================================

print(f"\n{'='*70}")
print("GENERATING FIGURES")
print("=" * 70)

w = 0.25

# Figure 1: Phase comparison
if len(phase_df) > 0:
    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(phase_df))
    em_vals = phase_df['Exploration_Mean'].values
    e_sem_vals = phase_df['Exploration_SEM'].values
    fm_vals = phase_df['Foraging_Mean'].values
    f_sem_vals = phase_df['Foraging_SEM'].values
    ax.bar(x-w/2, em_vals, w, yerr=e_sem_vals, label=f'HFD Exploration (N={n_exp})', capsize=8, color='#2ecc71', alpha=0.7, error_kw={'linewidth': 2})
    ax.bar(x+w/2, fm_vals, w, yerr=f_sem_vals, label=f'HFD Foraging (N={n_for})', capsize=8, color='#f39c12', alpha=0.7, error_kw={'linewidth': 2})
    for i, row in phase_df.iterrows():
        jitter = 0.06
        ax.scatter([i - w/2 + np.random.uniform(-jitter, jitter) for _ in row['Exploration_Sessions']],
                   row['Exploration_Sessions'], color='#2c3e50', s=40, zorder=5, edgecolors='white', linewidth=0.5)
        ax.scatter([i + w/2 + np.random.uniform(-jitter, jitter) for _ in row['Foraging_Sessions']],
                   row['Foraging_Sessions'], color='#2c3e50', s=40, zorder=5, edgecolors='white', linewidth=0.5)
    ax.set_xlabel('Time Lag (ms)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Session Mean Cross-Correlation +/- SEM', fontsize=12, fontweight='bold')
    ax.set_title('HFD LHA-ACA (Good Units): Exp vs For (Session-Level)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(phase_df['Lag_ms'].values.astype(int))
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    all_pts = np.concatenate([np.concatenate(phase_df['Exploration_Sessions'].values), np.concatenate(phase_df['Foraging_Sessions'].values)])
    ymin, ymax = np.min(all_pts), np.max(all_pts)
    margin = (ymax - ymin) * 0.4
    ax.set_ylim(ymin - margin * 0.3, ymax + margin * 1.2)
    for i, row in phase_df.iterrows():
        yp = max(max(row['Exploration_Sessions']), max(row['Foraging_Sessions']))
        ax.text(i, yp + margin*0.25, fmt_p(row['MannWhitney_Pvalue']), ha='center', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', alpha=0.7))
        ax.text(i, yp + margin*0.55, f"d = {row['Cohens_d']:.3f}", ha='center', fontsize=8, style='italic')
    plt.tight_layout()
    plt.savefig("figures/hfd_lha_aca_good_phase_comparison.png", dpi=150, bbox_inches='tight')
    print("[OK] Saved hfd_lha_aca_good_phase_comparison.png")
    plt.close()

# Figure 2: 3-way state comparison
if state_comparison_results and existing_means_path.exists():
    state_3way_df_plot = pd.DataFrame(state_comparison_results)
    fig, ax = plt.subplots(figsize=(16, 8))
    x = np.arange(len(state_3way_df_plot))

    fed_m = state_3way_df_plot['Fed_Mean'].values
    fed_se = state_3way_df_plot['Fed_SEM'].values
    fas_m = state_3way_df_plot['Fasted_Mean'].values
    fas_se = state_3way_df_plot['Fasted_SEM'].values
    hfd_m = state_3way_df_plot['HFD_Mean'].values
    hfd_se = state_3way_df_plot['HFD_SEM'].values

    ax.bar(x-w, fed_m, w, yerr=fed_se, label=f'Fed (N={n_fed})', capsize=6, color='#3498db', alpha=0.7, error_kw={'linewidth': 2})
    ax.bar(x, fas_m, w, yerr=fas_se, label=f'Fasted (N={n_fas})', capsize=6, color='#e74c3c', alpha=0.7, error_kw={'linewidth': 2})
    ax.bar(x+w, hfd_m, w, yerr=hfd_se, label=f'HFD (N={n_hfd})', capsize=6, color='#9b59b6', alpha=0.7, error_kw={'linewidth': 2})

    for i, row in state_3way_df_plot.iterrows():
        jitter = 0.04
        ax.scatter([i - w + np.random.uniform(-jitter, jitter) for _ in row['Fed_Sessions']],
                   row['Fed_Sessions'], color='#2c3e50', s=30, zorder=5, edgecolors='white', linewidth=0.5)
        ax.scatter([i + np.random.uniform(-jitter, jitter) for _ in row['Fasted_Sessions']],
                   row['Fasted_Sessions'], color='#2c3e50', s=30, zorder=5, edgecolors='white', linewidth=0.5)
        ax.scatter([i + w + np.random.uniform(-jitter, jitter) for _ in row['HFD_Sessions']],
                   row['HFD_Sessions'], color='#2c3e50', s=30, zorder=5, edgecolors='white', linewidth=0.5)

    ax.set_xlabel('Time Lag (ms)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Session Mean Cross-Correlation +/- SEM', fontsize=12, fontweight='bold')
    ax.set_title('LHA-ACA (Good Units): Fed vs Fasted vs HFD (Session-Level)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(state_3way_df_plot['Lag_ms'].values.astype(int))
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')

    all_pts = np.concatenate([np.concatenate(state_3way_df_plot['Fed_Sessions'].values),
                               np.concatenate(state_3way_df_plot['Fasted_Sessions'].values),
                               np.concatenate(state_3way_df_plot['HFD_Sessions'].values)])
    ymin, ymax = np.min(all_pts), np.max(all_pts)
    margin = (ymax - ymin) * 0.4
    ax.set_ylim(ymin - margin * 0.3, ymax + margin * 1.5)

    for i, row in state_3way_df_plot.iterrows():
        yp = max(max(row['Fed_Sessions']), max(row['Fasted_Sessions']), max(row['HFD_Sessions']))
        ax.text(i, yp + margin*0.25, f"KW {fmt_p(row['KW_Pvalue'])}", ha='center', fontsize=8, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig("figures/hfd_lha_aca_good_3way_state_comparison.png", dpi=150, bbox_inches='tight')
    print("[OK] Saved hfd_lha_aca_good_3way_state_comparison.png")
    plt.close()

# Figure 3: Summary tables
fig, axes = plt.subplots(1, 2, figsize=(24, 7))

if len(phase_df) > 0:
    ax = axes[0]; ax.axis('off')
    tbl = [['Lag', f'HFD Exp (N={n_exp})', f'HFD For (N={n_for})', 'p-value', "Cohen's d"]]
    for _, r in phase_df.iterrows():
        tbl.append([f"{int(r['Lag_ms'])}ms", f"{r['Exploration_Mean']:.6f} +/- {r['Exploration_SEM']:.6f}",
                     f"{r['Foraging_Mean']:.6f} +/- {r['Foraging_SEM']:.6f}",
                     fmt_p(r['MannWhitney_Pvalue']), f"{r['Cohens_d']:.3f}"])
    t = ax.table(cellText=tbl, cellLoc='center', loc='center', colWidths=[0.07,0.25,0.25,0.25,0.12])
    t.auto_set_font_size(False); t.set_fontsize(10); t.scale(1,2.2)
    for i in range(5): t[(0,i)].set_facecolor('#9b59b6'); t[(0,i)].set_text_props(weight='bold',color='white')
    for i in range(1,len(tbl)):
        for j in range(5): t[(i,j)].set_facecolor('#ecf0f1' if i%2==0 else 'white')
    ax.set_title('HFD LHA-ACA: Exp vs For (Session-Level)', fontsize=12, fontweight='bold', pad=20)
else:
    axes[0].axis('off')
    axes[0].text(0.5, 0.5, 'No phase data available', ha='center', va='center')

if state_comparison_results:
    ax = axes[1]; ax.axis('off')
    tbl = [['Lag', f'Fed (N={n_fed})', f'Fasted (N={n_fas})', f'HFD (N={n_hfd})', 'KW p-value']]
    for row in state_comparison_results:
        tbl.append([f"{int(row['Lag_ms'])}ms",
                     f"{row['Fed_Mean']:.6f} +/- {row['Fed_SEM']:.6f}",
                     f"{row['Fasted_Mean']:.6f} +/- {row['Fasted_SEM']:.6f}",
                     f"{row['HFD_Mean']:.6f} +/- {row['HFD_SEM']:.6f}",
                     fmt_p(row['KW_Pvalue'])])
    t = ax.table(cellText=tbl, cellLoc='center', loc='center', colWidths=[0.06,0.22,0.22,0.22,0.18])
    t.auto_set_font_size(False); t.set_fontsize(9); t.scale(1,2.2)
    for i in range(5): t[(0,i)].set_facecolor('#8e44ad'); t[(0,i)].set_text_props(weight='bold',color='white')
    for i in range(1,len(tbl)):
        for j in range(5): t[(i,j)].set_facecolor('#ecf0f1' if i%2==0 else 'white')
    ax.set_title('LHA-ACA: Fed vs Fasted vs HFD (Session-Level)', fontsize=12, fontweight='bold', pad=20)
else:
    axes[1].axis('off')
    axes[1].text(0.5, 0.5, 'No 3-way comparison data', ha='center', va='center')

plt.tight_layout()
plt.savefig("figures/hfd_lha_aca_good_summary_tables.png", dpi=150, bbox_inches='tight')
print("[OK] Saved hfd_lha_aca_good_summary_tables.png")
plt.close()

print(f"\n[DONE] HFD LHA-ACA cross-probe session-level analysis complete!")
