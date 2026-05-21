"""
HFD Probe-0 (ACA) Analysis -- Good Units Only
Fed-HFD sessions: 17-22
Compares: HFD Exploration vs Foraging (phase), and Fed vs Fasted vs HFD (state)
Loads existing Fed/Fasted session means for cross-state comparison.
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


LAGS_MS = [2, 5, 10, 50, 100]
BIN_SIZE_MS = 1
FS = 30000
BIN_SAMPLES = int(BIN_SIZE_MS * FS / 1000)
LAG_BINS = {lag: int(lag * FS / (1000 * BIN_SAMPLES)) for lag in LAGS_MS}


def prebin_spike_trains(sorting, unit_ids):
    """Bin and z-score all spike trains at once."""
    all_max = 0
    all_min = np.inf
    spike_trains = {}
    for uid in unit_ids:
        st = sorting.get_unit_spike_train(uid)
        spike_trains[uid] = st
        if len(st) > 0:
            all_max = max(all_max, np.max(st))
            all_min = min(all_min, np.min(st))

    n_bins = int((all_max - all_min) / BIN_SAMPLES) + 1
    binned = {}
    for uid in unit_ids:
        st = spike_trains[uid]
        t = np.zeros(n_bins)
        if len(st) > 0:
            b = ((st - all_min) // BIN_SAMPLES).astype(int)
            b = b[b < n_bins]
            np.add.at(t, b, 1)
        std_val = np.std(t)
        if std_val > 1e-8:
            t = (t - np.mean(t)) / std_val
        else:
            t = t - np.mean(t)
        binned[uid] = t
    return binned, n_bins


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
print("HFD PROBE-0 (ACA) ANALYSIS -- GOOD UNITS ONLY")
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
        sorted_path = p0.get("sorted") if p0 else None

        if sorted_path is None or sorted_path == '':
            print(f"  Session {session_num}: [SKIP] No sorted path")
            continue

        sorted_path_obj = Path(sorted_path)
        if not sorted_path_obj.exists():
            print(f"  Session {session_num}: [SKIP] Path does not exist: {sorted_path}")
            continue

        good_ids = get_good_unit_ids(sorted_path_obj)
        print(f"  Session {session_num}: {len(good_ids)} good units")

        if len(good_ids) < 2:
            print(f"    [SKIP] < 2 good units")
            continue

        try:
            sorting = se.read_kilosort(sorted_path_obj)
        except Exception as e:
            print(f"    [ERROR] {e}")
            continue

        available = set(sorting.get_unit_ids())
        good_ids = np.array([u for u in good_ids if u in available])
        print(f"    {len(good_ids)} good units available in sorting")

        if len(good_ids) < 2:
            continue

        t0 = time.time()
        print(f"    Pre-binning spike trains (1ms bins)...")
        binned, n_bins = prebin_spike_trains(sorting, good_ids)
        print(f"    Binned {len(good_ids)} units into {n_bins} bins ({n_bins/1000:.1f}s recording) in {time.time()-t0:.1f}s")

        total_pairs = len(good_ids) * (len(good_ids) - 1) // 2
        pair_count = 0
        t1_time = time.time()

        for i in range(len(good_ids)):
            t1_z = binned[good_ids[i]]
            for j in range(i+1, len(good_ids)):
                t2_z = binned[good_ids[j]]
                cc = cross_corr_fast(t1_z, t2_z, n_bins)
                group_pairs.append({
                    'unit_1': good_ids[i], 'unit_2': good_ids[j],
                    'session': session_num,
                    'correlation_2ms': cc[2], 'correlation_5ms': cc[5],
                    'correlation_10ms': cc[10], 'correlation_50ms': cc[50], 'correlation_100ms': cc[100]
                })
                pair_count += 1
                if pair_count % 5000 == 0:
                    elapsed = time.time() - t1_time
                    rate = pair_count / elapsed
                    remaining = (total_pairs - pair_count) / rate if rate > 0 else 0
                    print(f"      {pair_count}/{total_pairs} ({pair_count/total_pairs*100:.1f}%) - {remaining:.0f}s left")

        elapsed = time.time() - t0
        print(f"    Session {session_num}: {pair_count} pairs in {elapsed:.1f}s")

    if group_pairs:
        all_connectivity[group_name] = pd.DataFrame(group_pairs)
        print(f"\n  {group_name}: {len(group_pairs)} total pairs")
        out_file = f"data/hfd_aca_good_connectivity_{group_name}.csv"
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
    print("[ERROR] No data to analyze")
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

session_means.to_csv("data/hfd_aca_good_session_means.csv", index=False)

# --- PHASE COMPARISON (within HFD: Exploration vs Foraging) ---
print(f"\n--- PHASE COMPARISON (HFD Exploration vs Foraging, session-level) ---")
phase_results = []
for lag in LAGS_MS:
    col = f'correlation_{lag}ms'
    exp_vals = session_means[session_means['phase'] == 'exploration'][col].values
    for_vals = session_means[session_means['phase'] == 'foraging'][col].values
    if len(exp_vals) < 2 or len(for_vals) < 2:
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
    phase_df[save_cols].to_csv("data/hfd_aca_good_stats_phase_comparison.csv", index=False)


# =============================================================================
# STEP 3: 3-WAY STATE COMPARISON (Fed vs Fasted vs HFD)
# =============================================================================

print(f"\n{'='*70}")
print("3-WAY STATE COMPARISON: Fed vs Fasted vs HFD (session-level)")
print("=" * 70)

# Load existing Fed/Fasted session means
existing_means_path = Path("data/double_probe_aca_good_session_means.csv")
state_comparison_results = []

if existing_means_path.exists():
    existing_means = pd.read_csv(existing_means_path)
    fed_means = existing_means[existing_means['state'] == 'fed']
    fasted_means = existing_means[existing_means['state'] == 'fasted']
    hfd_means = session_means.copy()

    n_fed = len(fed_means)
    n_fas = len(fasted_means)

    print(f"  Fed sessions: {n_fed}, Fasted sessions: {len(fasted_means)}, HFD sessions: {n_hfd}")

    for lag in LAGS_MS:
        col = f'correlation_{lag}ms'
        fed_v = fed_means[col].values
        fas_v = fasted_means[col].values
        hfd_v = hfd_means[col].values

        print(f"\n  Lag {lag}ms:")

        # Kruskal-Wallis (3-way)
        if len(fed_v) >= 2 and len(fas_v) >= 2 and len(hfd_v) >= 2:
            kw_s, kw_p = sp_stats.kruskal(fed_v, fas_v, hfd_v)
            print(f"    Kruskal-Wallis: H={kw_s:.3f}, {fmt_p(kw_p)}")
        else:
            kw_s, kw_p = np.nan, np.nan

        # Pairwise Mann-Whitney
        comparisons = [
            ('Fed vs HFD', fed_v, hfd_v, n_fed, n_hfd),
            ('Fasted vs HFD', fas_v, hfd_v, len(fasted_means), n_hfd),
            ('Fed vs Fasted', fed_v, fas_v, n_fed, len(fasted_means))
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

        # Store means
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
    state_3way_df[save_cols].to_csv("data/hfd_aca_good_stats_3way_state_comparison.csv", index=False)
else:
    print("  [WARN] No existing session means found at data/double_probe_aca_good_session_means.csv")
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

# Figure 1: Phase comparison (HFD Exploration vs Foraging)
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
    ax.set_title('HFD ACA-ACA (Good Units): Exploration vs Foraging (Session-Level)', fontsize=14, fontweight='bold')
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
    plt.savefig("figures/hfd_aca_good_phase_comparison.png", dpi=150, bbox_inches='tight')
    print("[OK] Saved hfd_aca_good_phase_comparison.png")
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

    # Individual data points
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
    ax.set_title('ACA-ACA (Good Units): Fed vs Fasted vs HFD (Session-Level)', fontsize=14, fontweight='bold')
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
    plt.savefig("figures/hfd_aca_good_3way_state_comparison.png", dpi=150, bbox_inches='tight')
    print("[OK] Saved hfd_aca_good_3way_state_comparison.png")
    plt.close()

# Figure 3: Summary tables
fig, axes = plt.subplots(1, 2, figsize=(24, 7))

# Phase table
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
    ax.set_title('HFD ACA-ACA: Exp vs For (Session-Level)', fontsize=12, fontweight='bold', pad=20)
else:
    axes[0].axis('off')
    axes[0].text(0.5, 0.5, 'No phase data', ha='center', va='center')

# 3-way state table
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
    ax.set_title('ACA-ACA: Fed vs Fasted vs HFD (Session-Level)', fontsize=12, fontweight='bold', pad=20)
else:
    axes[1].axis('off')
    axes[1].text(0.5, 0.5, 'No 3-way comparison data', ha='center', va='center')

plt.tight_layout()
plt.savefig("figures/hfd_aca_good_summary_tables.png", dpi=150, bbox_inches='tight')
print("[OK] Saved hfd_aca_good_summary_tables.png")
plt.close()

print(f"\n[DONE] HFD Probe-0 ACA good-units analysis complete!")
