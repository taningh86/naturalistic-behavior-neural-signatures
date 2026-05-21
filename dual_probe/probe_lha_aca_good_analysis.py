"""
LHA-ACA Cross-Probe Analysis — Good Units Only
LHA from Probe-1 (depth 0-345 µm) × ACA from Probe-0.
1ms bins, lags at 2, 5, 10, 50, 100ms.
Fed sessions: 1, 3-10 | Fasted sessions: 11-16
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
LHA_DEPTH_MAX = 345  # µm

LAGS_MS = [2, 5, 10, 50, 100]
BIN_SIZE_MS = 1
FS = 30000
BIN_SAMPLES = int(BIN_SIZE_MS * FS / 1000)  # 30 samples per 1ms bin
LAG_BINS = {lag: int(lag * FS / (1000 * BIN_SAMPLES)) for lag in LAGS_MS}


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
    """Get unit IDs labeled 'good' AND in LHA depth range (0-345 µm)."""
    ci = sorted_path_obj / "cluster_info.tsv"
    if not ci.exists():
        print(f"    [WARN] No cluster_info.tsv at {sorted_path_obj}")
        return np.array([])

    df = pd.read_csv(ci, sep='\t')
    if 'depth' not in df.columns:
        return np.array([])

    label_col = None
    if 'group' in df.columns and df['group'].eq('good').any():
        label_col = 'group'
    elif 'KSLabel' in df.columns:
        label_col = 'KSLabel'
    if label_col is None:
        return np.array([])

    good_lha = df[(df[label_col] == 'good') &
                  (df['depth'] >= LHA_DEPTH_MIN) &
                  (df['depth'] <= LHA_DEPTH_MAX)]
    return good_lha['cluster_id'].values


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

def compute_ci_95(data):
    m = np.mean(data)
    sem = sp_stats.sem(data)
    ci = sem * sp_stats.t.ppf(0.975, len(data)-1)
    return m, m-ci, m+ci

def fmt_p(p):
    if p < 0.0001: return "p < 0.0001***"
    elif p < 0.001: return f"p = {p:.2e}***"
    elif p < 0.01: return f"p = {p:.4f}**"
    elif p < 0.05: return f"p = {p:.4f}*"
    else: return f"p = {p:.4f} ns"


# =============================================================================
# SESSION GROUPS
# =============================================================================

session_groups = {
    'fed_exploration':    [('session_1', 1), ('session_3', 3), ('session_5', 5), ('session_7', 7), ('session_9', 9)],
    'fed_foraging':       [('session_4', 4), ('session_6', 6), ('session_8', 8), ('session_10', 10)],
    'fasted_exploration': [('session_11', 11), ('session_13', 13), ('session_15', 15)],
    'fasted_foraging':    [('session_12', 12), ('session_14', 14), ('session_16', 16)]
}

print("=" * 70)
print("LHA-ACA CROSS-PROBE ANALYSIS — GOOD UNITS ONLY")
print("LHA from Probe-1 (0-345µm) × ACA from Probe-0")
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

        # Probe-0 (ACA)
        p0 = session_config.get("probe_0_aca", {})
        p0_sorted = p0.get("sorted") if p0 else None

        # Probe-1 (LHA)
        p1 = session_config.get("probe_1_lha_rsp", {})
        p1_sorted = p1.get("sorted") if p1 else None

        if p0_sorted is None or p1_sorted is None:
            print(f"  Session {session_num}: [SKIP] Missing sorted path (p0={p0_sorted is not None}, p1={p1_sorted is not None})")
            continue

        p0_path = Path(p0_sorted)
        p1_path = Path(p1_sorted)

        # Get good unit IDs
        aca_ids = get_good_unit_ids(p0_path)
        lha_ids = get_good_lha_unit_ids(p1_path)

        print(f"  Session {session_num}: ACA={len(aca_ids)} good, LHA={len(lha_ids)} good (0-345µm)")

        if len(aca_ids) < 1 or len(lha_ids) < 1:
            print(f"    [SKIP] Need at least 1 unit from each region")
            continue

        # Load both sortings
        try:
            sorting_aca = se.read_kilosort(p0_path)
            sorting_lha = se.read_kilosort(p1_path)
        except Exception as e:
            print(f"    [ERROR] {e}")
            continue

        # Filter to available units
        avail_aca = set(sorting_aca.get_unit_ids())
        avail_lha = set(sorting_lha.get_unit_ids())
        aca_ids = np.array([u for u in aca_ids if u in avail_aca])
        lha_ids = np.array([u for u in lha_ids if u in avail_lha])

        if len(aca_ids) < 1 or len(lha_ids) < 1:
            print(f"    [SKIP] After filtering: ACA={len(aca_ids)}, LHA={len(lha_ids)}")
            continue

        # Find shared time range across both probes
        t0 = time.time()
        print(f"    Pre-binning spike trains (1ms bins)...")
        aca_min, aca_max = get_global_time_range(sorting_aca, aca_ids)
        lha_min, lha_max = get_global_time_range(sorting_lha, lha_ids)
        global_min = min(aca_min, lha_min)
        global_max = max(aca_max, lha_max)

        # Pre-bin both probes on the same time axis
        binned_aca, n_bins = prebin_spike_trains(sorting_aca, aca_ids, global_min, global_max)
        binned_lha, _ = prebin_spike_trains(sorting_lha, lha_ids, global_min, global_max)
        print(f"    Binned ACA={len(aca_ids)}, LHA={len(lha_ids)} into {n_bins} bins ({n_bins/1000:.1f}s) in {time.time()-t0:.1f}s")

        # All LHA × ACA pairs (cross-region, not upper triangle)
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
        print(f"    Session {session_num}: {pair_count} LHA×ACA pairs in {elapsed:.1f}s")

    if group_pairs:
        all_connectivity[group_name] = pd.DataFrame(group_pairs)
        print(f"\n  {group_name}: {len(group_pairs)} total pairs")
        out_file = f"data/double_probe_lha_aca_good_connectivity_{group_name}.csv"
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

session_meta = {
    1: ('fed', 'exploration'), 3: ('fed', 'exploration'),
    4: ('fed', 'foraging'), 5: ('fed', 'exploration'),
    6: ('fed', 'foraging'), 7: ('fed', 'exploration'),
    8: ('fed', 'foraging'), 9: ('fed', 'exploration'),
    10: ('fed', 'foraging'),
    11: ('fasted', 'exploration'), 12: ('fasted', 'foraging'),
    13: ('fasted', 'exploration'), 14: ('fasted', 'foraging'),
    15: ('fasted', 'exploration'), 16: ('fasted', 'foraging'),
}

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

n_fed = len(session_means[session_means['state'] == 'fed'])
n_fas = len(session_means[session_means['state'] == 'fasted'])
n_exp = len(session_means[session_means['phase'] == 'exploration'])
n_for = len(session_means[session_means['phase'] == 'foraging'])
print(f"\nTotal pairs: {len(full_df)} across {len(session_means)} sessions")
print(f"  Fed sessions: {n_fed}, Fasted sessions: {n_fas}")
print(f"  Exploration sessions: {n_exp}, Foraging sessions: {n_for}")

session_means.to_csv("data/double_probe_lha_aca_good_session_means.csv", index=False)

# --- STATE COMPARISON (session-level) ---
print(f"\n--- STATE COMPARISON (Fed vs Fasted, session-level) ---")
state_results = []
for lag in LAGS_MS:
    col = f'correlation_{lag}ms'
    fed_vals = session_means[session_means['state'] == 'fed'][col].values
    fas_vals = session_means[session_means['state'] == 'fasted'][col].values
    if len(fed_vals) < 2 or len(fas_vals) < 2:
        continue
    mw_s, mw_p = sp_stats.mannwhitneyu(fed_vals, fas_vals, alternative='two-sided')
    d = compute_cohens_d(fed_vals, fas_vals)
    fm = np.mean(fed_vals); f_sem = sp_stats.sem(fed_vals)
    am = np.mean(fas_vals); a_sem = sp_stats.sem(fas_vals)
    fold = ((am - fm) / abs(fm)) * 100 if fm != 0 else 0
    state_results.append({
        'Lag_ms': lag, 'Fed_Mean': fm, 'Fed_SEM': f_sem, 'Fed_Sessions': fed_vals.tolist(),
        'Fasted_Mean': am, 'Fasted_SEM': a_sem, 'Fasted_Sessions': fas_vals.tolist(),
        'MannWhitney_Statistic': mw_s, 'MannWhitney_Pvalue': mw_p,
        'Cohens_d': d, 'N_Fed': len(fed_vals), 'N_Fasted': len(fas_vals)
    })
    print(f"  Lag {lag}ms: Fed={fm:.6f}+/-{f_sem:.6f} (N={len(fed_vals)}), "
          f"Fasted={am:.6f}+/-{a_sem:.6f} (N={len(fas_vals)}), "
          f"Change={fold:+.1f}%, {fmt_p(mw_p)}, d={d:.4f}")

state_df = pd.DataFrame(state_results)
if len(state_df) > 0:
    save_cols = ['Lag_ms', 'Fed_Mean', 'Fed_SEM', 'Fasted_Mean', 'Fasted_SEM',
                 'MannWhitney_Statistic', 'MannWhitney_Pvalue', 'Cohens_d', 'N_Fed', 'N_Fasted']
    state_df[save_cols].to_csv("data/double_probe_lha_aca_good_stats_state_comparison.csv", index=False)

# --- PHASE COMPARISON (session-level) ---
print(f"\n--- PHASE COMPARISON (Exploration vs Foraging, session-level) ---")
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
    phase_df[save_cols].to_csv("data/double_probe_lha_aca_good_stats_phase_comparison.csv", index=False)


# =============================================================================
# STEP 3: FIGURES (session-level with individual data points)
# =============================================================================

print(f"\n{'='*70}")
print("GENERATING FIGURES")
print("=" * 70)

w = 0.35

if len(state_df) > 0:
    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(state_df))
    fm_vals = state_df['Fed_Mean'].values
    f_sem_vals = state_df['Fed_SEM'].values
    am_vals = state_df['Fasted_Mean'].values
    a_sem_vals = state_df['Fasted_SEM'].values
    ax.bar(x-w/2, fm_vals, w, yerr=f_sem_vals, label=f'Fed (N={n_fed})', capsize=8, color='#3498db', alpha=0.7, error_kw={'linewidth': 2})
    ax.bar(x+w/2, am_vals, w, yerr=a_sem_vals, label=f'Fasted (N={n_fas})', capsize=8, color='#e74c3c', alpha=0.7, error_kw={'linewidth': 2})
    for i, row in state_df.iterrows():
        jitter = 0.06
        ax.scatter([i - w/2 + np.random.uniform(-jitter, jitter) for _ in row['Fed_Sessions']],
                   row['Fed_Sessions'], color='#2c3e50', s=40, zorder=5, edgecolors='white', linewidth=0.5)
        ax.scatter([i + w/2 + np.random.uniform(-jitter, jitter) for _ in row['Fasted_Sessions']],
                   row['Fasted_Sessions'], color='#2c3e50', s=40, zorder=5, edgecolors='white', linewidth=0.5)
    ax.set_xlabel('Time Lag (ms)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Session Mean Cross-Correlation +/- SEM', fontsize=12, fontweight='bold')
    ax.set_title('Double Probe LHA-ACA (Good Units): Fed vs Fasted (Session-Level)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(state_df['Lag_ms'].values.astype(int))
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    all_pts = np.concatenate([np.concatenate(state_df['Fed_Sessions'].values), np.concatenate(state_df['Fasted_Sessions'].values)])
    ymin, ymax = np.min(all_pts), np.max(all_pts)
    margin = (ymax - ymin) * 0.4
    ax.set_ylim(ymin - margin * 0.3, ymax + margin * 1.2)
    for i, row in state_df.iterrows():
        yp = max(max(row['Fed_Sessions']), max(row['Fasted_Sessions']))
        ax.text(i, yp + margin*0.25, fmt_p(row['MannWhitney_Pvalue']), ha='center', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', alpha=0.7))
        ax.text(i, yp + margin*0.55, f"d = {row['Cohens_d']:.3f}", ha='center', fontsize=8, style='italic')
    plt.tight_layout()
    plt.savefig("figures/double_probe_lha_aca_good_state_comparison.png", dpi=150, bbox_inches='tight')
    print("[OK] Saved double_probe_lha_aca_good_state_comparison.png")
    plt.close()

if len(phase_df) > 0:
    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(phase_df))
    em_vals = phase_df['Exploration_Mean'].values
    e_sem_vals = phase_df['Exploration_SEM'].values
    fm_vals = phase_df['Foraging_Mean'].values
    f_sem_vals = phase_df['Foraging_SEM'].values
    ax.bar(x-w/2, em_vals, w, yerr=e_sem_vals, label=f'Exploration (N={n_exp})', capsize=8, color='#2ecc71', alpha=0.7, error_kw={'linewidth': 2})
    ax.bar(x+w/2, fm_vals, w, yerr=f_sem_vals, label=f'Foraging (N={n_for})', capsize=8, color='#f39c12', alpha=0.7, error_kw={'linewidth': 2})
    for i, row in phase_df.iterrows():
        jitter = 0.06
        ax.scatter([i - w/2 + np.random.uniform(-jitter, jitter) for _ in row['Exploration_Sessions']],
                   row['Exploration_Sessions'], color='#2c3e50', s=40, zorder=5, edgecolors='white', linewidth=0.5)
        ax.scatter([i + w/2 + np.random.uniform(-jitter, jitter) for _ in row['Foraging_Sessions']],
                   row['Foraging_Sessions'], color='#2c3e50', s=40, zorder=5, edgecolors='white', linewidth=0.5)
    ax.set_xlabel('Time Lag (ms)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Session Mean Cross-Correlation +/- SEM', fontsize=12, fontweight='bold')
    ax.set_title('Double Probe LHA-ACA (Good Units): Exp vs For (Session-Level)', fontsize=14, fontweight='bold')
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
    plt.savefig("figures/double_probe_lha_aca_good_phase_comparison.png", dpi=150, bbox_inches='tight')
    print("[OK] Saved double_probe_lha_aca_good_phase_comparison.png")
    plt.close()

if len(state_df) > 0 and len(phase_df) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(22, 6))
    ax = axes[0]; ax.axis('off')
    tbl = [['Lag', f'Fed (N={n_fed})', f'Fasted (N={n_fas})', 'p-value', "Cohen's d"]]
    for _, r in state_df.iterrows():
        tbl.append([f"{int(r['Lag_ms'])}ms", f"{r['Fed_Mean']:.6f} +/- {r['Fed_SEM']:.6f}",
                     f"{r['Fasted_Mean']:.6f} +/- {r['Fasted_SEM']:.6f}",
                     fmt_p(r['MannWhitney_Pvalue']), f"{r['Cohens_d']:.3f}"])
    t = ax.table(cellText=tbl, cellLoc='center', loc='center', colWidths=[0.07,0.25,0.25,0.25,0.12])
    t.auto_set_font_size(False); t.set_fontsize(10); t.scale(1,2.2)
    for i in range(5): t[(0,i)].set_facecolor('#3498db'); t[(0,i)].set_text_props(weight='bold',color='white')
    for i in range(1,len(tbl)):
        for j in range(5): t[(i,j)].set_facecolor('#ecf0f1' if i%2==0 else 'white')
    ax.set_title('LHA-ACA (Good): Fed vs Fasted (Session-Level)', fontsize=12, fontweight='bold', pad=20)

    ax = axes[1]; ax.axis('off')
    tbl = [['Lag', f'Exploration (N={n_exp})', f'Foraging (N={n_for})', 'p-value', "Cohen's d"]]
    for _, r in phase_df.iterrows():
        tbl.append([f"{int(r['Lag_ms'])}ms", f"{r['Exploration_Mean']:.6f} +/- {r['Exploration_SEM']:.6f}",
                     f"{r['Foraging_Mean']:.6f} +/- {r['Foraging_SEM']:.6f}",
                     fmt_p(r['MannWhitney_Pvalue']), f"{r['Cohens_d']:.3f}"])
    t = ax.table(cellText=tbl, cellLoc='center', loc='center', colWidths=[0.07,0.25,0.25,0.25,0.12])
    t.auto_set_font_size(False); t.set_fontsize(10); t.scale(1,2.2)
    for i in range(5): t[(0,i)].set_facecolor('#2ecc71'); t[(0,i)].set_text_props(weight='bold',color='white')
    for i in range(1,len(tbl)):
        for j in range(5): t[(i,j)].set_facecolor('#ecf0f1' if i%2==0 else 'white')
    ax.set_title('LHA-ACA (Good): Exp vs For (Session-Level)', fontsize=12, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig("figures/double_probe_lha_aca_good_summary_tables.png", dpi=150, bbox_inches='tight')
    print("[OK] Saved double_probe_lha_aca_good_summary_tables.png")
    plt.close()

print(f"\n[DONE] LHA-ACA cross-probe session-level analysis complete!")
