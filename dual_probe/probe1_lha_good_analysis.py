"""
Probe-1 LHA-LHA Full Analysis — Good Units Only, Depth 0-345 µm
Uses 'good' label from cluster_info.tsv AND depth filter for LHA.
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


def get_good_lha_unit_ids(sorted_path_obj):
    """Get unit IDs labeled 'good' AND in LHA depth range (0-345 µm)."""
    ci = sorted_path_obj / "cluster_info.tsv"
    if not ci.exists():
        print(f"    [WARN] No cluster_info.tsv at {sorted_path_obj}")
        return np.array([])

    df = pd.read_csv(ci, sep='\t')

    if 'depth' not in df.columns:
        print(f"    [WARN] No 'depth' column in cluster_info.tsv")
        return np.array([])

    # Determine label column: prefer Phy-curated 'group' if populated, else KSLabel
    label_col = None
    if 'group' in df.columns and df['group'].eq('good').any():
        label_col = 'group'
    elif 'KSLabel' in df.columns:
        label_col = 'KSLabel'

    if label_col is None:
        print(f"    [WARN] No label column found")
        return np.array([])

    good_lha = df[(df[label_col] == 'good') &
                  (df['depth'] >= LHA_DEPTH_MIN) &
                  (df['depth'] <= LHA_DEPTH_MAX)]

    return good_lha['cluster_id'].values


LAGS_MS = [2, 5, 10, 50, 100]
BIN_SIZE_MS = 1
FS = 30000
BIN_SAMPLES = int(BIN_SIZE_MS * FS / 1000)  # 30 samples per 1ms bin

LAG_BINS = {lag: int(lag * FS / (1000 * BIN_SAMPLES)) for lag in LAGS_MS}


def prebin_spike_trains(sorting, unit_ids):
    """Bin and z-score all spike trains at once. Returns dict of z-scored arrays."""
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
    """Compute cross-correlation at all lags using fast dot products on z-scored arrays."""
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
print("PROBE-1 LHA-LHA FULL ANALYSIS — GOOD UNITS ONLY (0-345 µm)")
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
        p1 = session_config.get("probe_1_lha_rsp", {})
        sorted_path = p1.get("sorted") if p1 else None

        if sorted_path is None:
            print(f"  Session {session_num}: [SKIP] No sorted path")
            continue

        sorted_path_obj = Path(sorted_path)
        good_lha_ids = get_good_lha_unit_ids(sorted_path_obj)
        print(f"  Session {session_num}: {len(good_lha_ids)} good LHA units (0-345 µm)")

        if len(good_lha_ids) < 2:
            print(f"    [SKIP] < 2 good LHA units")
            continue

        try:
            sorting = se.read_kilosort(sorted_path_obj)
        except Exception as e:
            print(f"    [ERROR] {e}")
            continue

        # Filter to units that exist in sorting
        available = set(sorting.get_unit_ids())
        good_lha_ids = np.array([u for u in good_lha_ids if u in available])
        print(f"    {len(good_lha_ids)} good LHA units available in sorting")

        if len(good_lha_ids) < 2:
            continue

        # Pre-bin all spike trains for this session
        t0 = time.time()
        print(f"    Pre-binning spike trains (1ms bins)...")
        binned, n_bins = prebin_spike_trains(sorting, good_lha_ids)
        print(f"    Binned {len(good_lha_ids)} units into {n_bins} bins ({n_bins/1000:.1f}s recording) in {time.time()-t0:.1f}s")

        # Pairwise cross-correlations (upper triangle) using fast dot products
        total_pairs = len(good_lha_ids) * (len(good_lha_ids) - 1) // 2
        pair_count = 0
        t1_time = time.time()

        for i in range(len(good_lha_ids)):
            t1_z = binned[good_lha_ids[i]]
            for j in range(i+1, len(good_lha_ids)):
                t2_z = binned[good_lha_ids[j]]
                cc = cross_corr_fast(t1_z, t2_z, n_bins)
                group_pairs.append({
                    'unit_1': good_lha_ids[i], 'unit_2': good_lha_ids[j],
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
        out_file = f"data/lha_good_connectivity_{group_name}.csv"
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

lag_cols = [f'correlation_{lag}ms' for lag in [2, 5, 10, 50, 100]]
session_means = full_df.groupby('session')[lag_cols].mean().reset_index()
session_means['state'] = session_means['session'].map(lambda s: session_meta[s][0])
session_means['phase'] = session_means['session'].map(lambda s: session_meta[s][1])
session_means['n_pairs'] = full_df.groupby('session').size().values

print(f"\nPer-session mean correlations (N pairs in parentheses):")
for _, row in session_means.iterrows():
    s = int(row['session'])
    print(f"  Session {s} [{row['state']}/{row['phase']}] (n={int(row['n_pairs'])}): " +
          ", ".join([f"{lag}ms={row[f'correlation_{lag}ms']:.6f}" for lag in [2, 5, 10, 50, 100]]))

n_fed = len(session_means[session_means['state'] == 'fed'])
n_fas = len(session_means[session_means['state'] == 'fasted'])
n_exp = len(session_means[session_means['phase'] == 'exploration'])
n_for = len(session_means[session_means['phase'] == 'foraging'])
print(f"\nTotal pairs: {len(full_df)} across {len(session_means)} sessions")
print(f"  Fed sessions: {n_fed}, Fasted sessions: {n_fas}")
print(f"  Exploration sessions: {n_exp}, Foraging sessions: {n_for}")

session_means.to_csv("data/double_probe_lha_good_session_means.csv", index=False)

# --- STATE COMPARISON (session-level) ---
print(f"\n--- STATE COMPARISON (Fed vs Fasted, session-level) ---")
state_results = []
for lag in [2, 5, 10, 50, 100]:
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
    state_df[save_cols].to_csv("data/double_probe_lha_good_stats_state_comparison.csv", index=False)

# --- PHASE COMPARISON (session-level) ---
print(f"\n--- PHASE COMPARISON (Exploration vs Foraging, session-level) ---")
phase_results = []
for lag in [2, 5, 10, 50, 100]:
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
    phase_df[save_cols].to_csv("data/double_probe_lha_good_stats_phase_comparison.csv", index=False)


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
    ax.set_title('Double Probe LHA-LHA (Good Units 0-345um): Fed vs Fasted (Session-Level)', fontsize=14, fontweight='bold')
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
    plt.savefig("figures/double_probe_lha_good_state_comparison.png", dpi=150, bbox_inches='tight')
    print("[OK] Saved double_probe_lha_good_state_comparison.png")
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
    ax.set_title('Double Probe LHA-LHA (Good Units 0-345um): Exp vs For (Session-Level)', fontsize=14, fontweight='bold')
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
    plt.savefig("figures/double_probe_lha_good_phase_comparison.png", dpi=150, bbox_inches='tight')
    print("[OK] Saved double_probe_lha_good_phase_comparison.png")
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
    ax.set_title('LHA-LHA (Good 0-345um): Fed vs Fasted (Session-Level)', fontsize=12, fontweight='bold', pad=20)

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
    ax.set_title('LHA-LHA (Good 0-345um): Exp vs For (Session-Level)', fontsize=12, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig("figures/double_probe_lha_good_summary_tables.png", dpi=150, bbox_inches='tight')
    print("[OK] Saved double_probe_lha_good_summary_tables.png")
    plt.close()

print(f"\n[DONE] Probe-1 LHA good-units session-level analysis complete!")
