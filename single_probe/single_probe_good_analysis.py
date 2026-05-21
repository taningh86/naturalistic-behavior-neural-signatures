"""
Single-Probe Mouse01 Coordinates-1 Analysis — Good Units Only
LHA (<1300 µm) and RSP (>=1300 µm) from cluster_info.tsv.
Three network types: LHA-LHA, RSP-RSP, LHA-RSP.
1ms bins, lags at 2, 5, 10, 50, 100ms.
Fed sessions: 1-4 | Fasted sessions: 5-8
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

LHA_DEPTH_MAX = 1300   # µm — LHA is below this
RSP_DEPTH_MIN = 1300   # µm — RSP is at or above this

LAGS_MS = [2, 5, 10, 50, 100]
BIN_SIZE_MS = 1
FS = 30000
BIN_SAMPLES = int(BIN_SIZE_MS * FS / 1000)  # 30 samples per 1ms bin
LAG_BINS = {lag: int(lag * FS / (1000 * BIN_SAMPLES)) for lag in LAGS_MS}


def get_good_units_by_region(sorted_path_obj):
    """Get good LHA and RSP unit IDs from cluster_info.tsv, split by depth."""
    ci = sorted_path_obj / "cluster_info.tsv"
    if not ci.exists():
        print(f"    [WARN] No cluster_info.tsv at {sorted_path_obj}")
        return np.array([]), np.array([])

    df = pd.read_csv(ci, sep='\t')
    if 'depth' not in df.columns:
        print(f"    [WARN] No depth column in cluster_info.tsv")
        return np.array([]), np.array([])

    # Determine label column
    label_col = None
    if 'group' in df.columns and df['group'].eq('good').any():
        label_col = 'group'
    elif 'KSLabel' in df.columns:
        label_col = 'KSLabel'
    if label_col is None:
        print(f"    [WARN] No label column found")
        return np.array([]), np.array([])

    good = df[df[label_col] == 'good']
    lha_ids = good[good['depth'] < LHA_DEPTH_MAX]['cluster_id'].values
    rsp_ids = good[good['depth'] >= RSP_DEPTH_MIN]['cluster_id'].values
    return lha_ids, rsp_ids


def prebin_spike_trains(sorting, unit_ids, global_min=None, global_max=None):
    """Bin and z-score spike trains. If global_min/max not provided, compute from data."""
    spike_trains = {}
    for uid in unit_ids:
        st = sorting.get_unit_spike_train(uid)
        spike_trains[uid] = st

    if global_min is None or global_max is None:
        all_min, all_max = np.inf, 0
        for uid in unit_ids:
            st = spike_trains[uid]
            if len(st) > 0:
                all_min = min(all_min, np.min(st))
                all_max = max(all_max, np.max(st))
        global_min = all_min
        global_max = all_max

    n_bins = int((global_max - global_min) / BIN_SAMPLES) + 1
    binned = {}
    for uid in unit_ids:
        st = spike_trains[uid]
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
    return binned, n_bins, global_min, global_max


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

sessions = paths_config["single_probe"]["coordinates_1"]["mouse01"]["sessions"]

session_groups = {
    'fed_exploration':    [('session_1', 1), ('session_3', 3)],
    'fed_foraging':       [('session_2', 2), ('session_4', 4)],
    'fasted_exploration': [('session_5', 5), ('session_7', 7)],
    'fasted_foraging':    [('session_6', 6), ('session_8', 8)]
}

# Three network types to analyze
NETWORK_TYPES = ['LHA-LHA', 'RSP-RSP', 'LHA-RSP']


# =============================================================================
# STEP 1: CROSS-CORRELATIONS
# =============================================================================

# Storage: {network_type: {group_name: DataFrame}}
all_connectivity = {nt: {} for nt in NETWORK_TYPES}
overall_start = time.time()

for group_name, session_list in session_groups.items():
    print(f"\n{'='*70}")
    print(f"GROUP: {group_name.upper()}")
    print(f"{'='*70}")

    # Accumulate pairs per network type per group
    group_pairs = {nt: [] for nt in NETWORK_TYPES}

    for session_name, session_num in session_list:
        session_config = sessions[session_name]
        sorted_path = session_config.get("sorted")

        if sorted_path is None:
            print(f"  Session {session_num}: [SKIP] No sorted path")
            continue

        sp = Path(sorted_path)

        # Get good units split by region
        lha_ids, rsp_ids = get_good_units_by_region(sp)
        print(f"  Session {session_num}: LHA={len(lha_ids)} good (<{LHA_DEPTH_MAX}µm), RSP={len(rsp_ids)} good (>={RSP_DEPTH_MIN}µm)")

        if len(lha_ids) < 2 and len(rsp_ids) < 2:
            print(f"    [SKIP] Not enough units in either region")
            continue

        # Load sorting
        try:
            sorting = se.read_kilosort(sp)
        except Exception as e:
            print(f"    [ERROR] {e}")
            continue

        # Filter to available units in sorting
        avail = set(sorting.get_unit_ids())
        lha_ids = np.array([u for u in lha_ids if u in avail])
        rsp_ids = np.array([u for u in rsp_ids if u in avail])

        all_unit_ids = np.concatenate([lha_ids, rsp_ids])
        if len(all_unit_ids) < 2:
            print(f"    [SKIP] < 2 total units after filtering")
            continue

        # Pre-bin all units together (shared time axis)
        t0 = time.time()
        print(f"    Pre-binning {len(all_unit_ids)} spike trains (1ms bins)...")
        binned, n_bins, gmin, gmax = prebin_spike_trains(sorting, all_unit_ids)
        print(f"    Binned into {n_bins} bins ({n_bins/1000:.1f}s) in {time.time()-t0:.1f}s")

        # --- LHA-LHA pairs ---
        if len(lha_ids) >= 2:
            t1 = time.time()
            n_lha_pairs = len(lha_ids) * (len(lha_ids) - 1) // 2
            pair_count = 0
            for i in range(len(lha_ids)):
                for j in range(i + 1, len(lha_ids)):
                    cc = cross_corr_fast(binned[lha_ids[i]], binned[lha_ids[j]], n_bins)
                    group_pairs['LHA-LHA'].append({
                        'unit_1': lha_ids[i], 'unit_2': lha_ids[j],
                        'session': session_num,
                        'correlation_2ms': cc[2], 'correlation_5ms': cc[5],
                        'correlation_10ms': cc[10], 'correlation_50ms': cc[50], 'correlation_100ms': cc[100]
                    })
                    pair_count += 1
                    if pair_count % 5000 == 0:
                        elapsed = time.time() - t1
                        rate = pair_count / elapsed if elapsed > 0 else 1
                        remaining = (n_lha_pairs - pair_count) / rate
                        print(f"      LHA-LHA: {pair_count}/{n_lha_pairs} ({pair_count/n_lha_pairs*100:.1f}%) - {remaining:.0f}s left")
            print(f"    LHA-LHA: {pair_count} pairs in {time.time()-t1:.1f}s")

        # --- RSP-RSP pairs ---
        if len(rsp_ids) >= 2:
            t1 = time.time()
            n_rsp_pairs = len(rsp_ids) * (len(rsp_ids) - 1) // 2
            pair_count = 0
            for i in range(len(rsp_ids)):
                for j in range(i + 1, len(rsp_ids)):
                    cc = cross_corr_fast(binned[rsp_ids[i]], binned[rsp_ids[j]], n_bins)
                    group_pairs['RSP-RSP'].append({
                        'unit_1': rsp_ids[i], 'unit_2': rsp_ids[j],
                        'session': session_num,
                        'correlation_2ms': cc[2], 'correlation_5ms': cc[5],
                        'correlation_10ms': cc[10], 'correlation_50ms': cc[50], 'correlation_100ms': cc[100]
                    })
                    pair_count += 1
                    if pair_count % 5000 == 0:
                        elapsed = time.time() - t1
                        rate = pair_count / elapsed if elapsed > 0 else 1
                        remaining = (n_rsp_pairs - pair_count) / rate
                        print(f"      RSP-RSP: {pair_count}/{n_rsp_pairs} ({pair_count/n_rsp_pairs*100:.1f}%) - {remaining:.0f}s left")
            print(f"    RSP-RSP: {pair_count} pairs in {time.time()-t1:.1f}s")

        # --- LHA-RSP pairs (cross-region: all LHA × RSP) ---
        if len(lha_ids) >= 1 and len(rsp_ids) >= 1:
            t1 = time.time()
            n_cross_pairs = len(lha_ids) * len(rsp_ids)
            pair_count = 0
            for lha_uid in lha_ids:
                for rsp_uid in rsp_ids:
                    cc = cross_corr_fast(binned[lha_uid], binned[rsp_uid], n_bins)
                    group_pairs['LHA-RSP'].append({
                        'lha_unit': lha_uid, 'rsp_unit': rsp_uid,
                        'session': session_num,
                        'correlation_2ms': cc[2], 'correlation_5ms': cc[5],
                        'correlation_10ms': cc[10], 'correlation_50ms': cc[50], 'correlation_100ms': cc[100]
                    })
                    pair_count += 1
                    if pair_count % 5000 == 0:
                        elapsed = time.time() - t1
                        rate = pair_count / elapsed if elapsed > 0 else 1
                        remaining = (n_cross_pairs - pair_count) / rate
                        print(f"      LHA-RSP: {pair_count}/{n_cross_pairs} ({pair_count/n_cross_pairs*100:.1f}%) - {remaining:.0f}s left")
            print(f"    LHA-RSP: {pair_count} pairs in {time.time()-t1:.1f}s")

    # Save per-group CSVs for each network type
    for nt in NETWORK_TYPES:
        if group_pairs[nt]:
            df = pd.DataFrame(group_pairs[nt])
            all_connectivity[nt][group_name] = df
            nt_label = nt.lower().replace('-', '_')
            out_file = f"data/single_probe_{nt_label}_good_connectivity_{group_name}.csv"
            df.to_csv(out_file, index=False)
            print(f"  [OK] {nt}: {len(df)} pairs -> {out_file}")

total_time = time.time() - overall_start
print(f"\n[CROSS-CORRELATIONS DONE] {total_time:.1f}s ({total_time/60:.1f} min)")


# =============================================================================
# STEP 2: SESSION-LEVEL STATISTICAL ANALYSIS
# =============================================================================
# For each session, compute mean correlation at each lag.
# Then compare session-level means across conditions.
# Single-probe: N=4 fed vs N=4 fasted, N=4 exploration vs N=4 foraging.

# Map sessions to state/phase
session_meta = {
    1: ('fed', 'exploration'), 2: ('fed', 'foraging'),
    3: ('fed', 'exploration'), 4: ('fed', 'foraging'),
    5: ('fasted', 'exploration'), 6: ('fasted', 'foraging'),
    7: ('fasted', 'exploration'), 8: ('fasted', 'foraging'),
}

for nt in NETWORK_TYPES:
    nt_label = nt.lower().replace('-', '_')
    print(f"\n{'='*70}")
    print(f"SESSION-LEVEL STATISTICAL ANALYSIS: {nt}")
    print("=" * 70)

    # Collect all pairs across groups
    all_dfs = []
    for group_name, df in all_connectivity[nt].items():
        all_dfs.append(df)

    if not all_dfs:
        print(f"  [SKIP] No data for {nt}")
        continue

    full_df = pd.concat(all_dfs, ignore_index=True)

    # Compute per-session means
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
    print(f"\nTotal {nt} pairs: {len(full_df)} across {len(session_means)} sessions")
    print(f"  Fed sessions: {n_fed}, Fasted sessions: {n_fas}")
    print(f"  Exploration sessions: {n_exp}, Foraging sessions: {n_for}")

    # Save session-level means
    session_means.to_csv(f"data/single_probe_{nt_label}_good_session_means.csv", index=False)

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
        fm = np.mean(fed_vals)
        f_sem = sp_stats.sem(fed_vals)
        am = np.mean(fas_vals)
        a_sem = sp_stats.sem(fas_vals)
        fold = ((am - fm) / abs(fm)) * 100 if fm != 0 else 0
        state_results.append({
            'Lag_ms': lag,
            'Fed_Mean': fm, 'Fed_SEM': f_sem, 'Fed_Sessions': fed_vals.tolist(),
            'Fasted_Mean': am, 'Fasted_SEM': a_sem, 'Fasted_Sessions': fas_vals.tolist(),
            'MannWhitney_Statistic': mw_s, 'MannWhitney_Pvalue': mw_p,
            'Cohens_d': d, 'N_Fed': len(fed_vals), 'N_Fasted': len(fas_vals)
        })
        print(f"  Lag {lag}ms: Fed={fm:.6f}+/-{f_sem:.6f} (N={len(fed_vals)}), "
              f"Fasted={am:.6f}+/-{a_sem:.6f} (N={len(fas_vals)}), "
              f"Change={fold:+.1f}%, {fmt_p(mw_p)}, d={d:.4f}")

    state_df = pd.DataFrame(state_results)
    if len(state_df) > 0:
        # Save a clean version (without list columns)
        save_cols = ['Lag_ms', 'Fed_Mean', 'Fed_SEM', 'Fasted_Mean', 'Fasted_SEM',
                     'MannWhitney_Statistic', 'MannWhitney_Pvalue', 'Cohens_d', 'N_Fed', 'N_Fasted']
        state_df[save_cols].to_csv(f"data/single_probe_{nt_label}_good_stats_state_comparison.csv", index=False)

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
        em = np.mean(exp_vals)
        e_sem = sp_stats.sem(exp_vals)
        fmm = np.mean(for_vals)
        f_sem = sp_stats.sem(for_vals)
        fold = ((fmm - em) / abs(em)) * 100 if em != 0 else 0
        phase_results.append({
            'Lag_ms': lag,
            'Exploration_Mean': em, 'Exploration_SEM': e_sem, 'Exploration_Sessions': exp_vals.tolist(),
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
        phase_df[save_cols].to_csv(f"data/single_probe_{nt_label}_good_stats_phase_comparison.csv", index=False)

    # =================================================================
    # STEP 3: FIGURES (with individual session data points)
    # =================================================================

    print(f"\n--- GENERATING FIGURES: {nt} ---")
    w = 0.35

    # Figure 1: State comparison — bar + individual session dots
    if len(state_df) > 0:
        fig, ax = plt.subplots(figsize=(14, 7))
        x = np.arange(len(state_df))
        fm_vals = state_df['Fed_Mean'].values
        f_sem_vals = state_df['Fed_SEM'].values
        am_vals = state_df['Fasted_Mean'].values
        a_sem_vals = state_df['Fasted_SEM'].values

        ax.bar(x-w/2, fm_vals, w, yerr=f_sem_vals, label='Fed (N=4)', capsize=8,
               color='#3498db', alpha=0.7, error_kw={'linewidth': 2})
        ax.bar(x+w/2, am_vals, w, yerr=a_sem_vals, label='Fasted (N=4)', capsize=8,
               color='#e74c3c', alpha=0.7, error_kw={'linewidth': 2})

        # Overlay individual session data points
        for i, row in state_df.iterrows():
            fed_pts = row['Fed_Sessions']
            fas_pts = row['Fasted_Sessions']
            jitter = 0.06
            ax.scatter([i - w/2 + np.random.uniform(-jitter, jitter) for _ in fed_pts],
                       fed_pts, color='#2c3e50', s=50, zorder=5, edgecolors='white', linewidth=0.5)
            ax.scatter([i + w/2 + np.random.uniform(-jitter, jitter) for _ in fas_pts],
                       fas_pts, color='#2c3e50', s=50, zorder=5, edgecolors='white', linewidth=0.5)

        ax.set_xlabel('Time Lag (ms)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Session Mean Cross-Correlation +/- SEM', fontsize=12, fontweight='bold')
        ax.set_title(f'Single Probe {nt} (Good Units): Fed vs Fasted (Session-Level)', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(state_df['Lag_ms'].values.astype(int))
        ax.legend(fontsize=11, loc='upper left')
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')

        all_pts = np.concatenate([np.concatenate(state_df['Fed_Sessions'].values),
                                   np.concatenate(state_df['Fasted_Sessions'].values)])
        ymin = np.min(all_pts)
        ymax = np.max(all_pts)
        margin = (ymax - ymin) * 0.4
        ax.set_ylim(ymin - margin * 0.3, ymax + margin * 1.2)

        for i, row in state_df.iterrows():
            yp = max(max(row['Fed_Sessions']), max(row['Fasted_Sessions']))
            ax.text(i, yp + margin*0.25, fmt_p(row['MannWhitney_Pvalue']), ha='center', fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', alpha=0.7))
            ax.text(i, yp + margin*0.55, f"d = {row['Cohens_d']:.3f}", ha='center', fontsize=8, style='italic')

        plt.tight_layout()
        fname = f"figures/single_probe_{nt_label}_good_state_comparison.png"
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        print(f"  [OK] Saved {fname}")
        plt.close()

    # Figure 2: Phase comparison — bar + individual session dots
    if len(phase_df) > 0:
        fig, ax = plt.subplots(figsize=(14, 7))
        x = np.arange(len(phase_df))
        em_vals = phase_df['Exploration_Mean'].values
        e_sem_vals = phase_df['Exploration_SEM'].values
        fm_vals = phase_df['Foraging_Mean'].values
        f_sem_vals = phase_df['Foraging_SEM'].values

        ax.bar(x-w/2, em_vals, w, yerr=e_sem_vals, label='Exploration (N=4)', capsize=8,
               color='#2ecc71', alpha=0.7, error_kw={'linewidth': 2})
        ax.bar(x+w/2, fm_vals, w, yerr=f_sem_vals, label='Foraging (N=4)', capsize=8,
               color='#f39c12', alpha=0.7, error_kw={'linewidth': 2})

        # Overlay individual session data points
        for i, row in phase_df.iterrows():
            exp_pts = row['Exploration_Sessions']
            for_pts = row['Foraging_Sessions']
            jitter = 0.06
            ax.scatter([i - w/2 + np.random.uniform(-jitter, jitter) for _ in exp_pts],
                       exp_pts, color='#2c3e50', s=50, zorder=5, edgecolors='white', linewidth=0.5)
            ax.scatter([i + w/2 + np.random.uniform(-jitter, jitter) for _ in for_pts],
                       for_pts, color='#2c3e50', s=50, zorder=5, edgecolors='white', linewidth=0.5)

        ax.set_xlabel('Time Lag (ms)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Session Mean Cross-Correlation +/- SEM', fontsize=12, fontweight='bold')
        ax.set_title(f'Single Probe {nt} (Good Units): Exploration vs Foraging (Session-Level)', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(phase_df['Lag_ms'].values.astype(int))
        ax.legend(fontsize=11, loc='upper left')
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')

        all_pts = np.concatenate([np.concatenate(phase_df['Exploration_Sessions'].values),
                                   np.concatenate(phase_df['Foraging_Sessions'].values)])
        ymin = np.min(all_pts)
        ymax = np.max(all_pts)
        margin = (ymax - ymin) * 0.4
        ax.set_ylim(ymin - margin * 0.3, ymax + margin * 1.2)

        for i, row in phase_df.iterrows():
            yp = max(max(row['Exploration_Sessions']), max(row['Foraging_Sessions']))
            ax.text(i, yp + margin*0.25, fmt_p(row['MannWhitney_Pvalue']), ha='center', fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', alpha=0.7))
            ax.text(i, yp + margin*0.55, f"d = {row['Cohens_d']:.3f}", ha='center', fontsize=8, style='italic')

        plt.tight_layout()
        fname = f"figures/single_probe_{nt_label}_good_phase_comparison.png"
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        print(f"  [OK] Saved {fname}")
        plt.close()

    # Figure 3: Summary table
    if len(state_df) > 0 and len(phase_df) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(22, 6))
        ax = axes[0]; ax.axis('off')
        tbl = [['Lag', 'Fed (N=4)', 'Fasted (N=4)', 'p-value', "Cohen's d"]]
        for _, r in state_df.iterrows():
            tbl.append([f"{int(r['Lag_ms'])}ms",
                         f"{r['Fed_Mean']:.6f} +/- {r['Fed_SEM']:.6f}",
                         f"{r['Fasted_Mean']:.6f} +/- {r['Fasted_SEM']:.6f}",
                         fmt_p(r['MannWhitney_Pvalue']), f"{r['Cohens_d']:.3f}"])
        t = ax.table(cellText=tbl, cellLoc='center', loc='center', colWidths=[0.07,0.25,0.25,0.25,0.12])
        t.auto_set_font_size(False); t.set_fontsize(10); t.scale(1,2.2)
        for i in range(5): t[(0,i)].set_facecolor('#3498db'); t[(0,i)].set_text_props(weight='bold',color='white')
        for i in range(1,len(tbl)):
            for j in range(5): t[(i,j)].set_facecolor('#ecf0f1' if i%2==0 else 'white')
        ax.set_title(f'Single Probe {nt} (Good): Fed vs Fasted (Session-Level)', fontsize=12, fontweight='bold', pad=20)

        ax = axes[1]; ax.axis('off')
        tbl = [['Lag', 'Exploration (N=4)', 'Foraging (N=4)', 'p-value', "Cohen's d"]]
        for _, r in phase_df.iterrows():
            tbl.append([f"{int(r['Lag_ms'])}ms",
                         f"{r['Exploration_Mean']:.6f} +/- {r['Exploration_SEM']:.6f}",
                         f"{r['Foraging_Mean']:.6f} +/- {r['Foraging_SEM']:.6f}",
                         fmt_p(r['MannWhitney_Pvalue']), f"{r['Cohens_d']:.3f}"])
        t = ax.table(cellText=tbl, cellLoc='center', loc='center', colWidths=[0.07,0.25,0.25,0.25,0.12])
        t.auto_set_font_size(False); t.set_fontsize(10); t.scale(1,2.2)
        for i in range(5): t[(0,i)].set_facecolor('#2ecc71'); t[(0,i)].set_text_props(weight='bold',color='white')
        for i in range(1,len(tbl)):
            for j in range(5): t[(i,j)].set_facecolor('#ecf0f1' if i%2==0 else 'white')
        ax.set_title(f'Single Probe {nt} (Good): Exp vs For (Session-Level)', fontsize=12, fontweight='bold', pad=20)

        plt.tight_layout()
        fname = f"figures/single_probe_{nt_label}_good_summary_tables.png"
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        print(f"  [OK] Saved {fname}")
        plt.close()

print(f"\n[DONE] Single-probe session-level analysis complete!")
