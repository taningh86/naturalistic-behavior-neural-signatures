"""
RSP-RSP Network Analysis: Cross-correlation with statistical testing
Bin size: 100ms, Lags: 10ms, 50ms, 100ms
"""

import yaml
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import spikeinterface.extractors as se
import warnings
import time

warnings.filterwarnings('ignore')

# Load config and metrics
with open("paths.yaml") as f:
    paths_config = yaml.safe_load(f)

metrics_df = pd.read_csv("data/all_sessions_unit_metrics_by_region.csv")

# Filter for Mouse01 Coordinates-1, good RSP units only
good_rsp_units = metrics_df[
    (metrics_df['session'].str.contains('mouse01_coordinates_1')) &
    (metrics_df['passes_qc'] == True) &
    (metrics_df['region'] == 'RSP')
].copy()

print("="*70)
print("RSP-RSP NETWORK STRUCTURE ANALYSIS")
print("="*70)
print(f"Total RSP good units: {len(good_rsp_units)}")
print(f"  Fed: {len(good_rsp_units[good_rsp_units['state'] == 'fed'])}")
print(f"  Fasted: {len(good_rsp_units[good_rsp_units['state'] == 'fasted'])}")
print(f"\nBin size: 100ms")
print(f"Lags to test: 10ms, 50ms, 100ms\n")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def compute_cross_correlation_at_lags(spike_times_1, spike_times_2, lags_ms=[10, 50, 100], bin_size_ms=100, fs=30000):
    """Compute cross-correlation at specific time lags."""
    if len(spike_times_1) < 2 or len(spike_times_2) < 2:
        return {lag: np.nan for lag in lags_ms}

    spike_times_1 = np.array(spike_times_1)
    spike_times_2 = np.array(spike_times_2)

    max_time = max(np.max(spike_times_1), np.max(spike_times_2))
    min_time = min(np.min(spike_times_1), np.min(spike_times_2))

    bin_size_samples = int(bin_size_ms * fs / 1000)
    n_bins = int((max_time - min_time) / bin_size_samples) + 1

    train_1 = np.zeros(n_bins)
    train_2 = np.zeros(n_bins)

    bins_1 = (spike_times_1 - min_time) // bin_size_samples
    bins_2 = (spike_times_2 - min_time) // bin_size_samples

    bins_1 = bins_1[bins_1 < n_bins]
    bins_2 = bins_2[bins_2 < n_bins]

    train_1[bins_1] += 1
    train_2[bins_2] += 1

    train_1 = (train_1 - np.mean(train_1)) / (np.std(train_1) + 1e-8)
    train_2 = (train_2 - np.mean(train_2)) / (np.std(train_2) + 1e-8)

    results = {}
    for lag_ms in lags_ms:
        lag_bins = int(lag_ms * fs / (1000 * bin_size_samples))

        if lag_bins >= len(train_1):
            results[lag_ms] = np.nan
            continue

        if lag_bins > 0:
            train_2_shifted = np.roll(train_2, lag_bins)
            train_2_shifted[:lag_bins] = 0
        else:
            train_2_shifted = train_2

        correlation = np.corrcoef(train_1, train_2_shifted)[0, 1]
        results[lag_ms] = correlation if not np.isnan(correlation) else 0.0

    return results

def compute_session_rsp_connectivity(session_key, sorted_path, session_rsp_units, session_num, total_sessions):
    """Compute RSP-RSP connectivity for a single session."""
    try:
        sorting = se.read_kilosort(Path(sorted_path))
        unit_ids = session_rsp_units['unit_id'].values

        print(f"\n  Session {session_num}: Loading spike data for {len(unit_ids)} RSP units...")
        session_start = time.time()

        connectivity_data = []
        pair_count = 0
        total_pairs = len(unit_ids) * (len(unit_ids) - 1) // 2

        for i, uid1 in enumerate(unit_ids):
            spike_times_1 = sorting.get_unit_spike_train(uid1)

            for j, uid2 in enumerate(unit_ids):
                if i >= j:
                    continue

                spike_times_2 = sorting.get_unit_spike_train(uid2)
                cc_results = compute_cross_correlation_at_lags(spike_times_1, spike_times_2)

                connectivity_data.append({
                    'unit_1': uid1,
                    'unit_2': uid2,
                    'correlation_10ms': cc_results[10],
                    'correlation_50ms': cc_results[50],
                    'correlation_100ms': cc_results[100]
                })

                pair_count += 1

                if pair_count % 100 == 0:
                    elapsed = time.time() - session_start
                    rate = pair_count / elapsed
                    remaining = (total_pairs - pair_count) / rate
                    print(f"    Progress: {pair_count}/{total_pairs} pairs ({pair_count/total_pairs*100:.1f}%) - {remaining:.0f}s remaining")

        session_time = time.time() - session_start
        print(f"  Session {session_num} complete: {len(connectivity_data)} pairs in {session_time:.1f}s")

        return connectivity_data

    except Exception as e:
        print(f"  [ERROR] Session {session_num}: {e}")
        return None

# ============================================================================
# MAIN ANALYSIS
# ============================================================================

print("="*70)
print("PROCESSING SESSIONS")
print("="*70)

start_time = time.time()

session_groups = {
    'fed_exploration': [('session_1', 1), ('session_3', 3)],
    'fed_foraging': [('session_2', 2), ('session_4', 4)],
    'fasted_exploration': [('session_5', 5), ('session_7', 7)],
    'fasted_foraging': [('session_6', 6), ('session_8', 8)]
}

all_results = {}
session_counter = 0
total_sessions = 8

for group_name, session_list in session_groups.items():
    print(f"\n{'='*70}")
    print(f"GROUP: {group_name.upper()}")
    print(f"{'='*70}")

    group_connectivity = []

    for session_name, session_num in session_list:
        session_counter += 1
        session_key = f"mouse01_coordinates_1_{session_name}"
        session_data = good_rsp_units[good_rsp_units['session'] == session_key]

        if len(session_data) == 0:
            print(f"\n  Session {session_num}: No RSP units found")
            continue

        coords = "coordinates_1"
        session_config = paths_config["single_probe"][coords]["mouse01"]["sessions"][session_name]
        sorted_path = session_config["sorted"]

        conn_data = compute_session_rsp_connectivity(session_key, sorted_path, session_data, session_num, total_sessions)

        if conn_data:
            group_connectivity.extend(conn_data)

        elapsed = time.time() - start_time
        avg_time_per_session = elapsed / session_counter
        remaining_sessions = total_sessions - session_counter
        est_remaining = avg_time_per_session * remaining_sessions

        print(f"  [TIME] Elapsed: {elapsed:.0f}s, Est. remaining: {est_remaining:.0f}s ({est_remaining/60:.1f}min)")

    if group_connectivity:
        all_results[group_name] = pd.DataFrame(group_connectivity)
        print(f"\n  {group_name}: {len(group_connectivity)} RSP-RSP pairs")

# ============================================================================
# SAVE CONNECTIVITY DATA
# ============================================================================

print(f"\n{'='*70}")
print("SAVING CONNECTIVITY DATA")
print(f"{'='*70}\n")

for group_name, df in all_results.items():
    detail_file = Path(f"data/rsp_rsp_connectivity_{group_name}.csv")
    df.to_csv(detail_file, index=False)
    print(f"[OK] {detail_file}")

# ============================================================================
# LOAD DATA FOR STATISTICAL ANALYSIS
# ============================================================================

print(f"\n{'='*70}")
print("STATISTICAL ANALYSIS")
print(f"{'='*70}\n")

connectivity_data = []
for group in ['fed_exploration', 'fed_foraging', 'fasted_exploration', 'fasted_foraging']:
    df = pd.read_csv(f"data/rsp_rsp_connectivity_{group}.csv")
    state, phase = group.split('_')
    df['state'] = state
    df['phase'] = phase
    connectivity_data.append(df)

full_df = pd.concat(connectivity_data, ignore_index=True)

print(f"Total RSP-RSP pairs analyzed: {len(full_df)}")

# ============================================================================
# STATISTICAL TESTS
# ============================================================================

def compute_cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std > 0 else 0

def compute_ci_95(data):
    mean = np.mean(data)
    sem = stats.sem(data)
    ci = sem * stats.t.ppf((1 + 0.95) / 2, len(data) - 1)
    return mean, mean - ci, mean + ci

# State comparison
print("\n1. STATE COMPARISON: FED vs FASTED")
state_results = []

for lag in [10, 50, 100]:
    corr_col = f'correlation_{lag}ms'
    fed_data = full_df[full_df['state'] == 'fed'][corr_col].dropna().values
    fasted_data = full_df[full_df['state'] == 'fasted'][corr_col].dropna().values

    mw_stat, mw_pval = stats.mannwhitneyu(fed_data, fasted_data)
    t_stat, t_pval = stats.ttest_ind(fed_data, fasted_data)
    cohens_d = compute_cohens_d(fed_data, fasted_data)

    fed_mean, fed_ci_low, fed_ci_high = compute_ci_95(fed_data)
    fasted_mean, fasted_ci_low, fasted_ci_high = compute_ci_95(fasted_data)

    state_results.append({
        'Lag_ms': lag,
        'Fed_Mean': fed_mean,
        'Fed_CI_Low': fed_ci_low,
        'Fed_CI_High': fed_ci_high,
        'Fasted_Mean': fasted_mean,
        'Fasted_CI_Low': fasted_ci_low,
        'Fasted_CI_High': fasted_ci_high,
        'MannWhitney_Pvalue': mw_pval,
        'Ttest_Pvalue': t_pval,
        'Cohens_d': cohens_d,
        'N_Fed': len(fed_data),
        'N_Fasted': len(fasted_data)
    })

state_df = pd.DataFrame(state_results)

# Phase comparison
print("2. PHASE COMPARISON: EXPLORATION vs FORAGING")
phase_results = []

for lag in [10, 50, 100]:
    corr_col = f'correlation_{lag}ms'
    exp_data = full_df[full_df['phase'] == 'exploration'][corr_col].dropna().values
    for_data = full_df[full_df['phase'] == 'foraging'][corr_col].dropna().values

    mw_stat, mw_pval = stats.mannwhitneyu(exp_data, for_data)
    t_stat, t_pval = stats.ttest_ind(exp_data, for_data)
    cohens_d = compute_cohens_d(exp_data, for_data)

    exp_mean, exp_ci_low, exp_ci_high = compute_ci_95(exp_data)
    for_mean, for_ci_low, for_ci_high = compute_ci_95(for_data)

    phase_results.append({
        'Lag_ms': lag,
        'Exploration_Mean': exp_mean,
        'Exploration_CI_Low': exp_ci_low,
        'Exploration_CI_High': exp_ci_high,
        'Foraging_Mean': for_mean,
        'Foraging_CI_Low': for_ci_low,
        'Foraging_CI_High': for_ci_high,
        'MannWhitney_Pvalue': mw_pval,
        'Ttest_Pvalue': t_pval,
        'Cohens_d': cohens_d,
        'N_Exploration': len(exp_data),
        'N_Foraging': len(for_data)
    })

phase_df = pd.DataFrame(phase_results)

# Lag comparison
print("3. LAG COMPARISON: 10ms vs 50ms vs 100ms")
lag_results = []

for lag in [10, 50, 100]:
    corr_col = f'correlation_{lag}ms'
    lag_data = full_df[corr_col].dropna().values
    mean, ci_low, ci_high = compute_ci_95(lag_data)
    lag_results.append({
        'Lag_ms': lag,
        'Mean': mean,
        'CI_Low': ci_low,
        'CI_High': ci_high,
        'N': len(lag_data)
    })

lag_df = pd.DataFrame(lag_results)

# Save results
print("\nSaving statistical results...")
state_df.to_csv("data/rsp_rsp_stats_state_comparison.csv", index=False)
print("[OK] rsp_rsp_stats_state_comparison.csv")

phase_df.to_csv("data/rsp_rsp_stats_phase_comparison.csv", index=False)
print("[OK] rsp_rsp_stats_phase_comparison.csv")

lag_df.to_csv("data/rsp_rsp_stats_lag_comparison.csv", index=False)
print("[OK] rsp_rsp_stats_lag_comparison.csv\n")

# ============================================================================
# PRINT SUMMARIES
# ============================================================================

print("="*70)
print("RSP-RSP NETWORK SUMMARY")
print("="*70)
print(state_df.to_string(index=False))

print("\n" + "="*70)
print("PHASE COMPARISON")
print("="*70)
print(phase_df.to_string(index=False))

print("\n" + "="*70)
print("LAG COMPARISON")
print("="*70)
print(lag_df.to_string(index=False))

total_time = time.time() - start_time
print(f"\n{'='*70}")
print(f"TOTAL ANALYSIS TIME: {total_time:.1f}s ({total_time/60:.1f} minutes)")
print(f"{'='*70}")
print(f"\n[DONE] RSP-RSP network analysis complete!")
