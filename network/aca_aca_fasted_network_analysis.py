"""
ACA-ACA Network Analysis — FASTED sessions (dual-probe probe-0)
Bin size: 100ms, Lags: 10ms, 50ms, 100ms

Fasted session groups:
  fasted_exploration: sessions 11, 13, 15
  fasted_foraging: sessions 12, 14, 16
"""

import yaml
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import spikeinterface.extractors as se
import warnings
import time

warnings.filterwarnings('ignore')

with open("paths.yaml") as f:
    paths_config = yaml.safe_load(f)

metrics_df = pd.read_csv("data/double_probe_probe0_fasted_unit_metrics.csv")

good_aca_units = metrics_df[
    (metrics_df['passes_qc'] == True) &
    (metrics_df['region'] == 'ACA')
].copy()

print("="*70)
print("ACA-ACA NETWORK ANALYSIS — FASTED (DUAL-PROBE PROBE-0)")
print("="*70)
print(f"Total ACA good units (fasted): {len(good_aca_units)}")
print(f"  Exploration: {len(good_aca_units[good_aca_units['phase'] == 'exploration'])}")
print(f"  Foraging: {len(good_aca_units[good_aca_units['phase'] == 'foraging'])}")
print(f"\nBin size: 100ms")
print(f"Lags to test: 10ms, 50ms, 100ms\n")


def compute_cross_correlation_at_lags(spike_times_1, spike_times_2, lags_ms=[10, 50, 100], bin_size_ms=100, fs=30000):
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


def compute_session_aca_connectivity(session_key, sorted_path, session_aca_units, session_num, total_sessions):
    try:
        sorting = se.read_kilosort(Path(sorted_path))
        unit_ids = session_aca_units['unit_id'].values

        print(f"\n  Session {session_num}: Loading spike data for {len(unit_ids)} ACA units...")
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
                if pair_count % 500 == 0:
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


# =============================================================================
# MAIN
# =============================================================================

print("="*70)
print("PROCESSING FASTED SESSIONS")
print("="*70)

start_time = time.time()

session_groups = {
    'fasted_exploration': [('session_11', 11), ('session_13', 13), ('session_15', 15)],
    'fasted_foraging': [('session_12', 12), ('session_14', 14), ('session_16', 16)]
}

all_results = {}
session_counter = 0
total_sessions = 6

for group_name, session_list in session_groups.items():
    print(f"\n{'='*70}")
    print(f"GROUP: {group_name.upper()}")
    print(f"{'='*70}")

    group_connectivity = []

    for session_name, session_num in session_list:
        session_counter += 1

        session_filter_key = f"mouse01_double_probe_coor1_{session_name}"
        session_data = good_aca_units[good_aca_units['session'] == session_filter_key]

        if len(session_data) == 0:
            print(f"\n  Session {session_num}: No ACA units found")
            continue

        session_config = paths_config["double_probe"]["coordinates_1"]["mouse01"]["sessions"][session_name]
        probe0_data = session_config.get("probe_0_aca", {})
        sorted_path = probe0_data.get("sorted") if probe0_data else None

        if sorted_path is None:
            print(f"\n  Session {session_num}: No sorted path available")
            continue

        conn_data = compute_session_aca_connectivity(session_filter_key, sorted_path, session_data, session_num, total_sessions)

        if conn_data:
            group_connectivity.extend(conn_data)

        elapsed = time.time() - start_time
        avg_time_per_session = elapsed / session_counter
        remaining_sessions = total_sessions - session_counter
        est_remaining = avg_time_per_session * remaining_sessions
        print(f"  [TIME] Elapsed: {elapsed:.0f}s, Est. remaining: {est_remaining:.0f}s ({est_remaining/60:.1f}min)")

    if group_connectivity:
        all_results[group_name] = pd.DataFrame(group_connectivity)
        print(f"\n  {group_name}: {len(group_connectivity)} ACA-ACA pairs")

# =============================================================================
# AGGREGATE
# =============================================================================

print(f"\n{'='*70}")
print("AGGREGATING RESULTS")
print(f"{'='*70}\n")

summary_rows = []
for group_name, df in all_results.items():
    state, phase = group_name.split('_')
    for lag in [10, 50, 100]:
        corr_col = f'correlation_{lag}ms'
        correlations = df[corr_col].dropna().values
        if len(correlations) > 0:
            summary_rows.append({
                'Group': group_name,
                'State': state,
                'Phase': phase,
                'Lag_ms': lag,
                'Mean_Correlation': np.mean(correlations),
                'Std_Correlation': np.std(correlations),
                'Median_Correlation': np.median(correlations),
                'N_Pairs': len(correlations)
            })

summary_df = pd.DataFrame(summary_rows)

summary_file = Path("data/aca_aca_fasted_network_summary.csv")
summary_df.to_csv(summary_file, index=False)
print(f"[OK] Saved summary to: {summary_file}\n")

for group_name, df in all_results.items():
    detail_file = Path(f"data/aca_aca_connectivity_{group_name}.csv")
    df.to_csv(detail_file, index=False)
    print(f"[OK] Saved detailed data: {detail_file}")

# =============================================================================
# SUMMARY
# =============================================================================

print(f"\n{'='*70}")
print("ACA-ACA FASTED NETWORK SUMMARY")
print("="*70 + "\n")
print(summary_df.to_string(index=False))

print("\n" + "="*70)
print("KEY FINDINGS (FASTED)")
print("="*70)

for lag in [10, 50, 100]:
    lag_data = summary_df[summary_df['Lag_ms'] == lag]
    exp_mean = lag_data[lag_data['Phase'] == 'exploration']['Mean_Correlation'].mean()
    for_mean = lag_data[lag_data['Phase'] == 'foraging']['Mean_Correlation'].mean()

    if not np.isnan(exp_mean) and not np.isnan(for_mean):
        pct_change = ((for_mean - exp_mean) / abs(exp_mean)) * 100 if exp_mean != 0 else 0
        print(f"\nLag {lag}ms:")
        print(f"  Exploration: {exp_mean:.4f}")
        print(f"  Foraging: {for_mean:.4f}")
        print(f"  Change (For vs Exp): {pct_change:+.1f}%")

total_time = time.time() - start_time
print(f"\n{'='*70}")
print(f"TOTAL ANALYSIS TIME: {total_time:.1f}s ({total_time/60:.1f} minutes)")
print(f"{'='*70}")
print(f"\n[DONE] ACA-ACA fasted network analysis complete!")
