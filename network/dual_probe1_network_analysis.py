"""
Dual-Probe Probe-1 (LHA+RSP) Network Analysis:
LHA-LHA, RSP-RSP, and LHA-RSP cross-correlations.

Fed: sessions 1, 3-10 | Fasted: sessions 11-16
Groups: {fed/fasted}_{exploration/foraging}
"""

import yaml
from pathlib import Path
import numpy as np
import pandas as pd
import spikeinterface.extractors as se
import warnings
import time

warnings.filterwarnings('ignore')

with open("paths.yaml") as f:
    paths_config = yaml.safe_load(f)

metrics_df = pd.read_csv("data/double_probe_probe1_unit_metrics.csv")

# Filter for QC-passing units
good_units = metrics_df[metrics_df['passes_qc'] == True].copy()

good_lha = good_units[good_units['region'] == 'LHA']
good_rsp = good_units[good_units['region'] == 'RSP']

print("="*70)
print("PROBE-1 NETWORK ANALYSIS (LHA-LHA, RSP-RSP, LHA-RSP)")
print("="*70)
print(f"Total good LHA units: {len(good_lha)}")
print(f"  Fed: {len(good_lha[good_lha['state'] == 'fed'])}")
print(f"  Fasted: {len(good_lha[good_lha['state'] == 'fasted'])}")
print(f"Total good RSP units: {len(good_rsp)}")
print(f"  Fed: {len(good_rsp[good_rsp['state'] == 'fed'])}")
print(f"  Fasted: {len(good_rsp[good_rsp['state'] == 'fasted'])}")
print(f"\nBin size: 100ms, Lags: 10ms, 50ms, 100ms\n")


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


def compute_pairwise_connectivity(sorting, units_a, units_b, network_type, session_num):
    """Compute pairwise cross-correlations between two unit sets."""
    session_start = time.time()
    connectivity_data = []

    if network_type in ['LHA-LHA', 'RSP-RSP']:
        # Within-region: upper triangle only
        total_pairs = len(units_a) * (len(units_a) - 1) // 2
        pair_count = 0
        for i, uid1 in enumerate(units_a):
            st1 = sorting.get_unit_spike_train(uid1)
            for j, uid2 in enumerate(units_a):
                if i >= j:
                    continue
                st2 = sorting.get_unit_spike_train(uid2)
                cc = compute_cross_correlation_at_lags(st1, st2)
                connectivity_data.append({
                    'unit_1': uid1, 'unit_2': uid2,
                    'correlation_10ms': cc[10], 'correlation_50ms': cc[50], 'correlation_100ms': cc[100]
                })
                pair_count += 1
                if pair_count % 500 == 0:
                    elapsed = time.time() - session_start
                    rate = pair_count / elapsed
                    remaining = (total_pairs - pair_count) / rate if rate > 0 else 0
                    print(f"      {pair_count}/{total_pairs} ({pair_count/total_pairs*100:.1f}%) - {remaining:.0f}s left")
    else:
        # Cross-region: all pairs
        total_pairs = len(units_a) * len(units_b)
        pair_count = 0
        for uid1 in units_a:
            st1 = sorting.get_unit_spike_train(uid1)
            for uid2 in units_b:
                st2 = sorting.get_unit_spike_train(uid2)
                cc = compute_cross_correlation_at_lags(st1, st2)
                connectivity_data.append({
                    'unit_1': uid1, 'unit_2': uid2,
                    'correlation_10ms': cc[10], 'correlation_50ms': cc[50], 'correlation_100ms': cc[100]
                })
                pair_count += 1
                if pair_count % 500 == 0:
                    elapsed = time.time() - session_start
                    rate = pair_count / elapsed
                    remaining = (total_pairs - pair_count) / rate if rate > 0 else 0
                    print(f"      {pair_count}/{total_pairs} ({pair_count/total_pairs*100:.1f}%) - {remaining:.0f}s left")

    elapsed = time.time() - session_start
    print(f"    {network_type} session {session_num}: {len(connectivity_data)} pairs in {elapsed:.1f}s")
    return connectivity_data


# =============================================================================
# SESSION GROUPS
# =============================================================================

session_groups = {
    'fed_exploration': [('session_1', 1), ('session_3', 3), ('session_5', 5), ('session_7', 7), ('session_9', 9)],
    'fed_foraging': [('session_4', 4), ('session_6', 6), ('session_8', 8), ('session_10', 10)],
    'fasted_exploration': [('session_11', 11), ('session_13', 13), ('session_15', 15)],
    'fasted_foraging': [('session_12', 12), ('session_14', 14), ('session_16', 16)]
}

network_types = ['LHA-LHA', 'RSP-RSP', 'LHA-RSP']

# Results: {network_type: {group_name: DataFrame}}
all_results = {nt: {} for nt in network_types}

overall_start = time.time()

for group_name, session_list in session_groups.items():
    print(f"\n{'='*70}")
    print(f"GROUP: {group_name.upper()}")
    print(f"{'='*70}")

    group_data = {nt: [] for nt in network_types}

    for session_name, session_num in session_list:
        session_filter = f"mouse01_double_probe_coor1_{session_name}"

        # Get units for this session
        session_lha = good_lha[good_lha['session'] == session_filter]['unit_id'].values
        session_rsp = good_rsp[good_rsp['session'] == session_filter]['unit_id'].values

        print(f"\n  Session {session_num}: LHA={len(session_lha)}, RSP={len(session_rsp)}")

        if len(session_lha) == 0 and len(session_rsp) == 0:
            print(f"    [SKIP] No units")
            continue

        # Load sorting
        session_config = paths_config["double_probe"]["coordinates_1"]["mouse01"]["sessions"][session_name]
        p1 = session_config.get("probe_1_lha_rsp", {})
        sorted_path = p1.get("sorted") if p1 else None

        if sorted_path is None:
            print(f"    [SKIP] No sorted path")
            continue

        try:
            sorting = se.read_kilosort(Path(sorted_path))
        except Exception as e:
            print(f"    [ERROR] {e}")
            continue

        # LHA-LHA
        if len(session_lha) >= 2:
            conn = compute_pairwise_connectivity(sorting, session_lha, session_lha, 'LHA-LHA', session_num)
            if conn:
                group_data['LHA-LHA'].extend(conn)
        else:
            print(f"    LHA-LHA: skipped (<2 LHA units)")

        # RSP-RSP
        if len(session_rsp) >= 2:
            conn = compute_pairwise_connectivity(sorting, session_rsp, session_rsp, 'RSP-RSP', session_num)
            if conn:
                group_data['RSP-RSP'].extend(conn)
        else:
            print(f"    RSP-RSP: skipped (<2 RSP units)")

        # LHA-RSP
        if len(session_lha) >= 1 and len(session_rsp) >= 1:
            conn = compute_pairwise_connectivity(sorting, session_lha, session_rsp, 'LHA-RSP', session_num)
            if conn:
                group_data['LHA-RSP'].extend(conn)
        else:
            print(f"    LHA-RSP: skipped (need both regions)")

    # Save group data
    for nt in network_types:
        if group_data[nt]:
            all_results[nt][group_name] = pd.DataFrame(group_data[nt])
            print(f"\n  {nt} {group_name}: {len(group_data[nt])} pairs")

# =============================================================================
# AGGREGATE & SAVE
# =============================================================================

print(f"\n{'='*70}")
print("AGGREGATING AND SAVING RESULTS")
print(f"{'='*70}\n")

for nt in network_types:
    nt_prefix = nt.lower().replace('-', '_')  # e.g., lha_lha

    # Summary
    summary_rows = []
    for group_name, df in all_results[nt].items():
        parts = group_name.split('_')
        state = parts[0]
        phase = parts[1]
        for lag in [10, 50, 100]:
            corr_col = f'correlation_{lag}ms'
            vals = df[corr_col].dropna().values
            if len(vals) > 0:
                summary_rows.append({
                    'Group': group_name, 'State': state, 'Phase': phase, 'Lag_ms': lag,
                    'Mean_Correlation': np.mean(vals), 'Std_Correlation': np.std(vals),
                    'Median_Correlation': np.median(vals), 'N_Pairs': len(vals)
                })

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_file = f"data/dp1_{nt_prefix}_network_summary.csv"
        summary_df.to_csv(summary_file, index=False)
        print(f"[OK] {nt} summary: {summary_file}")

    # Detailed connectivity CSVs
    for group_name, df in all_results[nt].items():
        detail_file = f"data/dp1_{nt_prefix}_connectivity_{group_name}.csv"
        df.to_csv(detail_file, index=False)
        print(f"[OK] {nt} detail: {detail_file}")

# =============================================================================
# PRINT SUMMARIES
# =============================================================================

for nt in network_types:
    nt_prefix = nt.lower().replace('-', '_')
    summary_file = f"data/dp1_{nt_prefix}_network_summary.csv"
    try:
        s = pd.read_csv(summary_file)
        print(f"\n{'='*70}")
        print(f"{nt} NETWORK SUMMARY")
        print("="*70)
        print(s.to_string(index=False))
    except:
        print(f"\n[WARNING] No summary for {nt}")

total_time = time.time() - overall_start
print(f"\n{'='*70}")
print(f"TOTAL ANALYSIS TIME: {total_time:.1f}s ({total_time/60:.1f} minutes)")
print(f"{'='*70}")
print(f"\n[DONE] Probe-1 network analysis complete!")
