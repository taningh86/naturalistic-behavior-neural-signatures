"""
Per-Session LHA-LHA Hot Pairs Analysis
Compares individual fed vs fasted sessions to identify unit pairs with consistent changes
"""

import yaml
from pathlib import Path
import numpy as np
import pandas as pd
import spikeinterface.extractors as se
import warnings
import time

warnings.filterwarnings('ignore')

print("="*80)
print("PER-SESSION LHA-LHA HOT PAIRS ANALYSIS")
print("="*80)

# Load config and metrics
with open("paths.yaml") as f:
    paths_config = yaml.safe_load(f)

metrics_df = pd.read_csv("data/all_sessions_unit_metrics_by_region.csv")

# Filter for Mouse01 Coordinates-1, good LHA units only
good_lha_units = metrics_df[
    (metrics_df['session'].str.contains('mouse01_coordinates_1')) &
    (metrics_df['passes_qc'] == True) &
    (metrics_df['region'] == 'LHA')
].copy()

print(f"Total good LHA units: {len(good_lha_units)}\n")

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

def compute_session_correlations(session_name, session_num, sorted_path):
    """Compute all LHA-LHA correlations for a single session."""
    try:
        sorting = se.read_kilosort(Path(sorted_path))
        session_key = f"mouse01_coordinates_1_session_{session_num}"
        session_lha = good_lha_units[good_lha_units['session'] == session_key]

        if len(session_lha) == 0:
            print(f"  [SKIP] Session {session_num}: No good LHA units")
            return None

        unit_ids = session_lha['unit_id'].values
        print(f"  Session {session_num}: Computing correlations for {len(unit_ids)} units...")

        connectivity_data = []
        pair_count = 0
        total_pairs = len(unit_ids) * (len(unit_ids) - 1) // 2
        session_start = time.time()

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

                if pair_count % 200 == 0:
                    elapsed = time.time() - session_start
                    rate = pair_count / elapsed
                    remaining = (total_pairs - pair_count) / rate
                    print(f"    Progress: {pair_count}/{total_pairs} ({pair_count/total_pairs*100:.1f}%) - {remaining:.0f}s remaining")

        session_time = time.time() - session_start
        print(f"  Session {session_num} complete: {len(connectivity_data)} pairs in {session_time:.1f}s")

        return pd.DataFrame(connectivity_data)

    except Exception as e:
        print(f"  [ERROR] Session {session_num}: {e}")
        return None

# ============================================================================
# COMPUTE CORRELATIONS FOR ALL SESSIONS
# ============================================================================

print("\n" + "="*80)
print("COMPUTING PER-SESSION CORRELATIONS")
print("="*80)

session_data = {}
session_names = {
    1: ('session_1', 'exploration', 'fed'),
    2: ('session_2', 'foraging', 'fed'),
    3: ('session_3', 'exploration', 'fed'),
    4: ('session_4', 'foraging', 'fed'),
    5: ('session_5', 'exploration', 'fasted'),
    6: ('session_6', 'foraging', 'fasted'),
    7: ('session_7', 'exploration', 'fasted'),
    8: ('session_8', 'foraging', 'fasted')
}

print()
start_time = time.time()

for session_num in range(1, 9):
    session_name, phase, state = session_names[session_num]
    coords = "coordinates_1"
    session_config = paths_config["single_probe"][coords]["mouse01"]["sessions"][session_name]
    sorted_path = session_config["sorted"]

    df = compute_session_correlations(session_name, session_num, sorted_path)
    if df is not None:
        session_data[session_num] = df

    elapsed = time.time() - start_time
    avg_per_session = elapsed / session_num
    remaining_sessions = 8 - session_num
    est_remaining = avg_per_session * remaining_sessions
    print(f"  [TIME] Elapsed: {elapsed:.0f}s, Est. remaining: {est_remaining:.0f}s\n")

total_time = time.time() - start_time
print(f"All sessions processed in {total_time:.1f}s\n")

# ============================================================================
# COMPARE FED VS FASTED SESSIONS (SAME PHASE)
# ============================================================================

print("="*80)
print("COMPARING FED vs FASTED SESSIONS")
print("="*80)

def compare_sessions(fed_df, fasted_df, fed_num, fasted_num, phase_name, lags=[10, 50]):
    """Compare correlations between fed and fasted sessions."""

    comparison_results = {}

    for lag in lags:
        corr_col = f'correlation_{lag}ms'
        print(f"\n{'-'*80}")
        print(f"{phase_name.upper()} - Session {fed_num} (Fed) vs Session {fasted_num} (Fasted) - LAG {lag}ms")
        print(f"{'-'*80}")

        # Create pair identifiers (ensure consistent ordering)
        fed_df_copy = fed_df.copy()
        fasted_df_copy = fasted_df.copy()

        fed_df_copy['pair_id'] = fed_df_copy.apply(
            lambda row: tuple(sorted([row['unit_1'], row['unit_2']])), axis=1
        )
        fasted_df_copy['pair_id'] = fasted_df_copy.apply(
            lambda row: tuple(sorted([row['unit_1'], row['unit_2']])), axis=1
        )

        # Find common pairs
        fed_pairs = set(fed_df_copy['pair_id'])
        fasted_pairs = set(fasted_df_copy['pair_id'])
        common_pairs = fed_pairs & fasted_pairs

        print(f"Fed pairs: {len(fed_pairs)}, Fasted pairs: {len(fasted_pairs)}, Common: {len(common_pairs)}")

        # Calculate changes
        pair_changes = []

        for pair_id in common_pairs:
            fed_rows = fed_df_copy[fed_df_copy['pair_id'] == pair_id]
            fasted_rows = fasted_df_copy[fasted_df_copy['pair_id'] == pair_id]

            if len(fed_rows) > 0 and len(fasted_rows) > 0:
                fed_corr = fed_rows[corr_col].values[0]
                fasted_corr = fasted_rows[corr_col].values[0]
                delta_corr = fasted_corr - fed_corr

                pair_changes.append({
                    'unit_1': pair_id[0],
                    'unit_2': pair_id[1],
                    'fed_corr': fed_corr,
                    'fasted_corr': fasted_corr,
                    'delta_corr': delta_corr,
                    'abs_delta': abs(delta_corr)
                })

        pair_changes_df = pd.DataFrame(pair_changes)
        pair_changes_df = pair_changes_df.sort_values('abs_delta', ascending=False)

        print(f"\nTop 10 INCREASES (Fasted >> Fed):")
        top_inc = pair_changes_df[pair_changes_df['delta_corr'] > 0].head(10)
        print(top_inc[['unit_1', 'unit_2', 'fed_corr', 'fasted_corr', 'delta_corr']].to_string(index=False))

        print(f"\nTop 10 DECREASES (Fasted << Fed):")
        top_dec = pair_changes_df[pair_changes_df['delta_corr'] < 0].head(10)
        print(top_dec[['unit_1', 'unit_2', 'fed_corr', 'fasted_corr', 'delta_corr']].to_string(index=False))

        print(f"\nStatistics:")
        print(f"  Mean Delta_corr: {pair_changes_df['delta_corr'].mean():.6f}")
        print(f"  Median Delta_corr: {pair_changes_df['delta_corr'].median():.6f}")
        print(f"  Pairs increased: {len(pair_changes_df[pair_changes_df['delta_corr'] > 0])} ({len(pair_changes_df[pair_changes_df['delta_corr'] > 0])/len(pair_changes_df)*100:.1f}%)")
        print(f"  Pairs decreased: {len(pair_changes_df[pair_changes_df['delta_corr'] < 0])} ({len(pair_changes_df[pair_changes_df['delta_corr'] < 0])/len(pair_changes_df)*100:.1f}%)")

        comparison_results[lag] = pair_changes_df

    return comparison_results

# EXPLORATION PHASE
print("\n" + "="*80)
print("EXPLORATION PHASE")
print("="*80)

if 1 in session_data and 5 in session_data:
    exp_1_5 = compare_sessions(session_data[1], session_data[5], 1, 5, "exploration")

if 3 in session_data and 7 in session_data:
    exp_3_7 = compare_sessions(session_data[3], session_data[7], 3, 7, "exploration")

# FORAGING PHASE
print("\n" + "="*80)
print("FORAGING PHASE")
print("="*80)

if 2 in session_data and 6 in session_data:
    for_2_6 = compare_sessions(session_data[2], session_data[6], 2, 6, "foraging")

if 4 in session_data and 8 in session_data:
    for_4_8 = compare_sessions(session_data[4], session_data[8], 4, 8, "foraging")

# ============================================================================
# IDENTIFY CONSISTENT HOT PAIRS
# ============================================================================

print("\n" + "="*80)
print("CONSISTENT HOT PAIRS (Appear in Both Session Comparisons)")
print("="*80)

def find_consistent_pairs(comp1, comp2, phase_name, lag):
    """Find pairs that show consistent changes across two comparisons."""

    print(f"\n{phase_name.upper()} - LAG {lag}ms - Pairs appearing in BOTH comparisons:")

    df1 = comp1[lag].copy()
    df2 = comp2[lag].copy()

    df1['pair'] = df1.apply(lambda row: tuple(sorted([row['unit_1'], row['unit_2']])), axis=1)
    df2['pair'] = df2.apply(lambda row: tuple(sorted([row['unit_1'], row['unit_2']])), axis=1)

    pairs1 = set(df1['pair'])
    pairs2 = set(df2['pair'])
    common = pairs1 & pairs2

    print(f"  Pairs in comparison 1: {len(pairs1)}")
    print(f"  Pairs in comparison 2: {len(pairs2)}")
    print(f"  Common pairs: {len(common)}")

    if len(common) > 0:
        consistent_results = []

        for pair in common:
            row1 = df1[df1['pair'] == pair].iloc[0]
            row2 = df2[df2['pair'] == pair].iloc[0]

            mean_delta = (row1['delta_corr'] + row2['delta_corr']) / 2
            consistency = 1 if (row1['delta_corr'] > 0 and row2['delta_corr'] > 0) or (row1['delta_corr'] < 0 and row2['delta_corr'] < 0) else 0

            consistent_results.append({
                'unit_1': pair[0],
                'unit_2': pair[1],
                'comp1_delta': row1['delta_corr'],
                'comp2_delta': row2['delta_corr'],
                'mean_delta': mean_delta,
                'abs_mean_delta': abs(mean_delta),
                'consistent': consistency
            })

        consistent_df = pd.DataFrame(consistent_results)
        consistent_df = consistent_df.sort_values('abs_mean_delta', ascending=False)

        print(f"\n  Top 15 by magnitude (sorted by mean abs change):")
        print(consistent_df[['unit_1', 'unit_2', 'comp1_delta', 'comp2_delta', 'mean_delta', 'consistent']].head(15).to_string(index=False))

        # Save to CSV
        output_file = Path(f"data/lha_lha_consistent_hot_pairs_{phase_name}_{lag}ms.csv")
        consistent_df.to_csv(output_file, index=False)
        print(f"\n  [OK] Saved to: {output_file}")

        return consistent_df

    return None

# EXPLORATION
if 1 in session_data and 5 in session_data and 3 in session_data and 7 in session_data:
    exp_10ms = find_consistent_pairs(exp_1_5, exp_3_7, "exploration", 10)
    exp_50ms = find_consistent_pairs(exp_1_5, exp_3_7, "exploration", 50)

# FORAGING
if 2 in session_data and 6 in session_data and 4 in session_data and 8 in session_data:
    for_10ms = find_consistent_pairs(for_2_6, for_4_8, "foraging", 10)
    for_50ms = find_consistent_pairs(for_2_6, for_4_8, "foraging", 50)

print("\n" + "="*80)
print("[DONE] Per-session hot pairs analysis complete!")
print("="*80)
