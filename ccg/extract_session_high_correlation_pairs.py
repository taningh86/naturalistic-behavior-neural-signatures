"""
Extract high-correlation unit pairs for each session
Output: CSV files for each session showing strongly correlated LHA-LHA pairs
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
print("EXTRACTING HIGH-CORRELATION UNIT PAIRS PER SESSION")
print("="*80)

# Load config and metrics
with open("paths.yaml") as f:
    paths_config = yaml.safe_load(f)

metrics_df = pd.read_csv("data/all_sessions_unit_metrics_by_region.csv")

# Filter for good LHA units
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

def compute_and_filter_session_pairs(session_name, session_num, sorted_path, min_correlation=0.3):
    """Compute correlations and filter for high-correlation pairs."""
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

        df = pd.DataFrame(connectivity_data)
        session_time = time.time() - session_start
        print(f"  Session {session_num} complete: {len(connectivity_data)} total pairs in {session_time:.1f}s")

        # Filter for high correlations on 10ms lag (strongest effect)
        df['max_correlation'] = df[['correlation_10ms', 'correlation_50ms', 'correlation_100ms']].abs().max(axis=1)
        high_corr_df = df[df['max_correlation'] >= min_correlation].copy()

        print(f"  Pairs with correlation >= {min_correlation}: {len(high_corr_df)}")

        # Sort by max correlation
        high_corr_df = high_corr_df.sort_values('max_correlation', ascending=False)

        return high_corr_df

    except Exception as e:
        print(f"  [ERROR] Session {session_num}: {e}")
        return None

# ============================================================================
# PROCESS ALL 8 SESSIONS
# ============================================================================

print("\n" + "="*80)
print("PROCESSING ALL SESSIONS")
print("="*80)

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

    print(f"\n{'='*80}")
    print(f"SESSION {session_num}: {state.upper()} - {phase.upper()}")
    print(f"{'='*80}")

    high_corr_df = compute_and_filter_session_pairs(session_name, session_num, sorted_path, min_correlation=0.3)

    if high_corr_df is not None:
        # Create descriptive filename
        output_file = Path(f"data/lha_lha_high_corr_pairs_session{session_num}_{state}_{phase}.csv")

        # Save with all correlation lags
        high_corr_df = high_corr_df[['unit_1', 'unit_2', 'correlation_10ms', 'correlation_50ms', 'correlation_100ms', 'max_correlation']]
        high_corr_df.to_csv(output_file, index=False)

        print(f"[OK] Saved {len(high_corr_df)} high-correlation pairs to: {output_file}")

        # Print top 10
        print(f"\nTop 10 strongest pairs:")
        print(high_corr_df.head(10).to_string(index=False))

    elapsed = time.time() - start_time
    avg_per_session = elapsed / session_num
    remaining_sessions = 8 - session_num
    est_remaining = avg_per_session * remaining_sessions
    print(f"\n[TIME] Elapsed: {elapsed:.0f}s, Est. remaining: {est_remaining:.0f}s")

total_time = time.time() - start_time
print(f"\n" + "="*80)
print(f"TOTAL TIME: {total_time:.1f}s")
print(f"="*80)
print("\nOutput files created:")
for session_num in range(1, 9):
    session_name, phase, state = session_names[session_num]
    output_file = Path(f"data/lha_lha_high_corr_pairs_session{session_num}_{state}_{phase}.csv")
    print(f"  {output_file}")

print("\n[DONE]")
