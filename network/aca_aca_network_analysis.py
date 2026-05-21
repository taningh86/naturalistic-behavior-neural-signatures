"""
ACA-ACA Network Analysis: Cross-correlation for dual-probe probe-0 (ACA)
Bin size: 100ms
Lags: 10ms, 50ms, 100ms

All sessions are fed — groups by phase only:
  fed_exploration: sessions 1, 3, 5, 7, 9
  fed_foraging: sessions 4, 6, 8, 10
"""

import yaml
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import spikeinterface.extractors as se
import warnings
import time

warnings.filterwarnings('ignore')

# Load config and metrics
with open("paths.yaml") as f:
    paths_config = yaml.safe_load(f)

metrics_df = pd.read_csv("data/double_probe_probe0_unit_metrics.csv")

# Filter for good ACA units only
good_aca_units = metrics_df[
    (metrics_df['passes_qc'] == True) &
    (metrics_df['region'] == 'ACA')
].copy()

print("="*70)
print("ACA-ACA NETWORK STRUCTURE ANALYSIS (DUAL-PROBE PROBE-0)")
print("="*70)
print(f"Total ACA good units: {len(good_aca_units)}")
print(f"  Exploration: {len(good_aca_units[good_aca_units['phase'] == 'exploration'])}")
print(f"  Foraging: {len(good_aca_units[good_aca_units['phase'] == 'foraging'])}")
print(f"\nBin size: 100ms")
print(f"Lags to test: 10ms, 50ms, 100ms\n")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def compute_cross_correlation_at_lags(spike_times_1, spike_times_2, lags_ms=[10, 50, 100], bin_size_ms=100, fs=30000):
    """
    Compute cross-correlation at specific time lags.
    """
    if len(spike_times_1) < 2 or len(spike_times_2) < 2:
        return {lag: np.nan for lag in lags_ms}

    spike_times_1 = np.array(spike_times_1)
    spike_times_2 = np.array(spike_times_2)

    max_time = max(np.max(spike_times_1), np.max(spike_times_2))
    min_time = min(np.min(spike_times_1), np.min(spike_times_2))

    # Convert bin size to samples
    bin_size_samples = int(bin_size_ms * fs / 1000)

    # Create spike count vectors
    n_bins = int((max_time - min_time) / bin_size_samples) + 1

    train_1 = np.zeros(n_bins)
    train_2 = np.zeros(n_bins)

    bins_1 = (spike_times_1 - min_time) // bin_size_samples
    bins_2 = (spike_times_2 - min_time) // bin_size_samples

    bins_1 = bins_1[bins_1 < n_bins]
    bins_2 = bins_2[bins_2 < n_bins]

    train_1[bins_1] += 1
    train_2[bins_2] += 1

    # Normalize
    train_1 = (train_1 - np.mean(train_1)) / (np.std(train_1) + 1e-8)
    train_2 = (train_2 - np.mean(train_2)) / (np.std(train_2) + 1e-8)

    # Compute cross-correlation at each lag
    results = {}
    for lag_ms in lags_ms:
        lag_bins = int(lag_ms * fs / (1000 * bin_size_samples))

        if lag_bins >= len(train_1):
            results[lag_ms] = np.nan
            continue

        # Shift train_2 by lag
        if lag_bins > 0:
            train_2_shifted = np.roll(train_2, lag_bins)
            train_2_shifted[:lag_bins] = 0
        else:
            train_2_shifted = train_2

        # Compute correlation
        correlation = np.corrcoef(train_1, train_2_shifted)[0, 1]
        results[lag_ms] = correlation if not np.isnan(correlation) else 0.0

    return results


def compute_session_aca_connectivity(session_key, sorted_path, session_aca_units, session_num, total_sessions):
    """
    Compute ACA-ACA connectivity for a single session.
    """
    try:
        sorting = se.read_kilosort(Path(sorted_path))

        unit_ids = session_aca_units['unit_id'].values

        print(f"\n  Session {session_num}: Loading spike data for {len(unit_ids)} ACA units...")
        session_start = time.time()

        connectivity_data = []

        # Compute pairwise cross-correlations
        pair_count = 0
        total_pairs = len(unit_ids) * (len(unit_ids) - 1) // 2

        for i, uid1 in enumerate(unit_ids):
            spike_times_1 = sorting.get_unit_spike_train(uid1)

            for j, uid2 in enumerate(unit_ids):
                if i >= j:
                    continue

                spike_times_2 = sorting.get_unit_spike_train(uid2)

                # Compute cross-correlations at multiple lags
                cc_results = compute_cross_correlation_at_lags(spike_times_1, spike_times_2)

                connectivity_data.append({
                    'unit_1': uid1,
                    'unit_2': uid2,
                    'correlation_10ms': cc_results[10],
                    'correlation_50ms': cc_results[50],
                    'correlation_100ms': cc_results[100]
                })

                pair_count += 1

                # Progress update every 500 pairs
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


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

print("="*70)
print("PROCESSING SESSIONS")
print("="*70)

start_time = time.time()

# Session groups — all fed, grouped by phase
# Session 2 is skipped (null paths)
session_groups = {
    'fed_exploration': [('session_1', 1), ('session_3', 3), ('session_5', 5), ('session_7', 7), ('session_9', 9)],
    'fed_foraging': [('session_4', 4), ('session_6', 6), ('session_8', 8), ('session_10', 10)]
}

all_results = {}
session_counter = 0
total_sessions = 9  # 10 minus skipped session 2

for group_name, session_list in session_groups.items():
    print(f"\n{'='*70}")
    print(f"GROUP: {group_name.upper()}")
    print(f"{'='*70}")

    group_connectivity = []

    for session_name, session_num in session_list:
        session_counter += 1

        # Build session key for filtering
        session_filter_key = f"mouse01_double_probe_coor1_{session_name}"
        session_data = good_aca_units[good_aca_units['session'] == session_filter_key]

        if len(session_data) == 0:
            print(f"\n  Session {session_num}: No ACA units found")
            continue

        # Get sorted path from double_probe config
        session_config = paths_config["double_probe"]["coordinates_1"]["mouse01"]["sessions"][session_name]
        probe0_data = session_config.get("probe_0_aca", {})
        sorted_path = probe0_data.get("sorted") if probe0_data else None

        if sorted_path is None:
            print(f"\n  Session {session_num}: No sorted path available")
            continue

        # Compute connectivity
        conn_data = compute_session_aca_connectivity(session_filter_key, sorted_path, session_data, session_num, total_sessions)

        if conn_data:
            group_connectivity.extend(conn_data)

        # Time estimate
        elapsed = time.time() - start_time
        avg_time_per_session = elapsed / session_counter
        remaining_sessions = total_sessions - session_counter
        est_remaining = avg_time_per_session * remaining_sessions

        print(f"  [TIME] Elapsed: {elapsed:.0f}s, Est. remaining: {est_remaining:.0f}s ({est_remaining/60:.1f}min)")

    # Convert to DataFrame
    if group_connectivity:
        all_results[group_name] = pd.DataFrame(group_connectivity)
        print(f"\n  {group_name}: {len(group_connectivity)} ACA-ACA pairs")

# ============================================================================
# AGGREGATE RESULTS
# ============================================================================

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

# Save summary
summary_file = Path("data/aca_aca_network_summary.csv")
summary_df.to_csv(summary_file, index=False)
print(f"[OK] Saved summary to: {summary_file}\n")

# Save detailed data
for group_name, df in all_results.items():
    detail_file = Path(f"data/aca_aca_connectivity_{group_name}.csv")
    df.to_csv(detail_file, index=False)
    print(f"[OK] Saved detailed data: {detail_file}")

# ============================================================================
# VISUALIZATION
# ============================================================================

print(f"\n{'='*70}")
print("GENERATING VISUALIZATIONS")
print(f"{'='*70}\n")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('ACA-ACA Network Structure: Cross-Correlation Analysis (Dual-Probe Probe-0)', fontsize=13, fontweight='bold')

# Plot 1: Correlation by lag and phase
ax = axes[0, 0]
plot_data = summary_df[summary_df['Lag_ms'].isin([10, 50, 100])]
sns.barplot(data=plot_data, x='Lag_ms', y='Mean_Correlation', hue='Phase', ax=ax, palette='Set2')
ax.set_xlabel('Time Lag (ms)')
ax.set_ylabel('Mean Cross-Correlation')
ax.set_title('ACA-ACA Correlation by Lag and Phase')
ax.grid(True, alpha=0.3, axis='y')

# Plot 2: Correlation at 100ms lag by phase
ax = axes[0, 1]
plot_data_100 = summary_df[summary_df['Lag_ms'] == 100]
sns.barplot(data=plot_data_100, x='Phase', y='Mean_Correlation', ax=ax, palette='Set2')
ax.set_ylabel('Mean Cross-Correlation')
ax.set_title('ACA-ACA Correlation at 100ms Lag by Phase')
ax.grid(True, alpha=0.3, axis='y')

# Plot 3: All conditions, all lags
ax = axes[1, 0]
sns.pointplot(data=plot_data, x='Lag_ms', y='Mean_Correlation', hue='Phase', ax=ax, palette='Set2')
ax.set_xlabel('Time Lag (ms)')
ax.set_ylabel('Mean Cross-Correlation')
ax.set_title('ACA-ACA Correlation Across Phases')
ax.grid(True, alpha=0.3, axis='y')

# Plot 4: Distribution of correlations at 100ms lag
ax = axes[1, 1]
for group_name, df in all_results.items():
    data = df['correlation_100ms'].dropna().values
    label = group_name.replace('_', ' ').capitalize()
    ax.hist(data, bins=30, alpha=0.5, label=label)

ax.set_xlabel('Cross-Correlation Strength')
ax.set_ylabel('Frequency')
ax.set_title('Distribution of ACA-ACA Correlations (100ms Lag)')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
fig_file = Path("figures/aca_aca_network_analysis.png")
plt.savefig(fig_file, dpi=150, bbox_inches='tight')
print(f"[OK] Saved figure to: {fig_file}\n")
plt.close()

# ============================================================================
# SUMMARY TABLE
# ============================================================================

print("="*70)
print("ACA-ACA NETWORK SUMMARY")
print("="*70 + "\n")
print(summary_df.to_string(index=False))

# ============================================================================
# KEY FINDINGS
# ============================================================================

print("\n" + "="*70)
print("KEY FINDINGS")
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
print(f"\n[DONE] ACA-ACA network analysis complete!")
