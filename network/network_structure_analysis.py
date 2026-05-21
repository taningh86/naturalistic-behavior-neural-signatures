"""
Network Structure Analysis: Connectivity differences between fed/fasted and exploration/foraging
Computes spike-time cross-correlations and analyzes functional connectivity.
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

warnings.filterwarnings('ignore')

# Load config and metrics
with open("paths.yaml") as f:
    paths_config = yaml.safe_load(f)

metrics_df = pd.read_csv("data/all_sessions_unit_metrics_by_region.csv")

# Filter for Mouse01 Coordinates-1, good units only
good_units = metrics_df[
    (metrics_df['session'].str.contains('mouse01_coordinates_1')) &
    (metrics_df['passes_qc'] == True)
].copy()

print("="*70)
print("NETWORK STRUCTURE ANALYSIS")
print("Mouse01 Coordinates-1")
print("="*70)
print(f"Total good units: {len(good_units)}")
print(f"  Fed: {len(good_units[good_units['state'] == 'fed'])}")
print(f"  Fasted: {len(good_units[good_units['state'] == 'fasted'])}\n")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def compute_cross_correlation(spike_times_1, spike_times_2, max_lag_ms=100, fs=30000):
    """
    Compute peak cross-correlation between two spike trains.
    Returns correlation strength between -1 and 1.
    """
    if len(spike_times_1) < 2 or len(spike_times_2) < 2:
        return np.nan

    max_lag_samples = int(max_lag_ms * fs / 1000)

    # Convert spike times to binary vectors in a time window
    spike_times_1 = np.array(spike_times_1)
    spike_times_2 = np.array(spike_times_2)

    max_time = max(np.max(spike_times_1), np.max(spike_times_2))
    min_time = min(np.min(spike_times_1), np.min(spike_times_2))

    # Bin spikes at 1ms resolution
    bin_size = 1000  # 1ms in samples at 30kHz
    n_bins = int((max_time - min_time) / bin_size) + 1

    train_1 = np.zeros(n_bins)
    train_2 = np.zeros(n_bins)

    bins_1 = (spike_times_1 - min_time) // bin_size
    bins_2 = (spike_times_2 - min_time) // bin_size

    train_1[bins_1[bins_1 < n_bins]] += 1
    train_2[bins_2[bins_2 < n_bins]] += 1

    # Normalize
    train_1 = (train_1 - np.mean(train_1)) / (np.std(train_1) + 1e-8)
    train_2 = (train_2 - np.mean(train_2)) / (np.std(train_2) + 1e-8)

    # Compute cross-correlation
    cc = np.correlate(train_1, train_2, mode='same')
    cc = cc / len(cc)

    return np.max(np.abs(cc))

def compute_session_connectivity(session_key, sorted_path, session_good_units):
    """
    Compute connectivity matrix for a single session.
    Returns dict with connectivity strength between all neuron pairs.
    """
    try:
        sorting = se.read_kilosort(Path(sorted_path))

        unit_ids = session_good_units['unit_id'].values
        regions = session_good_units['region'].values

        print(f"  Computing connectivity for {len(unit_ids)} units...")

        connectivity_data = {}

        # Compute pairwise cross-correlations
        for i, uid1 in enumerate(unit_ids):
            spike_times_1 = sorting.get_unit_spike_train(uid1)

            for j, uid2 in enumerate(unit_ids):
                if i >= j:  # Only compute upper triangle (symmetric)
                    continue

                spike_times_2 = sorting.get_unit_spike_train(uid2)

                # Compute cross-correlation
                cc = compute_cross_correlation(spike_times_1, spike_times_2)

                # Determine region pair
                region_pair = f"{regions[i]}_{regions[j]}"
                if region_pair not in connectivity_data:
                    connectivity_data[region_pair] = []

                connectivity_data[region_pair].append({
                    'unit_1': uid1,
                    'unit_2': uid2,
                    'region_1': regions[i],
                    'region_2': regions[j],
                    'correlation': cc
                })

        return connectivity_data

    except Exception as e:
        print(f"  [ERROR] {e}")
        return None

# ============================================================================
# MAIN ANALYSIS
# ============================================================================

# Organize sessions by state and phase
session_groups = {
    'fed_exploration': ['mouse01_coordinates_1_session_1', 'mouse01_coordinates_1_session_3'],
    'fed_foraging': ['mouse01_coordinates_1_session_2', 'mouse01_coordinates_1_session_4'],
    'fasted_exploration': ['mouse01_coordinates_1_session_5', 'mouse01_coordinates_1_session_7'],
    'fasted_foraging': ['mouse01_coordinates_1_session_6', 'mouse01_coordinates_1_session_8']
}

all_results = {}

for group_name, session_list in session_groups.items():
    print(f"\n{'='*70}")
    print(f"Processing: {group_name.upper()}")
    print(f"{'='*70}")

    group_connectivity = {'LHA_LHA': [], 'RSP_RSP': [], 'LHA_RSP': []}

    for session_key in session_list:
        session_data = good_units[good_units['session'] == session_key]

        if len(session_data) == 0:
            continue

        print(f"\n  Session: {session_key}")

        # Get sorted path
        coords = "coordinates_1"
        parts = session_key.split('_')
        session_num = f"session_{parts[-1]}"
        session_config = paths_config["single_probe"][coords]["mouse01"]["sessions"][session_num]
        sorted_path = session_config["sorted"]

        # Compute connectivity
        conn_data = compute_session_connectivity(session_key, sorted_path, session_data)

        if conn_data:
            # Organize by region pair
            for region_pair, correlations in conn_data.items():
                if region_pair in group_connectivity:
                    group_connectivity[region_pair].extend(correlations)

    # Aggregate results for this group
    summary = {}
    for region_pair, data_list in group_connectivity.items():
        if len(data_list) > 0:
            correlations = [d['correlation'] for d in data_list if not np.isnan(d['correlation'])]
            summary[region_pair] = {
                'mean': np.mean(correlations),
                'std': np.std(correlations),
                'n_pairs': len(correlations),
                'data': correlations
            }

    all_results[group_name] = summary

    print(f"\n  Summary for {group_name}:")
    for region_pair, stats in summary.items():
        print(f"    {region_pair}: {stats['mean']:.3f} +/- {stats['std']:.3f} ({stats['n_pairs']} pairs)")

# ============================================================================
# SAVE SUMMARY METRICS
# ============================================================================

print(f"\n{'='*70}")
print("SAVING RESULTS")
print(f"{'='*70}\n")

# Create summary DataFrame
summary_rows = []
for group_name, stats_dict in all_results.items():
    state, phase = group_name.split('_')
    for region_pair, stats in stats_dict.items():
        summary_rows.append({
            'Group': group_name,
            'State': state,
            'Phase': phase,
            'Region_Pair': region_pair,
            'Mean_Correlation': stats['mean'],
            'Std_Correlation': stats['std'],
            'N_Pairs': stats['n_pairs']
        })

summary_df = pd.DataFrame(summary_rows)
summary_file = Path("data/network_summary_metrics.csv")
summary_df.to_csv(summary_file, index=False)
print(f"[OK] Saved network summary to: {summary_file}\n")

# ============================================================================
# VISUALIZATIONS
# ============================================================================

print("Generating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Network Structure: Fed vs Fasted, Exploration vs Foraging', fontsize=14, fontweight='bold')

# Plot 1: LHA-RSP connectivity comparison
ax = axes[0, 0]
plot_data = []
for group_name, stats_dict in all_results.items():
    if 'LHA_RSP' in stats_dict:
        state, phase = group_name.split('_')
        plot_data.append({
            'State': state.capitalize(),
            'Phase': phase.capitalize(),
            'LHA-RSP Connectivity': stats_dict['LHA_RSP']['mean']
        })

plot_df = pd.DataFrame(plot_data)
sns.barplot(data=plot_df, x='State', y='LHA-RSP Connectivity', hue='Phase', ax=ax, palette='Set2')
ax.set_ylabel('Mean Cross-Correlation')
ax.set_title('LHA-RSP Cross-Regional Connectivity')
ax.grid(True, alpha=0.3, axis='y')

# Plot 2: Within-region connectivity
ax = axes[0, 1]
plot_data = []
for group_name, stats_dict in all_results.items():
    state, phase = group_name.split('_')
    for region_pair in ['LHA_LHA', 'RSP_RSP']:
        if region_pair in stats_dict:
            plot_data.append({
                'State': state.capitalize(),
                'Phase': phase.capitalize(),
                'Region': region_pair.replace('_', '-'),
                'Connectivity': stats_dict[region_pair]['mean']
            })

plot_df = pd.DataFrame(plot_data)
sns.barplot(data=plot_df, x='State', y='Connectivity', hue='Region', ax=ax, palette='Set1')
ax.set_ylabel('Mean Cross-Correlation')
ax.set_title('Within-Region Connectivity')
ax.grid(True, alpha=0.3, axis='y')

# Plot 3: All connectivity by region pair
ax = axes[1, 0]
plot_data = []
for group_name, stats_dict in all_results.items():
    state, phase = group_name.split('_')
    for region_pair, stats in stats_dict.items():
        plot_data.append({
            'Group': f"{state}\n{phase}",
            'Region Pair': region_pair.replace('_', '-'),
            'Connectivity': stats['mean']
        })

plot_df = pd.DataFrame(plot_data)
pivot_df = plot_df.pivot(index='Group', columns='Region Pair', values='Connectivity')
pivot_df.plot(kind='bar', ax=ax, color=['blue', 'red', 'purple'])
ax.set_ylabel('Mean Cross-Correlation')
ax.set_xlabel('State & Phase')
ax.set_title('Network Connectivity by Region Pair')
ax.legend(title='Region Pair', bbox_to_anchor=(1.05, 1), loc='upper left')
ax.grid(True, alpha=0.3, axis='y')
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Plot 4: Distribution of correlations
ax = axes[1, 1]
for group_name in ['fed_exploration', 'fed_foraging', 'fasted_exploration', 'fasted_foraging']:
    if 'LHA_RSP' in all_results[group_name]:
        data = all_results[group_name]['LHA_RSP']['data']
        label = group_name.replace('_', ' ').capitalize()
        ax.hist(data, bins=30, alpha=0.5, label=label)

ax.set_xlabel('Cross-Correlation Strength')
ax.set_ylabel('Frequency')
ax.set_title('Distribution of LHA-RSP Pairwise Correlations')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
fig_file = Path("figures/network_structure_analysis.png")
plt.savefig(fig_file, dpi=150, bbox_inches='tight')
print(f"[OK] Saved figure to: {fig_file}\n")
plt.close()

# ============================================================================
# PRINT SUMMARY TABLE
# ============================================================================

print("="*70)
print("NETWORK STRUCTURE SUMMARY")
print("="*70 + "\n")
print(summary_df.to_string(index=False))

print("\n" + "="*70)
print("KEY FINDINGS")
print("="*70)

# Compare fed vs fasted LHA-RSP
fed_lha_rsp = summary_df[(summary_df['State'] == 'fed') & (summary_df['Region_Pair'] == 'LHA_RSP')]['Mean_Correlation'].values
fasted_lha_rsp = summary_df[(summary_df['State'] == 'fasted') & (summary_df['Region_Pair'] == 'LHA_RSP')]['Mean_Correlation'].values

if len(fed_lha_rsp) > 0 and len(fasted_lha_rsp) > 0:
    fed_mean = np.mean(fed_lha_rsp)
    fasted_mean = np.mean(fasted_lha_rsp)
    pct_change = ((fasted_mean - fed_mean) / fed_mean) * 100

    print(f"\nLHA-RSP Connectivity:")
    print(f"  Fed: {fed_mean:.3f}")
    print(f"  Fasted: {fasted_mean:.3f}")
    print(f"  Change: {pct_change:+.1f}%")

print(f"\n[DONE] Network structure analysis complete!")
