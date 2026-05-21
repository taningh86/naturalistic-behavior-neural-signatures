"""
Quick population analysis - firing rates, dimensionality, visualizations.
"""

import yaml
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
from sklearn.decomposition import PCA

# Load metrics
metrics_df = pd.read_csv("data/all_sessions_unit_metrics_by_region.csv")

# Filter for Mouse01 Coordinates-1, good units only
good_units = metrics_df[
    (metrics_df['session'].str.contains('mouse01_coordinates_1')) &
    (metrics_df['passes_qc'] == True)
].copy()

print(f"\nAnalyzing {len(good_units)} good units from Mouse01 Coordinates-1")
print(f"  Fed: {len(good_units[good_units['state'] == 'fed'])} units")
print(f"  Fasted: {len(good_units[good_units['state'] == 'fasted'])} units\n")

# ============================================================================
# 1. FIRING RATE
# ============================================================================
print("="*70)
print("1. FIRING RATE BY STATE AND REGION")
print("="*70)

for state in ['fed', 'fasted']:
    state_data = good_units[good_units['state'] == state]
    print(f"\n{state.upper()}:")
    print(f"  Overall: {state_data['firing_rate_hz'].mean():.2f} +/- {state_data['firing_rate_hz'].std():.2f} Hz")

    for region in ['LHA', 'RSP']:
        region_data = state_data[state_data['region'] == region]
        if len(region_data) > 0:
            print(f"  {region}: {region_data['firing_rate_hz'].mean():.2f} +/- {region_data['firing_rate_hz'].std():.2f} Hz ({len(region_data)} units)")

# Statistical test
fed_rates = good_units[good_units['state'] == 'fed']['firing_rate_hz']
fasted_rates = good_units[good_units['state'] == 'fasted']['firing_rate_hz']
stat, pval = mannwhitneyu(fed_rates, fasted_rates)
print(f"\nMann-Whitney U test (Fed vs Fasted): p = {pval:.4f}")

# ============================================================================
# 2. SPIKE QUALITY
# ============================================================================
print("\n" + "="*70)
print("2. SPIKE QUALITY (ISI VIOLATIONS)")
print("="*70)

for state in ['fed', 'fasted']:
    state_data = good_units[good_units['state'] == state]
    print(f"\n{state.upper()}:")
    print(f"  Overall: {state_data['isi_violations'].mean()*100:.2f}% +/- {state_data['isi_violations'].std()*100:.2f}%")

    for region in ['LHA', 'RSP']:
        region_data = state_data[state_data['region'] == region]
        if len(region_data) > 0:
            print(f"  {region}: {region_data['isi_violations'].mean()*100:.2f}% +/- {region_data['isi_violations'].std()*100:.2f}%")

# ============================================================================
# 3. UNIT DISTRIBUTION
# ============================================================================
print("\n" + "="*70)
print("3. UNIT DISTRIBUTION BY STATE AND REGION")
print("="*70)

for state in ['fed', 'fasted']:
    state_data = good_units[good_units['state'] == state]
    lha_count = (state_data['region'] == 'LHA').sum()
    rsp_count = (state_data['region'] == 'RSP').sum()
    print(f"\n{state.upper()}: {len(state_data)} total")
    print(f"  LHA: {lha_count} ({lha_count/len(state_data)*100:.1f}%)")
    print(f"  RSP: {rsp_count} ({rsp_count/len(state_data)*100:.1f}%)")

# ============================================================================
# 4. VISUALIZATIONS
# ============================================================================
print("\n" + "="*70)
print("4. GENERATING VISUALIZATIONS")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Firing rates
ax = axes[0, 0]
sns.boxplot(data=good_units, x='state', y='firing_rate_hz', hue='region', ax=ax)
ax.set_ylabel('Firing Rate (Hz)')
ax.set_xlabel('Metabolic State')
ax.set_title('Firing Rate by State and Region')
ax.grid(True, alpha=0.3, axis='y')

# Plot 2: ISI violations
ax = axes[0, 1]
good_units['isi_violations_pct'] = good_units['isi_violations'] * 100
sns.boxplot(data=good_units, x='state', y='isi_violations_pct', hue='region', ax=ax)
ax.set_ylabel('ISI Violations (%)')
ax.set_xlabel('Metabolic State')
ax.set_title('Spike Quality by State and Region')
ax.grid(True, alpha=0.3, axis='y')

# Plot 3: Presence ratio
ax = axes[1, 0]
sns.boxplot(data=good_units, x='state', y='presence_ratio', hue='region', ax=ax)
ax.set_ylabel('Presence Ratio')
ax.set_xlabel('Metabolic State')
ax.set_title('Unit Activity Consistency by State')
ax.grid(True, alpha=0.3, axis='y')

# Plot 4: Unit counts by state and region
ax = axes[1, 1]
unit_counts = good_units.groupby(['state', 'region']).size().unstack()
unit_counts.plot(kind='bar', ax=ax, color=['blue', 'red'])
ax.set_ylabel('Good Unit Count')
ax.set_xlabel('Metabolic State')
ax.set_title('Distribution of Good Units')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.legend(title='Region')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
fig_file = Path("figures/population_dynamics_mouse01_coor1.png")
plt.savefig(fig_file, dpi=150, bbox_inches='tight')
print(f"[OK] Saved figure to: {fig_file}\n")
plt.close()

# ============================================================================
# 5. SUMMARY TABLE
# ============================================================================
print("="*70)
print("5. SUMMARY TABLE")
print("="*70)

summary_data = []
for state in ['fed', 'fasted']:
    for region in ['LHA', 'RSP']:
        region_state = good_units[(good_units['state'] == state) & (good_units['region'] == region)]
        if len(region_state) > 0:
            summary_data.append({
                'State': state.capitalize(),
                'Region': region,
                'N Units': len(region_state),
                'Firing Rate (Hz)': f"{region_state['firing_rate_hz'].mean():.2f} +/- {region_state['firing_rate_hz'].std():.2f}",
                'ISI Violations (%)': f"{region_state['isi_violations'].mean()*100:.2f}%",
                'Presence Ratio': f"{region_state['presence_ratio'].mean():.2f}"
            })

summary_df = pd.DataFrame(summary_data)
print("\n" + summary_df.to_string(index=False))

print("\n[DONE] Analysis complete!")
