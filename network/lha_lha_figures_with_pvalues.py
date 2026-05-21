"""
Generate figures with explicit p-values displayed prominently
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Load statistical results
state_df = pd.read_csv("data/lha_lha_stats_state_comparison.csv")
phase_df = pd.read_csv("data/lha_lha_stats_phase_comparison.csv")
lag_df = pd.read_csv("data/lha_lha_stats_lag_comparison.csv")

print("Generating figures with explicit p-values...\n")

# ============================================================================
# FIGURE 1: State Comparison with P-values
# ============================================================================

fig, ax = plt.subplots(figsize=(12, 7))

x_pos = np.arange(len(state_df))
width = 0.35

fed_means = state_df['Fed_Mean'].values
fed_cis = (state_df['Fed_CI_High'].values - state_df['Fed_CI_Low'].values) / 2
fasted_means = state_df['Fasted_Mean'].values
fasted_cis = (state_df['Fasted_CI_High'].values - state_df['Fasted_CI_Low'].values) / 2

bars1 = ax.bar(x_pos - width/2, fed_means, width, yerr=fed_cis, label='Fed', capsize=8,
               color='#3498db', alpha=0.8, error_kw={'linewidth': 2})
bars2 = ax.bar(x_pos + width/2, fasted_means, width, yerr=fasted_cis, label='Fasted', capsize=8,
               color='#e74c3c', alpha=0.8, error_kw={'linewidth': 2})

ax.set_xlabel('Time Lag (ms)', fontsize=12, fontweight='bold')
ax.set_ylabel('Mean Cross-Correlation ± 95% CI', fontsize=12, fontweight='bold')
ax.set_title('LHA-LHA Network: State Comparison (Fed vs Fasted)', fontsize=14, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(state_df['Lag_ms'].values.astype(int))
ax.legend(fontsize=11, loc='upper left')
ax.grid(True, alpha=0.3, axis='y', linestyle='--')
ax.set_ylim(0, max(fasted_means + fasted_cis) * 1.3)

# Add p-values explicitly
for i, row in state_df.iterrows():
    pval = row['MannWhitney_Pvalue']
    cohens = row['Cohens_d']

    # Position p-value above bars
    y_pos = max(fed_means[i] + fed_cis[i], fasted_means[i] + fasted_cis[i]) + 0.02

    # Format p-value
    if pval < 0.0001:
        pval_str = "p < 0.0001***"
    elif pval < 0.001:
        pval_str = f"p = {pval:.4f}**"
    elif pval < 0.01:
        pval_str = f"p = {pval:.4f}*"
    elif pval < 0.05:
        pval_str = f"p = {pval:.4f}*"
    else:
        pval_str = f"p = {pval:.4f} ns"

    ax.text(i, y_pos, pval_str, ha='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', alpha=0.7))
    ax.text(i, y_pos + 0.025, f"d = {cohens:.3f}", ha='center', fontsize=9, style='italic')

plt.tight_layout()
fig1_file = Path("figures/lha_lha_state_comparison_with_pvalues.png")
plt.savefig(fig1_file, dpi=150, bbox_inches='tight')
print(f"[OK] Saved: {fig1_file}\n")
plt.close()

# ============================================================================
# FIGURE 2: Phase Comparison with P-values
# ============================================================================

fig, ax = plt.subplots(figsize=(12, 7))

x_pos = np.arange(len(phase_df))
width = 0.35

exp_means = phase_df['Exploration_Mean'].values
exp_cis = (phase_df['Exploration_CI_High'].values - phase_df['Exploration_CI_Low'].values) / 2
for_means = phase_df['Foraging_Mean'].values
for_cis = (phase_df['Foraging_CI_High'].values - phase_df['Foraging_CI_Low'].values) / 2

bars1 = ax.bar(x_pos - width/2, exp_means, width, yerr=exp_cis, label='Exploration', capsize=8,
               color='#2ecc71', alpha=0.8, error_kw={'linewidth': 2})
bars2 = ax.bar(x_pos + width/2, for_means, width, yerr=for_cis, label='Foraging', capsize=8,
               color='#f39c12', alpha=0.8, error_kw={'linewidth': 2})

ax.set_xlabel('Time Lag (ms)', fontsize=12, fontweight='bold')
ax.set_ylabel('Mean Cross-Correlation ± 95% CI', fontsize=12, fontweight='bold')
ax.set_title('LHA-LHA Network: Phase Comparison (Exploration vs Foraging)', fontsize=14, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(phase_df['Lag_ms'].values.astype(int))
ax.legend(fontsize=11, loc='upper left')
ax.grid(True, alpha=0.3, axis='y', linestyle='--')
ax.set_ylim(0, max(for_means + for_cis) * 1.3)

# Add p-values explicitly
for i, row in phase_df.iterrows():
    pval = row['MannWhitney_Pvalue']
    cohens = row['Cohens_d']

    y_pos = max(exp_means[i] + exp_cis[i], for_means[i] + for_cis[i]) + 0.01

    # Format p-value
    if pval < 0.0001:
        pval_str = "p < 0.0001***"
    elif pval < 0.001:
        pval_str = f"p = {pval:.4f}***"
    elif pval < 0.01:
        pval_str = f"p = {pval:.4f}**"
    elif pval < 0.05:
        pval_str = f"p = {pval:.4f}*"
    else:
        pval_str = f"p = {pval:.4f} ns"

    ax.text(i, y_pos, pval_str, ha='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', alpha=0.7))
    ax.text(i, y_pos + 0.015, f"d = {cohens:.3f}", ha='center', fontsize=9, style='italic')

plt.tight_layout()
fig2_file = Path("figures/lha_lha_phase_comparison_with_pvalues.png")
plt.savefig(fig2_file, dpi=150, bbox_inches='tight')
print(f"[OK] Saved: {fig2_file}\n")
plt.close()

# ============================================================================
# FIGURE 3: Summary Table Figure with All Statistics
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# State comparison table
ax = axes[0]
ax.axis('tight')
ax.axis('off')

state_table_data = []
state_table_data.append(['Lag (ms)', 'Fed Mean ± CI', 'Fasted Mean ± CI', 'p-value', "Cohen's d"])

for i, row in state_df.iterrows():
    lag = int(row['Lag_ms'])
    fed_str = f"{row['Fed_Mean']:.4f} ± {(row['Fed_CI_High']-row['Fed_CI_Low'])/2:.4f}"
    fasted_str = f"{row['Fasted_Mean']:.4f} ± {(row['Fasted_CI_High']-row['Fasted_CI_Low'])/2:.4f}"

    pval = row['MannWhitney_Pvalue']
    if pval < 0.0001:
        pval_str = "p < 0.0001***"
    elif pval < 0.001:
        pval_str = f"{pval:.2e}***"
    else:
        pval_str = f"{pval:.4f}***"

    d = row['Cohens_d']

    state_table_data.append([str(lag), fed_str, fasted_str, pval_str, f"{d:.3f}"])

table1 = ax.table(cellText=state_table_data, cellLoc='center', loc='center',
                  colWidths=[0.1, 0.25, 0.25, 0.2, 0.15])
table1.auto_set_font_size(False)
table1.set_fontsize(10)
table1.scale(1, 2.5)

# Header formatting
for i in range(5):
    table1[(0, i)].set_facecolor('#3498db')
    table1[(0, i)].set_text_props(weight='bold', color='white')

# Alternate row colors
for i in range(1, len(state_table_data)):
    for j in range(5):
        if i % 2 == 0:
            table1[(i, j)].set_facecolor('#ecf0f1')
        else:
            table1[(i, j)].set_facecolor('white')

ax.set_title('State Comparison: Fed vs Fasted\n(Mann-Whitney U test, 95% CI)',
             fontsize=12, fontweight='bold', pad=20)

# Phase comparison table
ax = axes[1]
ax.axis('tight')
ax.axis('off')

phase_table_data = []
phase_table_data.append(['Lag (ms)', 'Exploration Mean ± CI', 'Foraging Mean ± CI', 'p-value', "Cohen's d"])

for i, row in phase_df.iterrows():
    lag = int(row['Lag_ms'])
    exp_str = f"{row['Exploration_Mean']:.4f} ± {(row['Exploration_CI_High']-row['Exploration_CI_Low'])/2:.4f}"
    for_str = f"{row['Foraging_Mean']:.4f} ± {(row['Foraging_CI_High']-row['Foraging_CI_Low'])/2:.4f}"

    pval = row['MannWhitney_Pvalue']
    if pval < 0.0001:
        pval_str = "p < 0.0001***"
    elif pval < 0.001:
        pval_str = f"{pval:.2e}***"
    elif pval < 0.05:
        pval_str = f"{pval:.4f}*"
    else:
        pval_str = f"{pval:.4f} ns"

    d = row['Cohens_d']

    phase_table_data.append([str(lag), exp_str, for_str, pval_str, f"{d:.3f}"])

table2 = ax.table(cellText=phase_table_data, cellLoc='center', loc='center',
                  colWidths=[0.1, 0.25, 0.25, 0.2, 0.15])
table2.auto_set_font_size(False)
table2.set_fontsize(10)
table2.scale(1, 2.5)

# Header formatting
for i in range(5):
    table2[(0, i)].set_facecolor('#2ecc71')
    table2[(0, i)].set_text_props(weight='bold', color='white')

# Alternate row colors
for i in range(1, len(phase_table_data)):
    for j in range(5):
        if i % 2 == 0:
            table2[(i, j)].set_facecolor('#ecf0f1')
        else:
            table2[(i, j)].set_facecolor('white')

ax.set_title('Phase Comparison: Exploration vs Foraging\n(Mann-Whitney U test, 95% CI)',
             fontsize=12, fontweight='bold', pad=20)

plt.tight_layout()
fig3_file = Path("figures/lha_lha_statistical_summary_tables.png")
plt.savefig(fig3_file, dpi=150, bbox_inches='tight')
print(f"[OK] Saved: {fig3_file}\n")
plt.close()

# ============================================================================
# FIGURE 4: Lag Comparison with Explicit P-values from Kruskal-Wallis
# ============================================================================

fig, ax = plt.subplots(figsize=(12, 7))

lag_means = lag_df['Mean'].values
lag_cis = (lag_df['CI_High'].values - lag_df['CI_Low'].values) / 2

bars = ax.bar(lag_df['Lag_ms'].values, lag_means, yerr=lag_cis, capsize=10,
              color=['#9b59b6', '#8e44ad', '#7d3c98'], alpha=0.8, error_kw={'linewidth': 2},
              width=20)

ax.set_xlabel('Time Lag (ms)', fontsize=12, fontweight='bold')
ax.set_ylabel('Mean Cross-Correlation ± 95% CI', fontsize=12, fontweight='bold')
ax.set_title('LHA-LHA Network: Lag Comparison (Kruskal-Wallis Test: H=1243.56, p<0.0001***)',
             fontsize=14, fontweight='bold')
ax.set_xticks([10, 50, 100])
ax.grid(True, alpha=0.3, axis='y', linestyle='--')
ax.set_ylim(0, max(lag_means + lag_cis) * 1.4)

# Add annotations showing that 10ms and 50ms are not different, but both differ from 100ms
y_top = max(lag_means + lag_cis) + 0.02

# Draw bracket showing 10ms = 50ms
ax.plot([10, 50], [y_top + 0.01, y_top + 0.01], 'k-', linewidth=2)
ax.text(30, y_top + 0.015, 'ns', ha='center', fontsize=11, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.7))

# Draw brackets showing 10ms >> 100ms and 50ms >> 100ms
ax.plot([10, 100], [y_top + 0.05, y_top + 0.05], 'k-', linewidth=2)
ax.text(55, y_top + 0.055, 'p<0.0001***', ha='center', fontsize=11, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

ax.plot([50, 100], [y_top + 0.08, y_top + 0.08], 'k-', linewidth=2)
ax.text(75, y_top + 0.085, 'p<0.0001***', ha='center', fontsize=11, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

plt.tight_layout()
fig4_file = Path("figures/lha_lha_lag_comparison_with_pvalues.png")
plt.savefig(fig4_file, dpi=150, bbox_inches='tight')
print(f"[OK] Saved: {fig4_file}\n")
plt.close()

print("="*70)
print("ALL FIGURES GENERATED WITH EXPLICIT P-VALUES")
print("="*70)
print("\nFigures created:")
print(f"  1. {fig1_file}")
print(f"  2. {fig2_file}")
print(f"  3. {fig3_file}")
print(f"  4. {fig4_file}")
print("\n[DONE]")
