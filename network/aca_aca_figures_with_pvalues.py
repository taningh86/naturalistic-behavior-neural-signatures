"""
Generate ACA-ACA figures with explicit p-values displayed prominently.
Dual-Probe Probe-0 (ACA) — Phase comparison and lag comparison only (all fed).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Load statistical results
phase_df = pd.read_csv("data/aca_aca_stats_phase_comparison.csv")
lag_df = pd.read_csv("data/aca_aca_stats_lag_comparison.csv")

print("Generating ACA-ACA figures with explicit p-values...\n")

# ============================================================================
# FIGURE 1: Phase Comparison with P-values
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
ax.set_ylabel('Mean Cross-Correlation +/- 95% CI', fontsize=12, fontweight='bold')
ax.set_title('ACA-ACA Network (Dual-Probe Probe-0): Phase Comparison\n(Exploration vs Foraging — All Fed)', fontsize=14, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(phase_df['Lag_ms'].values.astype(int))
ax.legend(fontsize=11, loc='upper left')
ax.grid(True, alpha=0.3, axis='y', linestyle='--')

# Set y-limit with padding for annotations
max_val = max(max(exp_means + exp_cis), max(for_means + for_cis))
ax.set_ylim(0, max_val * 1.5)

# Add p-values explicitly
for i, row in phase_df.iterrows():
    pval = row['MannWhitney_Pvalue']
    cohens = row['Cohens_d']

    y_pos = max(exp_means[i] + exp_cis[i], for_means[i] + for_cis[i])

    # Format p-value
    if pval < 0.0001:
        pval_str = "p < 0.0001***"
    elif pval < 0.001:
        pval_str = f"p = {pval:.2e}***"
    elif pval < 0.01:
        pval_str = f"p = {pval:.4f}**"
    elif pval < 0.05:
        pval_str = f"p = {pval:.4f}*"
    else:
        pval_str = f"p = {pval:.4f} ns"

    ax.text(i, y_pos + max_val * 0.08, pval_str, ha='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', alpha=0.7))
    ax.text(i, y_pos + max_val * 0.18, f"d = {cohens:.4f}", ha='center', fontsize=9, style='italic')

plt.tight_layout()
fig1_file = Path("figures/aca_aca_phase_comparison_with_pvalues.png")
plt.savefig(fig1_file, dpi=150, bbox_inches='tight')
print(f"[OK] Saved: {fig1_file}\n")
plt.close()

# ============================================================================
# FIGURE 2: Summary Table Figure
# ============================================================================

fig, ax = plt.subplots(figsize=(14, 5))
ax.axis('tight')
ax.axis('off')

phase_table_data = []
phase_table_data.append(['Lag (ms)', 'Exploration Mean +/- CI', 'Foraging Mean +/- CI', 'p-value (MW)', "Cohen's d", 'N_Exp', 'N_For'])

for i, row in phase_df.iterrows():
    lag = int(row['Lag_ms'])
    exp_str = f"{row['Exploration_Mean']:.6f} +/- {(row['Exploration_CI_High']-row['Exploration_CI_Low'])/2:.6f}"
    for_str = f"{row['Foraging_Mean']:.6f} +/- {(row['Foraging_CI_High']-row['Foraging_CI_Low'])/2:.6f}"

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

    phase_table_data.append([str(lag), exp_str, for_str, pval_str, f"{d:.4f}",
                             str(int(row['N_Exploration'])), str(int(row['N_Foraging']))])

table = ax.table(cellText=phase_table_data, cellLoc='center', loc='center',
                 colWidths=[0.08, 0.22, 0.22, 0.16, 0.1, 0.08, 0.08])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.5)

# Header formatting
for i in range(7):
    table[(0, i)].set_facecolor('#2ecc71')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Alternate row colors
for i in range(1, len(phase_table_data)):
    for j in range(7):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#ecf0f1')
        else:
            table[(i, j)].set_facecolor('white')

ax.set_title('ACA-ACA Network (Dual-Probe Probe-0): Phase Comparison Statistics\n(Mann-Whitney U test, 95% CI)',
             fontsize=13, fontweight='bold', pad=20)

plt.tight_layout()
fig2_file = Path("figures/aca_aca_statistical_summary_tables.png")
plt.savefig(fig2_file, dpi=150, bbox_inches='tight')
print(f"[OK] Saved: {fig2_file}\n")
plt.close()

# ============================================================================
# FIGURE 3: Lag Comparison with P-values
# ============================================================================

fig, ax = plt.subplots(figsize=(12, 7))

lag_means = lag_df['Mean'].values
lag_cis = (lag_df['CI_High'].values - lag_df['CI_Low'].values) / 2

bars = ax.bar(lag_df['Lag_ms'].values, lag_means, yerr=lag_cis, capsize=10,
              color=['#9b59b6', '#8e44ad', '#7d3c98'], alpha=0.8, error_kw={'linewidth': 2},
              width=20)

ax.set_xlabel('Time Lag (ms)', fontsize=12, fontweight='bold')
ax.set_ylabel('Mean Cross-Correlation +/- 95% CI', fontsize=12, fontweight='bold')
ax.set_title('ACA-ACA Network (Dual-Probe Probe-0): Lag Comparison', fontsize=14, fontweight='bold')
ax.set_xticks([10, 50, 100])
ax.grid(True, alpha=0.3, axis='y', linestyle='--')

max_val = max(lag_means + lag_cis)
ax.set_ylim(0, max_val * 1.6)

# Add annotations
y_top = max_val + max_val * 0.05

# Draw bracket showing 10ms = 50ms
ax.plot([10, 50], [y_top + max_val * 0.02, y_top + max_val * 0.02], 'k-', linewidth=2)
ax.text(30, y_top + max_val * 0.05, 'ns (identical)', ha='center', fontsize=11, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.7))

# Draw brackets showing 10ms vs 100ms and 50ms vs 100ms
ax.plot([10, 100], [y_top + max_val * 0.15, y_top + max_val * 0.15], 'k-', linewidth=2)
ax.text(55, y_top + max_val * 0.18, '10ms vs 100ms', ha='center', fontsize=10, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

ax.plot([50, 100], [y_top + max_val * 0.28, y_top + max_val * 0.28], 'k-', linewidth=2)
ax.text(75, y_top + max_val * 0.31, '50ms vs 100ms', ha='center', fontsize=10, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

plt.tight_layout()
fig3_file = Path("figures/aca_aca_lag_comparison_with_pvalues.png")
plt.savefig(fig3_file, dpi=150, bbox_inches='tight')
print(f"[OK] Saved: {fig3_file}\n")
plt.close()

print("="*70)
print("ALL ACA-ACA FIGURES GENERATED WITH EXPLICIT P-VALUES")
print("="*70)
print("\nFigures created:")
print(f"  1. {fig1_file}")
print(f"  2. {fig2_file}")
print(f"  3. {fig3_file}")
print("\n[DONE]")
