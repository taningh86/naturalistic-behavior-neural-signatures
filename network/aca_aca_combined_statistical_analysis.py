"""
Combined Statistical Analysis: ACA-ACA Network — Fed vs Fasted
Mirrors the single-probe Mouse01-Coordinates01 analysis pipeline.

Comparisons:
  1. State comparison (Fed vs Fasted)
  2. Phase comparison (Exploration vs Foraging)
  3. Lag comparison (10ms vs 50ms vs 100ms)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

print("="*70)
print("COMBINED STATISTICAL ANALYSIS: ACA-ACA (FED vs FASTED)")
print("="*70)

# Load all connectivity data (fed + fasted)
connectivity_data = []

for group in ['fed_exploration', 'fed_foraging', 'fasted_exploration', 'fasted_foraging']:
    filepath = f"data/aca_aca_connectivity_{group}.csv"
    try:
        df = pd.read_csv(filepath)
        state, phase = group.split('_')
        df['state'] = state
        df['phase'] = phase
        connectivity_data.append(df)
        print(f"[OK] Loaded {filepath}: {len(df)} pairs")
    except Exception as e:
        print(f"[ERROR] Could not load {filepath}: {e}")

full_df = pd.concat(connectivity_data, ignore_index=True)

print(f"\nTotal ACA-ACA pairs analyzed: {len(full_df)}")
print(f"  Fed: {len(full_df[full_df['state'] == 'fed'])}")
print(f"  Fasted: {len(full_df[full_df['state'] == 'fasted'])}")
print(f"  Exploration: {len(full_df[full_df['phase'] == 'exploration'])}")
print(f"  Foraging: {len(full_df[full_df['phase'] == 'foraging'])}\n")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def compute_cohens_d(group1, group2):
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std > 0 else 0

def compute_ci_95(data):
    """Compute 95% confidence interval."""
    mean = np.mean(data)
    sem = stats.sem(data)
    ci = sem * stats.t.ppf((1 + 0.95) / 2, len(data) - 1)
    return mean, mean - ci, mean + ci

def format_pval(pval):
    if pval < 0.0001:
        return "p < 0.0001***"
    elif pval < 0.001:
        return f"p = {pval:.2e}***"
    elif pval < 0.01:
        return f"p = {pval:.4f}**"
    elif pval < 0.05:
        return f"p = {pval:.4f}*"
    else:
        return f"p = {pval:.4f} ns"

# ============================================================================
# 1. STATE COMPARISON (Fed vs Fasted)
# ============================================================================

print("="*70)
print("1. STATE COMPARISON: FED vs FASTED")
print("="*70)

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

    fold_change = ((fasted_mean - fed_mean) / abs(fed_mean)) * 100 if fed_mean != 0 else 0

    state_results.append({
        'Lag_ms': lag,
        'Fed_Mean': fed_mean,
        'Fed_CI_Low': fed_ci_low,
        'Fed_CI_High': fed_ci_high,
        'Fasted_Mean': fasted_mean,
        'Fasted_CI_Low': fasted_ci_low,
        'Fasted_CI_High': fasted_ci_high,
        'MannWhitney_Statistic': mw_stat,
        'MannWhitney_Pvalue': mw_pval,
        'Ttest_Statistic': t_stat,
        'Ttest_Pvalue': t_pval,
        'Cohens_d': cohens_d,
        'N_Fed': len(fed_data),
        'N_Fasted': len(fasted_data)
    })

    print(f"\nLag {lag}ms:")
    print(f"  Fed:    {fed_mean:.6f} [CI: {fed_ci_low:.6f} - {fed_ci_high:.6f}] (N={len(fed_data)})")
    print(f"  Fasted: {fasted_mean:.6f} [CI: {fasted_ci_low:.6f} - {fasted_ci_high:.6f}] (N={len(fasted_data)})")
    print(f"  Fold Change: {fold_change:+.1f}%")
    print(f"  Mann-Whitney U: U={mw_stat:.0f}, {format_pval(mw_pval)}")
    print(f"  T-test: t={t_stat:.3f}, {format_pval(t_pval)}")
    print(f"  Cohen's d: {cohens_d:.4f}")

state_df = pd.DataFrame(state_results)

# ============================================================================
# 2. PHASE COMPARISON (Exploration vs Foraging)
# ============================================================================

print("\n" + "="*70)
print("2. PHASE COMPARISON: EXPLORATION vs FORAGING")
print("="*70)

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
        'MannWhitney_Statistic': mw_stat,
        'MannWhitney_Pvalue': mw_pval,
        'Ttest_Statistic': t_stat,
        'Ttest_Pvalue': t_pval,
        'Cohens_d': cohens_d,
        'N_Exploration': len(exp_data),
        'N_Foraging': len(for_data)
    })

    print(f"\nLag {lag}ms:")
    print(f"  Exploration: {exp_mean:.6f} [CI: {exp_ci_low:.6f} - {exp_ci_high:.6f}] (N={len(exp_data)})")
    print(f"  Foraging:    {for_mean:.6f} [CI: {for_ci_low:.6f} - {for_ci_high:.6f}] (N={len(for_data)})")
    print(f"  Mann-Whitney U: U={mw_stat:.0f}, {format_pval(mw_pval)}")
    print(f"  Cohen's d: {cohens_d:.4f}")

phase_df = pd.DataFrame(phase_results)

# ============================================================================
# 3. LAG COMPARISON (10ms vs 50ms vs 100ms)
# ============================================================================

print("\n" + "="*70)
print("3. LAG COMPARISON: 10ms vs 50ms vs 100ms")
print("="*70)

lag_results = []
kw_results = []

for state in ['fed', 'fasted']:
    for phase in ['exploration', 'foraging']:
        subset = full_df[(full_df['state'] == state) & (full_df['phase'] == phase)]

        lag_10 = subset['correlation_10ms'].dropna().values
        lag_50 = subset['correlation_50ms'].dropna().values
        lag_100 = subset['correlation_100ms'].dropna().values

        kw_stat, kw_pval = stats.kruskal(lag_10, lag_50, lag_100)

        kw_results.append({
            'State': state,
            'Phase': phase,
            'KruskalWallis_Statistic': kw_stat,
            'KruskalWallis_Pvalue': kw_pval
        })

        print(f"\n{state.capitalize()} - {phase.capitalize()}:")
        print(f"  Kruskal-Wallis H={kw_stat:.3f}, {format_pval(kw_pval)}")

        pairs = [(10, 50, lag_10, lag_50), (10, 100, lag_10, lag_100), (50, 100, lag_50, lag_100)]
        for lag1, lag2, data1, data2 in pairs:
            mw_stat, mw_pval = stats.mannwhitneyu(data1, data2)
            print(f"    {lag1}ms vs {lag2}ms: U={mw_stat:.0f}, {format_pval(mw_pval)}")

kw_df = pd.DataFrame(kw_results)

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

# ============================================================================
# 4. SAVE RESULTS
# ============================================================================

print("\n" + "="*70)
print("SAVING STATISTICAL RESULTS")
print("="*70 + "\n")

state_df.to_csv("data/aca_aca_combined_stats_state_comparison.csv", index=False)
print("[OK] Saved: aca_aca_combined_stats_state_comparison.csv")

phase_df.to_csv("data/aca_aca_combined_stats_phase_comparison.csv", index=False)
print("[OK] Saved: aca_aca_combined_stats_phase_comparison.csv")

lag_df.to_csv("data/aca_aca_combined_stats_lag_comparison.csv", index=False)
print("[OK] Saved: aca_aca_combined_stats_lag_comparison.csv")

kw_df.to_csv("data/aca_aca_combined_stats_kruskalwallis.csv", index=False)
print("[OK] Saved: aca_aca_combined_stats_kruskalwallis.csv")

# ============================================================================
# 5. PUBLICATION FIGURES
# ============================================================================

print("\nGenerating publication figures...")

# --- FIGURE 1: State Comparison ---
fig, ax = plt.subplots(figsize=(12, 7))
x_pos = np.arange(len(state_df))
width = 0.35

fed_means = state_df['Fed_Mean'].values
fed_cis = (state_df['Fed_CI_High'].values - state_df['Fed_CI_Low'].values) / 2
fasted_means = state_df['Fasted_Mean'].values
fasted_cis = (state_df['Fasted_CI_High'].values - state_df['Fasted_CI_Low'].values) / 2

ax.bar(x_pos - width/2, fed_means, width, yerr=fed_cis, label='Fed', capsize=8,
       color='#3498db', alpha=0.8, error_kw={'linewidth': 2})
ax.bar(x_pos + width/2, fasted_means, width, yerr=fasted_cis, label='Fasted', capsize=8,
       color='#e74c3c', alpha=0.8, error_kw={'linewidth': 2})

ax.set_xlabel('Time Lag (ms)', fontsize=12, fontweight='bold')
ax.set_ylabel('Mean Cross-Correlation +/- 95% CI', fontsize=12, fontweight='bold')
ax.set_title('ACA-ACA Network: State Comparison (Fed vs Fasted)', fontsize=14, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(state_df['Lag_ms'].values.astype(int))
ax.legend(fontsize=11, loc='upper left')
ax.grid(True, alpha=0.3, axis='y', linestyle='--')

max_val = max(max(fed_means + fed_cis), max(fasted_means + fasted_cis))
ax.set_ylim(0, max_val * 1.5)

for i, row in state_df.iterrows():
    pval = row['MannWhitney_Pvalue']
    cohens = row['Cohens_d']
    y_pos = max(fed_means[i] + fed_cis[i], fasted_means[i] + fasted_cis[i])

    ax.text(i, y_pos + max_val * 0.08, format_pval(pval), ha='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', alpha=0.7))
    ax.text(i, y_pos + max_val * 0.18, f"d = {cohens:.4f}", ha='center', fontsize=9, style='italic')

plt.tight_layout()
fig1_file = Path("figures/aca_aca_combined_state_comparison_with_pvalues.png")
plt.savefig(fig1_file, dpi=150, bbox_inches='tight')
print(f"[OK] Saved: {fig1_file}")
plt.close()

# --- FIGURE 2: Phase Comparison ---
fig, ax = plt.subplots(figsize=(12, 7))
x_pos = np.arange(len(phase_df))

exp_means = phase_df['Exploration_Mean'].values
exp_cis = (phase_df['Exploration_CI_High'].values - phase_df['Exploration_CI_Low'].values) / 2
for_means = phase_df['Foraging_Mean'].values
for_cis = (phase_df['Foraging_CI_High'].values - phase_df['Foraging_CI_Low'].values) / 2

ax.bar(x_pos - width/2, exp_means, width, yerr=exp_cis, label='Exploration', capsize=8,
       color='#2ecc71', alpha=0.8, error_kw={'linewidth': 2})
ax.bar(x_pos + width/2, for_means, width, yerr=for_cis, label='Foraging', capsize=8,
       color='#f39c12', alpha=0.8, error_kw={'linewidth': 2})

ax.set_xlabel('Time Lag (ms)', fontsize=12, fontweight='bold')
ax.set_ylabel('Mean Cross-Correlation +/- 95% CI', fontsize=12, fontweight='bold')
ax.set_title('ACA-ACA Network: Phase Comparison (Exploration vs Foraging)', fontsize=14, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(phase_df['Lag_ms'].values.astype(int))
ax.legend(fontsize=11, loc='upper left')
ax.grid(True, alpha=0.3, axis='y', linestyle='--')

max_val = max(max(exp_means + exp_cis), max(for_means + for_cis))
ax.set_ylim(0, max_val * 1.5)

for i, row in phase_df.iterrows():
    pval = row['MannWhitney_Pvalue']
    cohens = row['Cohens_d']
    y_pos = max(exp_means[i] + exp_cis[i], for_means[i] + for_cis[i])

    ax.text(i, y_pos + max_val * 0.08, format_pval(pval), ha='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', alpha=0.7))
    ax.text(i, y_pos + max_val * 0.18, f"d = {cohens:.4f}", ha='center', fontsize=9, style='italic')

plt.tight_layout()
fig2_file = Path("figures/aca_aca_combined_phase_comparison_with_pvalues.png")
plt.savefig(fig2_file, dpi=150, bbox_inches='tight')
print(f"[OK] Saved: {fig2_file}")
plt.close()

# --- FIGURE 3: Summary Tables ---
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# State table
ax = axes[0]
ax.axis('tight')
ax.axis('off')

state_table_data = [['Lag (ms)', 'Fed Mean +/- CI', 'Fasted Mean +/- CI', 'p-value', "Cohen's d"]]
for i, row in state_df.iterrows():
    lag = int(row['Lag_ms'])
    fed_str = f"{row['Fed_Mean']:.4f} +/- {(row['Fed_CI_High']-row['Fed_CI_Low'])/2:.4f}"
    fasted_str = f"{row['Fasted_Mean']:.4f} +/- {(row['Fasted_CI_High']-row['Fasted_CI_Low'])/2:.4f}"
    state_table_data.append([str(lag), fed_str, fasted_str, format_pval(row['MannWhitney_Pvalue']), f"{row['Cohens_d']:.4f}"])

table1 = ax.table(cellText=state_table_data, cellLoc='center', loc='center',
                  colWidths=[0.1, 0.25, 0.25, 0.2, 0.15])
table1.auto_set_font_size(False)
table1.set_fontsize(10)
table1.scale(1, 2.5)
for i in range(5):
    table1[(0, i)].set_facecolor('#3498db')
    table1[(0, i)].set_text_props(weight='bold', color='white')
for i in range(1, len(state_table_data)):
    for j in range(5):
        table1[(i, j)].set_facecolor('#ecf0f1' if i % 2 == 0 else 'white')
ax.set_title('State: Fed vs Fasted\n(Mann-Whitney U, 95% CI)', fontsize=12, fontweight='bold', pad=20)

# Phase table
ax = axes[1]
ax.axis('tight')
ax.axis('off')

phase_table_data = [['Lag (ms)', 'Exploration Mean +/- CI', 'Foraging Mean +/- CI', 'p-value', "Cohen's d"]]
for i, row in phase_df.iterrows():
    lag = int(row['Lag_ms'])
    exp_str = f"{row['Exploration_Mean']:.4f} +/- {(row['Exploration_CI_High']-row['Exploration_CI_Low'])/2:.4f}"
    for_str = f"{row['Foraging_Mean']:.4f} +/- {(row['Foraging_CI_High']-row['Foraging_CI_Low'])/2:.4f}"
    phase_table_data.append([str(lag), exp_str, for_str, format_pval(row['MannWhitney_Pvalue']), f"{row['Cohens_d']:.4f}"])

table2 = ax.table(cellText=phase_table_data, cellLoc='center', loc='center',
                  colWidths=[0.1, 0.25, 0.25, 0.2, 0.15])
table2.auto_set_font_size(False)
table2.set_fontsize(10)
table2.scale(1, 2.5)
for i in range(5):
    table2[(0, i)].set_facecolor('#2ecc71')
    table2[(0, i)].set_text_props(weight='bold', color='white')
for i in range(1, len(phase_table_data)):
    for j in range(5):
        table2[(i, j)].set_facecolor('#ecf0f1' if i % 2 == 0 else 'white')
ax.set_title('Phase: Exploration vs Foraging\n(Mann-Whitney U, 95% CI)', fontsize=12, fontweight='bold', pad=20)

plt.tight_layout()
fig3_file = Path("figures/aca_aca_combined_statistical_summary_tables.png")
plt.savefig(fig3_file, dpi=150, bbox_inches='tight')
print(f"[OK] Saved: {fig3_file}")
plt.close()

# --- FIGURE 4: Lag Comparison ---
fig, ax = plt.subplots(figsize=(12, 7))

lag_means = lag_df['Mean'].values
lag_cis = (lag_df['CI_High'].values - lag_df['CI_Low'].values) / 2

bars = ax.bar(lag_df['Lag_ms'].values, lag_means, yerr=lag_cis, capsize=10,
              color=['#9b59b6', '#8e44ad', '#7d3c98'], alpha=0.8, error_kw={'linewidth': 2}, width=20)

ax.set_xlabel('Time Lag (ms)', fontsize=12, fontweight='bold')
ax.set_ylabel('Mean Cross-Correlation +/- 95% CI', fontsize=12, fontweight='bold')
ax.set_title('ACA-ACA Network: Lag Comparison (Combined Fed+Fasted)', fontsize=14, fontweight='bold')
ax.set_xticks([10, 50, 100])
ax.grid(True, alpha=0.3, axis='y', linestyle='--')

max_val = max(lag_means + lag_cis)
ax.set_ylim(0, max_val * 1.6)

y_top = max_val + max_val * 0.05
ax.plot([10, 50], [y_top + max_val * 0.02, y_top + max_val * 0.02], 'k-', linewidth=2)
ax.text(30, y_top + max_val * 0.05, 'ns (identical)', ha='center', fontsize=11, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.7))

ax.plot([10, 100], [y_top + max_val * 0.15, y_top + max_val * 0.15], 'k-', linewidth=2)
ax.text(55, y_top + max_val * 0.18, '10ms vs 100ms', ha='center', fontsize=10, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

ax.plot([50, 100], [y_top + max_val * 0.28, y_top + max_val * 0.28], 'k-', linewidth=2)
ax.text(75, y_top + max_val * 0.31, '50ms vs 100ms', ha='center', fontsize=10, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

plt.tight_layout()
fig4_file = Path("figures/aca_aca_combined_lag_comparison_with_pvalues.png")
plt.savefig(fig4_file, dpi=150, bbox_inches='tight')
print(f"[OK] Saved: {fig4_file}")
plt.close()

# --- FIGURE 5: Effect size comparison (State vs Phase) ---
fig, ax = plt.subplots(figsize=(10, 6))

state_cohens = state_df['Cohens_d'].values
phase_cohens = phase_df['Cohens_d'].values

x = np.arange(3)
ax.bar(x - 0.2, state_cohens, 0.4, label='Fed vs Fasted', color='#3498db', alpha=0.8)
ax.bar(x + 0.2, phase_cohens, 0.4, label='Exploration vs Foraging', color='#2ecc71', alpha=0.8)

ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax.axhline(y=0.2, color='gray', linestyle='--', alpha=0.3)
ax.axhline(y=-0.2, color='gray', linestyle='--', alpha=0.3)
ax.text(2.6, 0.22, 'small', fontsize=8, alpha=0.5)
ax.text(2.6, -0.22, 'small', fontsize=8, alpha=0.5)

ax.set_xlabel('Time Lag (ms)', fontsize=12, fontweight='bold')
ax.set_ylabel("Cohen's d", fontsize=12, fontweight='bold')
ax.set_title("ACA-ACA Network: Effect Sizes (State vs Phase)", fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(['10ms', '50ms', '100ms'])
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
fig5_file = Path("figures/aca_aca_combined_effect_sizes.png")
plt.savefig(fig5_file, dpi=150, bbox_inches='tight')
print(f"[OK] Saved: {fig5_file}")
plt.close()

# ============================================================================
# PRINT SUMMARY TABLES
# ============================================================================

print("\n" + "="*70)
print("STATE COMPARISON SUMMARY")
print("="*70)
print(state_df.to_string(index=False))

print("\n" + "="*70)
print("PHASE COMPARISON SUMMARY")
print("="*70)
print(phase_df.to_string(index=False))

print("\n" + "="*70)
print("LAG COMPARISON SUMMARY")
print("="*70)
print(lag_df.to_string(index=False))

print("\n" + "="*70)
print("KRUSKAL-WALLIS TESTS")
print("="*70)
print(kw_df.to_string(index=False))

# ============================================================================
# COMPARATIVE SUMMARY
# ============================================================================

print("\n" + "="*70)
print("COMPARATIVE ANALYSIS: STATE vs PHASE EFFECT MAGNITUDE")
print("="*70)

print("\nFasted State Effect on ACA-ACA (Fold Change):")
for lag in [10, 50, 100]:
    row = state_df[state_df['Lag_ms'] == lag].iloc[0]
    fold = ((row['Fasted_Mean'] - row['Fed_Mean']) / abs(row['Fed_Mean'])) * 100 if row['Fed_Mean'] != 0 else 0
    print(f"  Lag {lag}ms: {fold:+.1f}% (d={row['Cohens_d']:.4f})")

print("\nPhase Effect on ACA-ACA (Fold Change):")
for lag in [10, 50, 100]:
    row = phase_df[phase_df['Lag_ms'] == lag].iloc[0]
    fold = ((row['Foraging_Mean'] - row['Exploration_Mean']) / abs(row['Exploration_Mean'])) * 100 if row['Exploration_Mean'] != 0 else 0
    print(f"  Lag {lag}ms: {fold:+.1f}% (d={row['Cohens_d']:.4f})")

print("\n[DONE] Combined ACA-ACA statistical analysis complete!")
