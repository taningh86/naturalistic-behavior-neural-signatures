"""
Statistical Analysis of LHA-LHA Network Data
Comprehensive statistical testing: state, phase, and lag comparisons
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
print("STATISTICAL ANALYSIS: LHA-LHA NETWORK DATA")
print("="*70)

# Load detailed connectivity data
connectivity_data = []

for group in ['fed_exploration', 'fed_foraging', 'fasted_exploration', 'fasted_foraging']:
    df = pd.read_csv(f"data/lha_lha_connectivity_{group}.csv")
    state, phase = group.split('_')
    df['state'] = state
    df['phase'] = phase
    connectivity_data.append(df)

full_df = pd.concat(connectivity_data, ignore_index=True)

print(f"\nTotal LHA-LHA pairs analyzed: {len(full_df)}")
print(f"Fed: {len(full_df[full_df['state'] == 'fed'])}")
print(f"Fasted: {len(full_df[full_df['state'] == 'fasted'])}")
print(f"Exploration: {len(full_df[full_df['phase'] == 'exploration'])}")
print(f"Foraging: {len(full_df[full_df['phase'] == 'foraging'])}\n")

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

    # Mann-Whitney U test
    mw_stat, mw_pval = stats.mannwhitneyu(fed_data, fasted_data)

    # T-test
    t_stat, t_pval = stats.ttest_ind(fed_data, fasted_data)

    # Cohen's d
    cohens_d = compute_cohens_d(fed_data, fasted_data)

    # 95% CI
    fed_mean, fed_ci_low, fed_ci_high = compute_ci_95(fed_data)
    fasted_mean, fasted_ci_low, fasted_ci_high = compute_ci_95(fasted_data)

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

    sig_marker = "***" if mw_pval < 0.001 else ("**" if mw_pval < 0.01 else ("*" if mw_pval < 0.05 else "ns"))

    print(f"\nLag {lag}ms:")
    print(f"  Fed:    {fed_mean:.4f} [CI: {fed_ci_low:.4f} - {fed_ci_high:.4f}]")
    print(f"  Fasted: {fasted_mean:.4f} [CI: {fasted_ci_low:.4f} - {fasted_ci_high:.4f}]")
    print(f"  Mann-Whitney U test: U={mw_stat:.0f}, p={mw_pval:.4f} {sig_marker}")
    print(f"  T-test: t={t_stat:.3f}, p={t_pval:.4f} {sig_marker}")
    print(f"  Cohen's d: {cohens_d:.3f}")

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

    # Mann-Whitney U test
    mw_stat, mw_pval = stats.mannwhitneyu(exp_data, for_data)

    # T-test
    t_stat, t_pval = stats.ttest_ind(exp_data, for_data)

    # Cohen's d
    cohens_d = compute_cohens_d(exp_data, for_data)

    # 95% CI
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

    sig_marker = "***" if mw_pval < 0.001 else ("**" if mw_pval < 0.01 else ("*" if mw_pval < 0.05 else "ns"))

    print(f"\nLag {lag}ms:")
    print(f"  Exploration: {exp_mean:.4f} [CI: {exp_ci_low:.4f} - {exp_ci_high:.4f}]")
    print(f"  Foraging:    {for_mean:.4f} [CI: {for_ci_low:.4f} - {for_ci_high:.4f}]")
    print(f"  Mann-Whitney U test: U={mw_stat:.0f}, p={mw_pval:.4f} {sig_marker}")
    print(f"  T-test: t={t_stat:.3f}, p={t_pval:.4f} {sig_marker}")
    print(f"  Cohen's d: {cohens_d:.3f}")

phase_df = pd.DataFrame(phase_results)

# ============================================================================
# 3. LAG COMPARISON (10ms vs 50ms vs 100ms)
# ============================================================================

print("\n" + "="*70)
print("3. LAG COMPARISON: 10ms vs 50ms vs 100ms")
print("="*70)

lag_results = []

# Kruskal-Wallis test across all lags (within each state/phase combination)
kw_results = []

for state in ['fed', 'fasted']:
    for phase in ['exploration', 'foraging']:
        subset = full_df[(full_df['state'] == state) & (full_df['phase'] == phase)]

        lag_10 = subset['correlation_10ms'].dropna().values
        lag_50 = subset['correlation_50ms'].dropna().values
        lag_100 = subset['correlation_100ms'].dropna().values

        # Kruskal-Wallis test
        kw_stat, kw_pval = stats.kruskal(lag_10, lag_50, lag_100)

        kw_results.append({
            'State': state,
            'Phase': phase,
            'KruskalWallis_Statistic': kw_stat,
            'KruskalWallis_Pvalue': kw_pval
        })

        sig_marker = "***" if kw_pval < 0.001 else ("**" if kw_pval < 0.01 else ("*" if kw_pval < 0.05 else "ns"))
        print(f"\n{state.capitalize()} - {phase.capitalize()}:")
        print(f"  Kruskal-Wallis H={kw_stat:.3f}, p={kw_pval:.4f} {sig_marker}")

        # Pairwise Mann-Whitney comparisons
        print(f"  Pairwise comparisons:")

        pairs = [(10, 50, lag_10, lag_50), (10, 100, lag_10, lag_100), (50, 100, lag_50, lag_100)]
        for lag1, lag2, data1, data2 in pairs:
            mw_stat, mw_pval = stats.mannwhitneyu(data1, data2)
            sig_marker_pw = "***" if mw_pval < 0.001 else ("**" if mw_pval < 0.01 else ("*" if mw_pval < 0.05 else "ns"))
            print(f"    {lag1}ms vs {lag2}ms: U={mw_stat:.0f}, p={mw_pval:.4f} {sig_marker_pw}")

kw_df = pd.DataFrame(kw_results)

# Compute means and CIs for each lag
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

state_df.to_csv("data/lha_lha_stats_state_comparison.csv", index=False)
print("[OK] Saved: lha_lha_stats_state_comparison.csv")

phase_df.to_csv("data/lha_lha_stats_phase_comparison.csv", index=False)
print("[OK] Saved: lha_lha_stats_phase_comparison.csv")

lag_df.to_csv("data/lha_lha_stats_lag_comparison.csv", index=False)
print("[OK] Saved: lha_lha_stats_lag_comparison.csv")

kw_df.to_csv("data/lha_lha_stats_kruskalwallis.csv", index=False)
print("[OK] Saved: lha_lha_stats_kruskalwallis.csv\n")

# ============================================================================
# 5. VISUALIZATIONS WITH CONFIDENCE INTERVALS
# ============================================================================

print("Generating visualizations with confidence intervals...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('LHA-LHA Network: Statistical Comparisons with 95% CI', fontsize=14, fontweight='bold')

# Plot 1: State comparison with CI
ax = axes[0, 0]
x_pos = np.arange(len(state_df))
width = 0.35

fed_means = state_df['Fed_Mean'].values
fed_cis = (state_df['Fed_CI_High'].values - state_df['Fed_CI_Low'].values) / 2
fasted_means = state_df['Fasted_Mean'].values
fasted_cis = (state_df['Fasted_CI_High'].values - state_df['Fasted_CI_Low'].values) / 2

ax.bar(x_pos - width/2, fed_means, width, yerr=fed_cis, label='Fed', capsize=5, color='blue', alpha=0.7)
ax.bar(x_pos + width/2, fasted_means, width, yerr=fasted_cis, label='Fasted', capsize=5, color='red', alpha=0.7)
ax.set_xlabel('Time Lag (ms)')
ax.set_ylabel('Mean Correlation')
ax.set_title('State Comparison (Fed vs Fasted)')
ax.set_xticks(x_pos)
ax.set_xticklabels(state_df['Lag_ms'].values)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Add p-values
for i, row in state_df.iterrows():
    if row['MannWhitney_Pvalue'] < 0.05:
        ax.text(i, max(fed_means[i], fasted_means[i]) + max(fed_cis[i], fasted_cis[i]) + 0.01,
                f"p={row['MannWhitney_Pvalue']:.3f}", ha='center', fontsize=9, fontweight='bold')

# Plot 2: Phase comparison with CI
ax = axes[0, 1]
exp_means = phase_df['Exploration_Mean'].values
exp_cis = (phase_df['Exploration_CI_High'].values - phase_df['Exploration_CI_Low'].values) / 2
for_means = phase_df['Foraging_Mean'].values
for_cis = (phase_df['Foraging_CI_High'].values - phase_df['Foraging_CI_Low'].values) / 2

ax.bar(x_pos - width/2, exp_means, width, yerr=exp_cis, label='Exploration', capsize=5, color='green', alpha=0.7)
ax.bar(x_pos + width/2, for_means, width, yerr=for_cis, label='Foraging', capsize=5, color='orange', alpha=0.7)
ax.set_xlabel('Time Lag (ms)')
ax.set_ylabel('Mean Correlation')
ax.set_title('Phase Comparison (Exploration vs Foraging)')
ax.set_xticks(x_pos)
ax.set_xticklabels(phase_df['Lag_ms'].values)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Add p-values
for i, row in phase_df.iterrows():
    if row['MannWhitney_Pvalue'] < 0.05:
        ax.text(i, max(exp_means[i], for_means[i]) + max(exp_cis[i], for_cis[i]) + 0.01,
                f"p={row['MannWhitney_Pvalue']:.3f}", ha='center', fontsize=9, fontweight='bold')

# Plot 3: Lag comparison with CI
ax = axes[1, 0]
lag_means = lag_df['Mean'].values
lag_cis = (lag_df['CI_High'].values - lag_df['CI_Low'].values) / 2

ax.bar(lag_df['Lag_ms'].values, lag_means, yerr=lag_cis, capsize=5, color='purple', alpha=0.7, width=15)
ax.set_xlabel('Time Lag (ms)')
ax.set_ylabel('Mean Correlation')
ax.set_title('Lag Comparison (Across All Data)')
ax.set_xticks([10, 50, 100])
ax.grid(True, alpha=0.3, axis='y')

# Plot 4: Effect sizes (Cohen's d)
ax = axes[1, 1]

# State comparisons
state_cohens = state_df['Cohens_d'].values
ax.bar(np.arange(3) - 0.2, state_cohens, 0.4, label='Fed vs Fasted', color='blue', alpha=0.7)

# Phase comparisons
phase_cohens = phase_df['Cohens_d'].values
ax.bar(np.arange(3) + 0.2, phase_cohens, 0.4, label='Exploration vs Foraging', color='green', alpha=0.7)

ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax.set_xlabel('Time Lag (ms)')
ax.set_ylabel("Cohen's d")
ax.set_title("Effect Sizes: State and Phase Comparisons")
ax.set_xticks(np.arange(3))
ax.set_xticklabels(['10ms', '50ms', '100ms'])
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Add effect size interpretation lines
ax.axhline(y=0.2, color='gray', linestyle='--', alpha=0.3, linewidth=1)
ax.axhline(y=-0.2, color='gray', linestyle='--', alpha=0.3, linewidth=1)
ax.text(2.5, 0.22, 'small', fontsize=8, alpha=0.5)
ax.text(2.5, -0.22, 'small', fontsize=8, alpha=0.5)

plt.tight_layout()
fig_file = Path("figures/lha_lha_statistical_analysis.png")
plt.savefig(fig_file, dpi=150, bbox_inches='tight')
print(f"[OK] Saved figure to: {fig_file}\n")
plt.close()

# ============================================================================
# PRINT SUMMARY TABLES
# ============================================================================

print("="*70)
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
print("KRUSKAL-WALLIS TESTS (Lag Comparison)")
print("="*70)
print(kw_df.to_string(index=False))

print("\n[DONE] Statistical analysis complete!")
