"""
Statistical Analysis of ACA-ACA Network Data (Dual-Probe Probe-0)
Phase comparison (exploration vs foraging) and lag comparisons.
No state comparison — all sessions are fed.
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
print("STATISTICAL ANALYSIS: ACA-ACA NETWORK DATA (DUAL-PROBE PROBE-0)")
print("="*70)

# Load detailed connectivity data — only fed groups exist
connectivity_data = []

for group in ['fed_exploration', 'fed_foraging']:
    df = pd.read_csv(f"data/aca_aca_connectivity_{group}.csv")
    state, phase = group.split('_')
    df['state'] = state
    df['phase'] = phase
    connectivity_data.append(df)

full_df = pd.concat(connectivity_data, ignore_index=True)

print(f"\nTotal ACA-ACA pairs analyzed: {len(full_df)}")
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
# 1. PHASE COMPARISON (Exploration vs Foraging)
# ============================================================================

print("="*70)
print("1. PHASE COMPARISON: EXPLORATION vs FORAGING")
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
    print(f"  Exploration: {exp_mean:.6f} [CI: {exp_ci_low:.6f} - {exp_ci_high:.6f}]")
    print(f"  Foraging:    {for_mean:.6f} [CI: {for_ci_low:.6f} - {for_ci_high:.6f}]")
    print(f"  Mann-Whitney U test: U={mw_stat:.0f}, p={mw_pval:.2e} {sig_marker}")
    print(f"  T-test: t={t_stat:.3f}, p={t_pval:.2e} {sig_marker}")
    print(f"  Cohen's d: {cohens_d:.4f}")

phase_df = pd.DataFrame(phase_results)

# ============================================================================
# 2. LAG COMPARISON (10ms vs 50ms vs 100ms)
# ============================================================================

print("\n" + "="*70)
print("2. LAG COMPARISON: 10ms vs 50ms vs 100ms")
print("="*70)

lag_results = []

# Kruskal-Wallis test across all lags (within each phase)
kw_results = []

for phase in ['exploration', 'foraging']:
    subset = full_df[full_df['phase'] == phase]

    lag_10 = subset['correlation_10ms'].dropna().values
    lag_50 = subset['correlation_50ms'].dropna().values
    lag_100 = subset['correlation_100ms'].dropna().values

    # Kruskal-Wallis test
    kw_stat, kw_pval = stats.kruskal(lag_10, lag_50, lag_100)

    kw_results.append({
        'Phase': phase,
        'KruskalWallis_Statistic': kw_stat,
        'KruskalWallis_Pvalue': kw_pval
    })

    sig_marker = "***" if kw_pval < 0.001 else ("**" if kw_pval < 0.01 else ("*" if kw_pval < 0.05 else "ns"))
    print(f"\n{phase.capitalize()}:")
    print(f"  Kruskal-Wallis H={kw_stat:.3f}, p={kw_pval:.2e} {sig_marker}")

    # Pairwise Mann-Whitney comparisons
    print(f"  Pairwise comparisons:")

    pairs = [(10, 50, lag_10, lag_50), (10, 100, lag_10, lag_100), (50, 100, lag_50, lag_100)]
    for lag1, lag2, data1, data2 in pairs:
        mw_stat, mw_pval = stats.mannwhitneyu(data1, data2)
        sig_marker_pw = "***" if mw_pval < 0.001 else ("**" if mw_pval < 0.01 else ("*" if mw_pval < 0.05 else "ns"))
        print(f"    {lag1}ms vs {lag2}ms: U={mw_stat:.0f}, p={mw_pval:.2e} {sig_marker_pw}")

kw_df = pd.DataFrame(kw_results)

# Compute means and CIs for each lag (across all data)
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
# 3. SAVE RESULTS
# ============================================================================

print("\n" + "="*70)
print("SAVING STATISTICAL RESULTS")
print("="*70 + "\n")

phase_df.to_csv("data/aca_aca_stats_phase_comparison.csv", index=False)
print("[OK] Saved: aca_aca_stats_phase_comparison.csv")

lag_df.to_csv("data/aca_aca_stats_lag_comparison.csv", index=False)
print("[OK] Saved: aca_aca_stats_lag_comparison.csv")

kw_df.to_csv("data/aca_aca_stats_kruskalwallis.csv", index=False)
print("[OK] Saved: aca_aca_stats_kruskalwallis.csv")

# ============================================================================
# 4. VISUALIZATIONS WITH CONFIDENCE INTERVALS
# ============================================================================

print("\nGenerating visualizations with confidence intervals...")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('ACA-ACA Network (Dual-Probe Probe-0): Statistical Comparisons with 95% CI', fontsize=13, fontweight='bold')

# Plot 1: Phase comparison with CI
ax = axes[0]
x_pos = np.arange(len(phase_df))
width = 0.35

exp_means = phase_df['Exploration_Mean'].values
exp_cis = (phase_df['Exploration_CI_High'].values - phase_df['Exploration_CI_Low'].values) / 2
for_means = phase_df['Foraging_Mean'].values
for_cis = (phase_df['Foraging_CI_High'].values - phase_df['Foraging_CI_Low'].values) / 2

ax.bar(x_pos - width/2, exp_means, width, yerr=exp_cis, label='Exploration', capsize=5, color='#2ecc71', alpha=0.8)
ax.bar(x_pos + width/2, for_means, width, yerr=for_cis, label='Foraging', capsize=5, color='#e67e22', alpha=0.8)
ax.set_xlabel('Time Lag (ms)')
ax.set_ylabel('Mean Correlation')
ax.set_title('Phase Comparison (Exploration vs Foraging)')
ax.set_xticks(x_pos)
ax.set_xticklabels(phase_df['Lag_ms'].values)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Add p-values
for i, row in phase_df.iterrows():
    pval = row['MannWhitney_Pvalue']
    y_pos = max(exp_means[i] + exp_cis[i], for_means[i] + for_cis[i])
    if pval < 0.0001:
        pval_str = "p<0.0001***"
    elif pval < 0.05:
        pval_str = f"p={pval:.4f}*"
    else:
        pval_str = f"p={pval:.4f} ns"
    ax.text(i, y_pos + 0.0002, pval_str, ha='center', fontsize=8, fontweight='bold')

# Plot 2: Lag comparison with CI
ax = axes[1]
lag_means = lag_df['Mean'].values
lag_cis = (lag_df['CI_High'].values - lag_df['CI_Low'].values) / 2

ax.bar(lag_df['Lag_ms'].values, lag_means, yerr=lag_cis, capsize=5, color='#9b59b6', alpha=0.8, width=15)
ax.set_xlabel('Time Lag (ms)')
ax.set_ylabel('Mean Correlation')
ax.set_title('Lag Comparison (Across All Data)')
ax.set_xticks([10, 50, 100])
ax.grid(True, alpha=0.3, axis='y')

# Plot 3: Effect sizes (Cohen's d) for phase comparison
ax = axes[2]
phase_cohens = phase_df['Cohens_d'].values
colors = ['#2ecc71' if d < 0 else '#e67e22' for d in phase_cohens]
ax.bar(['10ms', '50ms', '100ms'], phase_cohens, color=colors, alpha=0.8)
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax.set_xlabel('Time Lag (ms)')
ax.set_ylabel("Cohen's d")
ax.set_title("Effect Sizes: Phase Comparison")
ax.grid(True, alpha=0.3, axis='y')

# Effect size interpretation lines
ax.axhline(y=0.2, color='gray', linestyle='--', alpha=0.3, linewidth=1)
ax.axhline(y=-0.2, color='gray', linestyle='--', alpha=0.3, linewidth=1)
ax.text(2.3, 0.22, 'small', fontsize=8, alpha=0.5)
ax.text(2.3, -0.22, 'small', fontsize=8, alpha=0.5)

plt.tight_layout()
fig_file = Path("figures/aca_aca_statistical_analysis.png")
plt.savefig(fig_file, dpi=150, bbox_inches='tight')
print(f"[OK] Saved figure to: {fig_file}\n")
plt.close()

# ============================================================================
# PRINT SUMMARY TABLES
# ============================================================================

print("="*70)
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

print("\n[DONE] ACA-ACA statistical analysis complete!")
