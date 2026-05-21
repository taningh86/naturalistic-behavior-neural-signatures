"""
GRU (500ms) vs GRU-ODE (10ms) Comparison Figure
=================================================
Cross-timescale comparison: discrete GRU at 500ms bins vs
continuous GRU-ODE at 10ms bins (Poisson).

Prediction metrics differ (R2 vs D2), but latent dynamics metrics
(PR, variance, speed) are directly comparable.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Load results -- condition_specific only
gru_df = pd.read_csv("data/gru_pooled_by_region_results.csv")
gru = gru_df[gru_df['model_type'] == 'condition_specific'].copy()

ode_df = pd.read_csv("data/gru_ode_10ms_poisson_results.csv")
ode = ode_df[ode_df['model_type'] == 'condition_specific'].copy()

# Merge on session + region
merged = gru.merge(ode, on=['region', 'session', 'state', 'phase', 'n_neurons'],
                   suffixes=('_gru', '_ode'))

# Separate by region and state
groups = {
    'LHA Fed': merged[(merged['region'] == 'LHA') & (merged['state'] == 'Fed')],
    'LHA Fasted': merged[(merged['region'] == 'LHA') & (merged['state'] == 'Fasted')],
    'RSP Fed': merged[(merged['region'] == 'RSP') & (merged['state'] == 'Fed')],
    'RSP Fasted': merged[(merged['region'] == 'RSP') & (merged['state'] == 'Fasted')],
}

group_colors = {
    'LHA Fed': '#E65100',
    'LHA Fasted': '#FF9800',
    'RSP Fed': '#1565C0',
    'RSP Fasted': '#64B5F6',
}

# =====================================================================
# Figure: 2x3 panels
# Row 1: Prediction (R2 vs D2), PR, Variance
# Row 2: Speed, PR scatter, Speed scatter
# =====================================================================

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('GRU vs GRU-ODE Comparison\nCondition-Specific Pooled Models',
             fontsize=14, fontweight='bold')

# Consistent colors for the two models
GRU_COLOR = 'gray'
ODE_COLOR = '#E65100'

# --- Panel A: Prediction accuracy (different metrics, side by side) ---
ax = axes[0, 0]
x_positions = np.arange(len(groups))
width = 0.35

gru_means = [g['test_r2'].mean() for g in groups.values()]
gru_sems = [g['test_r2'].sem() for g in groups.values()]
ode_means = [g['d2'].mean() for g in groups.values()]
ode_sems = [g['d2'].sem() for g in groups.values()]

bars1 = ax.bar(x_positions - width/2, gru_means, width, yerr=gru_sems,
               color=GRU_COLOR, edgecolor='black', linewidth=0.5,
               capsize=3, label='GRU (R$^2$)', alpha=0.8)
bars2 = ax.bar(x_positions + width/2, ode_means, width, yerr=ode_sems,
               color=ODE_COLOR, edgecolor='black', linewidth=0.5,
               capsize=3, label='GRU-ODE (D$^2$)', alpha=0.8)

# Individual session points
for i, (name, grp) in enumerate(groups.items()):
    ax.scatter([i - width/2]*len(grp), grp['test_r2'], c='black', s=15, zorder=5, alpha=0.6)
    ax.scatter([i + width/2]*len(grp), grp['d2'], c='black', s=15, zorder=5, alpha=0.6)

ax.set_xticks(x_positions)
ax.set_xticklabels(groups.keys(), fontsize=9, rotation=15)
ax.set_ylabel('Explained Variance', fontsize=11)
ax.set_title('A) Prediction Accuracy\n(R$^2$ vs D$^2$ -- different metrics)', fontsize=11)
ax.legend(fontsize=9)
ax.axhline(y=0, color='black', linewidth=0.5, linestyle='-')

# --- Panel B: Participation Ratio ---
ax = axes[0, 1]
gru_means = [g['pr_gru'].mean() for g in groups.values()]
gru_sems = [g['pr_gru'].sem() for g in groups.values()]
ode_means = [g['pr_ode'].mean() for g in groups.values()]
ode_sems = [g['pr_ode'].sem() for g in groups.values()]

bars1 = ax.bar(x_positions - width/2, gru_means, width, yerr=gru_sems,
               color=GRU_COLOR, edgecolor='black', linewidth=0.5,
               capsize=3, label='GRU', alpha=0.8)
bars2 = ax.bar(x_positions + width/2, ode_means, width, yerr=ode_sems,
               color=ODE_COLOR, edgecolor='black', linewidth=0.5,
               capsize=3, label='GRU-ODE', alpha=0.8)

for i, (name, grp) in enumerate(groups.items()):
    ax.scatter([i - width/2]*len(grp), grp['pr_gru'], c='black', s=15, zorder=5, alpha=0.6)
    ax.scatter([i + width/2]*len(grp), grp['pr_ode'], c='black', s=15, zorder=5, alpha=0.6)

ax.set_xticks(x_positions)
ax.set_xticklabels(groups.keys(), fontsize=9, rotation=15)
ax.set_ylabel('Participation Ratio', fontsize=11)
ax.set_title('B) Latent Dimensionality', fontsize=11)
ax.legend(fontsize=9)

# --- Panel C: Hidden Variance ---
ax = axes[0, 2]
gru_means = [g['variance_gru'].mean() for g in groups.values()]
gru_sems = [g['variance_gru'].sem() for g in groups.values()]
ode_means = [g['variance_ode'].mean() for g in groups.values()]
ode_sems = [g['variance_ode'].sem() for g in groups.values()]

bars1 = ax.bar(x_positions - width/2, gru_means, width, yerr=gru_sems,
               color=GRU_COLOR, edgecolor='black', linewidth=0.5,
               capsize=3, label='GRU', alpha=0.8)
bars2 = ax.bar(x_positions + width/2, ode_means, width, yerr=ode_sems,
               color=ODE_COLOR, edgecolor='black', linewidth=0.5,
               capsize=3, label='GRU-ODE', alpha=0.8)

for i, (name, grp) in enumerate(groups.items()):
    ax.scatter([i - width/2]*len(grp), grp['variance_gru'], c='black', s=15, zorder=5, alpha=0.6)
    ax.scatter([i + width/2]*len(grp), grp['variance_ode'], c='black', s=15, zorder=5, alpha=0.6)

ax.set_xticks(x_positions)
ax.set_xticklabels(groups.keys(), fontsize=9, rotation=15)
ax.set_ylabel('Hidden Variance', fontsize=11)
ax.set_title('C) Hidden State Spread', fontsize=11)
ax.legend(fontsize=9)

# --- Panel D: Trajectory Speed ---
ax = axes[1, 0]
gru_means = [g['speed_gru'].mean() for g in groups.values()]
gru_sems = [g['speed_gru'].sem() for g in groups.values()]
ode_means = [g['speed_ode'].mean() for g in groups.values()]
ode_sems = [g['speed_ode'].sem() for g in groups.values()]

bars1 = ax.bar(x_positions - width/2, gru_means, width, yerr=gru_sems,
               color=GRU_COLOR, edgecolor='black', linewidth=0.5,
               capsize=3, label='GRU', alpha=0.8)
bars2 = ax.bar(x_positions + width/2, ode_means, width, yerr=ode_sems,
               color=ODE_COLOR, edgecolor='black', linewidth=0.5,
               capsize=3, label='GRU-ODE', alpha=0.8)

for i, (name, grp) in enumerate(groups.items()):
    ax.scatter([i - width/2]*len(grp), grp['speed_gru'], c='black', s=15, zorder=5, alpha=0.6)
    ax.scatter([i + width/2]*len(grp), grp['speed_ode'], c='black', s=15, zorder=5, alpha=0.6)

ax.set_xticks(x_positions)
ax.set_xticklabels(groups.keys(), fontsize=9, rotation=15)
ax.set_ylabel('Trajectory Speed', fontsize=11)
ax.set_title('D) Latent Dynamics Speed', fontsize=11)
ax.legend(fontsize=9)

# --- Panel E: PR scatter (session-matched) ---
ax = axes[1, 1]
for name, grp in groups.items():
    ax.scatter(grp['pr_gru'], grp['pr_ode'], c=group_colors[name],
               s=80, edgecolors='black', linewidths=0.5, label=name, alpha=0.8,
               marker='o' if 'Fed' in name else '^')

all_pr = np.concatenate([merged['pr_gru'].values, merged['pr_ode'].values])
lims = [all_pr.min() - 1, all_pr.max() + 1]
ax.plot(lims, lims, '--', color='gray', linewidth=1, alpha=0.5)
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_aspect('equal')
ax.set_xlabel('GRU PR', fontsize=11)
ax.set_ylabel('GRU-ODE PR', fontsize=11)
ax.set_title('E) PR: Session-Matched', fontsize=11)
r_pr = np.corrcoef(merged['pr_gru'], merged['pr_ode'])[0, 1]
ax.text(0.05, 0.95, f'r = {r_pr:.3f}', transform=ax.transAxes, fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
ax.legend(fontsize=8, loc='lower right')

# --- Panel F: Speed scatter (session-matched) ---
ax = axes[1, 2]
for name, grp in groups.items():
    ax.scatter(grp['speed_gru'], grp['speed_ode'], c=group_colors[name],
               s=80, edgecolors='black', linewidths=0.5, label=name, alpha=0.8,
               marker='o' if 'Fed' in name else '^')

all_spd = np.concatenate([merged['speed_gru'].values, merged['speed_ode'].values])
lims = [all_spd.min() - 0.2, all_spd.max() + 0.2]
ax.plot(lims, lims, '--', color='gray', linewidth=1, alpha=0.5)
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_aspect('equal')
ax.set_xlabel('GRU Speed', fontsize=11)
ax.set_ylabel('GRU-ODE Speed', fontsize=11)
ax.set_title('F) Speed: Session-Matched', fontsize=11)
r_spd = np.corrcoef(merged['speed_gru'], merged['speed_ode'])[0, 1]
ax.text(0.05, 0.95, f'r = {r_spd:.3f}', transform=ax.transAxes, fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
ax.legend(fontsize=8, loc='lower right')

plt.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig("figures/gru_500ms_vs_gru_ode_10ms_comparison.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: figures/gru_500ms_vs_gru_ode_10ms_comparison.png")

# Summary table
print("\nSummary by group:")
print(f"{'Group':<15} {'Metric':<12} {'GRU 500ms':>10} {'ODE 10ms':>10} {'Diff':>10}")
print("-" * 60)
for name, grp in groups.items():
    for m_gru, m_ode, label in [
        ('test_r2', 'd2', 'R2/D2'),
        ('pr_gru', 'pr_ode', 'PR'),
        ('variance_gru', 'variance_ode', 'Variance'),
        ('speed_gru', 'speed_ode', 'Speed'),
    ]:
        g_val = grp[m_gru].mean()
        o_val = grp[m_ode].mean()
        print(f"{name:<15} {label:<12} {g_val:>10.4f} {o_val:>10.4f} {o_val - g_val:>+10.4f}")
    print()

# Paired stats on comparable metrics
print("\nPaired Wilcoxon tests (all 16 sessions):")
for m_gru, m_ode, label in [
    ('pr_gru', 'pr_ode', 'PR'),
    ('variance_gru', 'variance_ode', 'Variance'),
    ('speed_gru', 'speed_ode', 'Speed'),
]:
    stat, p = stats.wilcoxon(merged[m_gru], merged[m_ode])
    direction = "ODE > GRU" if merged[m_ode].mean() > merged[m_gru].mean() else "GRU > ODE"
    print(f"  {label:<12}: p={p:.4f}, {direction}")
