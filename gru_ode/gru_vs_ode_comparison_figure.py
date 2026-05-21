"""
GRU vs GRU-ODE Comparison Figure
=================================
Clean comparison of discrete GRU vs continuous GRU-ODE (500ms bins).
Both used condition-specific pooled models with identical architecture
except GRU-ODE replaces the recurrence with an ODE between observations.

Uses scatter plots (each point = session) with identity line to show
whether GRU-ODE matches GRU performance.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load results
gru_df = pd.read_csv("data/gru_pooled_by_region_results.csv")
ode_df = pd.read_csv("data/gru_ode_pooled_by_region_results.csv")

# Use condition_specific models only (not combined)
gru = gru_df[gru_df['model_type'] == 'condition_specific'].copy()
ode = ode_df[ode_df['model_type'] == 'condition_specific'].copy()

# Merge on session + region
merged = gru.merge(ode, on=['region', 'session', 'state', 'phase', 'n_neurons'],
                   suffixes=('_gru', '_ode'))

# Metrics to compare
metrics = {
    'test_r2': 'Test R$^2$',
    'pr': 'Participation Ratio',
    'variance': 'Hidden Variance',
    'speed': 'Trajectory Speed',
}

# Colors and markers
region_colors = {'LHA': '#E65100', 'RSP': '#1565C0'}
state_markers = {'Fed': 'o', 'Fasted': '^'}

fig, axes = plt.subplots(2, 2, figsize=(10, 10))
fig.suptitle('GRU vs GRU-ODE Performance Comparison\n(500ms bins, condition-specific pooled models)',
             fontsize=14, fontweight='bold')

for ax, (metric, label) in zip(axes.flat, metrics.items()):
    gru_col = f'{metric}_gru'
    ode_col = f'{metric}_ode'

    # Plot identity line
    all_vals = np.concatenate([merged[gru_col].values, merged[ode_col].values])
    vmin, vmax = all_vals.min(), all_vals.max()
    pad = (vmax - vmin) * 0.1
    lims = [vmin - pad, vmax + pad]
    ax.plot(lims, lims, '--', color='gray', linewidth=1, alpha=0.5, zorder=1)

    # Plot each session
    plotted_combos = set()
    for _, row in merged.iterrows():
        region = row['region']
        state = row['state']
        color = region_colors[region]
        marker = state_markers[state]
        combo = (region, state)
        label_text = f"{region} {state}" if combo not in plotted_combos else None
        plotted_combos.add(combo)

        ax.scatter(row[gru_col], row[ode_col], c=color, marker=marker,
                   s=80, edgecolors='black', linewidths=0.5, zorder=5,
                   alpha=0.8, label=label_text)

    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel(f'GRU {label}', fontsize=11)
    ax.set_ylabel(f'GRU-ODE {label}', fontsize=11)
    ax.set_title(label, fontsize=12, fontweight='bold')
    ax.set_aspect('equal')

    # Add R and mean difference as text
    r_val = np.corrcoef(merged[gru_col], merged[ode_col])[0, 1]
    mean_diff = (merged[ode_col] - merged[gru_col]).mean()
    ax.text(0.05, 0.95, f'r = {r_val:.3f}\nmean diff = {mean_diff:+.4f}',
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

# Single legend for all panels
handles, labels_leg = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels_leg, loc='lower center', ncol=4, fontsize=11,
           bbox_to_anchor=(0.5, -0.02))

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.savefig("figures/gru_vs_gru_ode_comparison.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: figures/gru_vs_gru_ode_comparison.png")

# Also print summary table
print("\nSummary (condition_specific models):")
print(f"{'Metric':<20} {'GRU mean':>10} {'ODE mean':>10} {'Diff':>10} {'Corr':>8}")
print("-" * 60)
for metric, label in metrics.items():
    gru_vals = merged[f'{metric}_gru'].values
    ode_vals = merged[f'{metric}_ode'].values
    r = np.corrcoef(gru_vals, ode_vals)[0, 1]
    print(f"{label:<20} {gru_vals.mean():>10.4f} {ode_vals.mean():>10.4f} "
          f"{(ode_vals - gru_vals).mean():>+10.4f} {r:>8.3f}")
