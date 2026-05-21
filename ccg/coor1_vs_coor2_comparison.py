"""
Compare Exploration vs Foraging phase effects between Coordinates-1 (fed only)
and Coordinates-2 (all fed). Session-level means with individual data points.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as sp_stats

LAGS_MS = [2, 5, 10, 50, 100]

def fmt_p(p):
    if p < 0.001: return f"p={p:.2e}***"
    elif p < 0.01: return f"p={p:.3f}**"
    elif p < 0.05: return f"p={p:.3f}*"
    else: return f"p={p:.2f} ns"

def cohens_d(g1, g2):
    n1, n2 = len(g1), len(g2)
    v1, v2 = np.var(g1, ddof=1), np.var(g2, ddof=1)
    ps = np.sqrt(((n1-1)*v1 + (n2-1)*v2) / (n1+n2-2))
    return (np.mean(g1) - np.mean(g2)) / ps if ps > 0 else 0

# Load session means
c1_lha = pd.read_csv("data/single_probe_lha_lha_good_session_means.csv")
c1_rsp = pd.read_csv("data/single_probe_rsp_rsp_good_session_means.csv")
c1_cross = pd.read_csv("data/single_probe_lha_rsp_good_session_means.csv")
c2_lha = pd.read_csv("data/coor2_lha_lha_good_session_means.csv")
c2_rsp = pd.read_csv("data/coor2_rsp_rsp_good_session_means.csv")
c2_cross = pd.read_csv("data/coor2_lha_rsp_good_session_means.csv")

# Filter Coor1 to fed only
c1_lha_fed = c1_lha[c1_lha['state'] == 'fed']
c1_rsp_fed = c1_rsp[c1_rsp['state'] == 'fed']
c1_cross_fed = c1_cross[c1_cross['state'] == 'fed']

networks = [
    ('LHA-LHA', c1_lha_fed, c2_lha),
    ('RSP-RSP', c1_rsp_fed, c2_rsp),
    ('LHA-RSP', c1_cross_fed, c2_cross),
]

# ── Print comparison table ──
print("=" * 90)
print("EXPLORATION vs FORAGING: Coordinates-1 (fed only) vs Coordinates-2")
print("=" * 90)

all_stats = []
for nt, c1, c2 in networks:
    print(f"\n{'-'*70}")
    print(f"  {nt}")
    print(f"{'-'*70}")
    print(f"  {'Lag':>5}  {'C1 Exp (N=2)':>14} {'C1 For (N=2)':>14} {'C1 d':>7}  "
          f"{'C2 Exp (N=3)':>14} {'C2 For (N=3)':>14} {'C2 d':>7}  {'Same dir?':>9}")

    for lag in LAGS_MS:
        col = f'correlation_{lag}ms'
        c1_exp = c1[c1['phase'] == 'exploration'][col].values
        c1_for = c1[c1['phase'] == 'foraging'][col].values
        c2_exp = c2[c2['phase'] == 'exploration'][col].values
        c2_for = c2[c2['phase'] == 'foraging'][col].values

        d1 = cohens_d(c1_exp, c1_for) if len(c1_exp) >= 2 and len(c1_for) >= 2 else 0
        d2 = cohens_d(c2_exp, c2_for) if len(c2_exp) >= 2 and len(c2_for) >= 2 else 0
        same = "YES" if (d1 > 0 and d2 > 0) or (d1 < 0 and d2 < 0) else "NO"

        # Pooled test (combine Coor1 fed + Coor2)
        pool_exp = np.concatenate([c1_exp, c2_exp])
        pool_for = np.concatenate([c1_for, c2_for])
        if len(pool_exp) >= 3 and len(pool_for) >= 3:
            _, pool_p = sp_stats.mannwhitneyu(pool_exp, pool_for, alternative='two-sided')
            pool_d = cohens_d(pool_exp, pool_for)
        else:
            pool_p, pool_d = np.nan, np.nan

        print(f"  {lag:>3}ms  {np.mean(c1_exp):.6f}±{np.std(c1_exp):.6f}  "
              f"{np.mean(c1_for):.6f}±{np.std(c1_for):.6f}  {d1:>+6.2f}   "
              f"{np.mean(c2_exp):.6f}±{np.std(c2_exp):.6f}  "
              f"{np.mean(c2_for):.6f}±{np.std(c2_for):.6f}  {d2:>+6.2f}   {same:>5}")

        all_stats.append({
            'Network': nt, 'Lag_ms': lag,
            'C1_Exp_Mean': np.mean(c1_exp), 'C1_For_Mean': np.mean(c1_for), 'C1_d': d1,
            'C2_Exp_Mean': np.mean(c2_exp), 'C2_For_Mean': np.mean(c2_for), 'C2_d': d2,
            'Pooled_Exp_Mean': np.mean(pool_exp), 'Pooled_For_Mean': np.mean(pool_for),
            'Pooled_d': pool_d, 'Pooled_p': pool_p,
            'Same_direction': same,
        })

    # Pooled summary
    print(f"\n  Pooled (Coor1 fed + Coor2, N=5 Exp vs N=5 For):")
    for lag in LAGS_MS:
        col = f'correlation_{lag}ms'
        pool_exp = np.concatenate([c1[c1['phase']=='exploration'][col].values,
                                    c2[c2['phase']=='exploration'][col].values])
        pool_for = np.concatenate([c1[c1['phase']=='foraging'][col].values,
                                    c2[c2['phase']=='foraging'][col].values])
        _, p = sp_stats.mannwhitneyu(pool_exp, pool_for, alternative='two-sided')
        d = cohens_d(pool_exp, pool_for)
        print(f"    {lag:>3}ms: Exp={np.mean(pool_exp):.6f}, For={np.mean(pool_for):.6f}, "
              f"d={d:+.3f}, {fmt_p(p)}")

stats_df = pd.DataFrame(all_stats)
stats_df.to_csv("data/coor1_vs_coor2_phase_comparison.csv", index=False)

# ── Figure: 3-panel comparison ──
fig, axes = plt.subplots(1, 3, figsize=(20, 7))
colors = {
    'c1_exp': '#27ae60', 'c1_for': '#f39c12',
    'c2_exp': '#2ecc71', 'c2_for': '#e67e22',
}

for ax_idx, (nt, c1, c2) in enumerate(networks):
    ax = axes[ax_idx]
    x = np.arange(len(LAGS_MS))
    w = 0.18

    for gi, (label, data, phase, color, hatch) in enumerate([
        ('Coor1 Exp', c1, 'exploration', '#27ae60', ''),
        ('Coor1 For', c1, 'foraging', '#f39c12', ''),
        ('Coor2 Exp', c2, 'exploration', '#2ecc71', '//'),
        ('Coor2 For', c2, 'foraging', '#e67e22', '//'),
    ]):
        means = []
        sems = []
        pts_list = []
        for lag in LAGS_MS:
            col = f'correlation_{lag}ms'
            vals = data[data['phase'] == phase][col].values
            means.append(np.mean(vals))
            sems.append(sp_stats.sem(vals) if len(vals) > 1 else 0)
            pts_list.append(vals)

        positions = x + (gi - 1.5) * w
        bars = ax.bar(positions, means, w * 0.9, yerr=sems, label=label,
                      capsize=4, color=color, alpha=0.7, hatch=hatch,
                      edgecolor='white' if not hatch else color,
                      error_kw={'linewidth': 1.5})

        # Individual session dots
        for i, pts in enumerate(pts_list):
            jitter = np.random.uniform(-w*0.15, w*0.15, len(pts))
            ax.scatter(positions[i] + jitter, pts, color='#2c3e50', s=30,
                       zorder=5, edgecolors='white', linewidth=0.5)

    ax.set_xlabel('Time Lag (ms)', fontsize=11, fontweight='bold')
    if ax_idx == 0:
        ax.set_ylabel('Session Mean Cross-Correlation ± SEM', fontsize=11, fontweight='bold')
    ax.set_title(nt, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([str(l) for l in LAGS_MS])
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.axhline(0, color='gray', linewidth=0.5, linestyle='-')

    if ax_idx == 0:
        ax.legend(fontsize=9, loc='upper right', ncol=2)

    # Add pooled p-values at top
    for i, lag in enumerate(LAGS_MS):
        col = f'correlation_{lag}ms'
        pool_exp = np.concatenate([c1[c1['phase']=='exploration'][col].values,
                                    c2[c2['phase']=='exploration'][col].values])
        pool_for = np.concatenate([c1[c1['phase']=='foraging'][col].values,
                                    c2[c2['phase']=='foraging'][col].values])
        _, p = sp_stats.mannwhitneyu(pool_exp, pool_for, alternative='two-sided')
        d = cohens_d(pool_exp, pool_for)

        # Position text above the bars
        all_vals = np.concatenate([pool_exp, pool_for])
        ymax = np.max(all_vals)
        yrange = np.max(all_vals) - np.min(all_vals)
        ax.text(i, ymax + yrange * 0.15, f'd={d:+.2f}\n{fmt_p(p)}',
                ha='center', fontsize=7, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow', alpha=0.8))

fig.suptitle('Exploration vs Foraging: Coordinates-1 (Fed Only) vs Coordinates-2\n'
             'Pooled stats: N=5 Exp vs N=5 For (Coor1 fed + Coor2)',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig("figures/coor1_vs_coor2_phase_comparison.png", dpi=150, bbox_inches='tight')
print(f"\n[OK] Saved figures/coor1_vs_coor2_phase_comparison.png")
plt.close()


# ── Summary table figure ──
fig, ax = plt.subplots(figsize=(16, 8))
ax.axis('off')

header = ['Network', 'Lag', 'C1 Exp\n(N=2)', 'C1 For\n(N=2)', 'C1 d',
          'C2 Exp\n(N=3)', 'C2 For\n(N=3)', 'C2 d',
          'Pooled d\n(N=5v5)', 'Pooled p']
rows = [header]
for _, r in stats_df.iterrows():
    rows.append([
        r['Network'], f"{int(r['Lag_ms'])}ms",
        f"{r['C1_Exp_Mean']:.6f}", f"{r['C1_For_Mean']:.6f}", f"{r['C1_d']:+.2f}",
        f"{r['C2_Exp_Mean']:.6f}", f"{r['C2_For_Mean']:.6f}", f"{r['C2_d']:+.2f}",
        f"{r['Pooled_d']:+.2f}", fmt_p(r['Pooled_p']),
    ])

t = ax.table(cellText=rows, cellLoc='center', loc='center',
             colWidths=[0.08, 0.05, 0.10, 0.10, 0.06, 0.10, 0.10, 0.06, 0.08, 0.10])
t.auto_set_font_size(False)
t.set_fontsize(9)
t.scale(1, 1.8)

# Style header
for j in range(len(header)):
    t[(0, j)].set_facecolor('#2c3e50')
    t[(0, j)].set_text_props(weight='bold', color='white')

# Alternate row colors, highlight same-direction rows
for i in range(1, len(rows)):
    same = stats_df.iloc[i-1]['Same_direction']
    for j in range(len(header)):
        if same == 'YES':
            t[(i, j)].set_facecolor('#d5f5e3')  # light green
        else:
            t[(i, j)].set_facecolor('#fadbd8')  # light red

ax.set_title('Exploration vs Foraging: Coor1 (Fed Only) vs Coor2\n'
             'Green = same direction, Red = opposite direction',
             fontsize=13, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig("figures/coor1_vs_coor2_phase_table.png", dpi=150, bbox_inches='tight')
print(f"[OK] Saved figures/coor1_vs_coor2_phase_table.png")
plt.close()

print("\n[DONE]")
