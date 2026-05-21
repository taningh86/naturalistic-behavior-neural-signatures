"""
Dual-Probe Probe-1 (LHA+RSP): Combined Statistical Analysis & Figures
Fed vs Fasted, Exploration vs Foraging for LHA-LHA, RSP-RSP, LHA-RSP networks.
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
print("DUAL-PROBE PROBE-1: COMBINED STATISTICAL ANALYSIS")
print("="*70)

def compute_cohens_d(g1, g2):
    n1, n2 = len(g1), len(g2)
    v1, v2 = np.var(g1, ddof=1), np.var(g2, ddof=1)
    ps = np.sqrt(((n1-1)*v1 + (n2-1)*v2) / (n1+n2-2))
    return (np.mean(g1) - np.mean(g2)) / ps if ps > 0 else 0

def compute_ci_95(data):
    m = np.mean(data)
    sem = stats.sem(data)
    ci = sem * stats.t.ppf(0.975, len(data)-1)
    return m, m-ci, m+ci

def fmt_p(p):
    if p < 0.0001: return "p < 0.0001***"
    elif p < 0.001: return f"p = {p:.2e}***"
    elif p < 0.01: return f"p = {p:.4f}**"
    elif p < 0.05: return f"p = {p:.4f}*"
    else: return f"p = {p:.4f} ns"

network_types = {
    'LHA-LHA': 'dp1_lha_lha',
    'RSP-RSP': 'dp1_rsp_rsp',
    'LHA-RSP': 'dp1_lha_rsp'
}

for nt_label, nt_prefix in network_types.items():
    print(f"\n{'#'*70}")
    print(f"  {nt_label} NETWORK ANALYSIS")
    print(f"{'#'*70}")

    # Load connectivity data
    conn_data = []
    for group in ['fed_exploration', 'fed_foraging', 'fasted_exploration', 'fasted_foraging']:
        filepath = f"data/{nt_prefix}_connectivity_{group}.csv"
        try:
            df = pd.read_csv(filepath)
            s, p = group.split('_')
            df['state'] = s
            df['phase'] = p
            conn_data.append(df)
            print(f"  [OK] {filepath}: {len(df)} pairs")
        except Exception as e:
            print(f"  [WARN] {filepath}: {e}")

    if not conn_data:
        print(f"  [SKIP] No data for {nt_label}")
        continue

    full_df = pd.concat(conn_data, ignore_index=True)
    print(f"\n  Total {nt_label} pairs: {len(full_df)}")
    print(f"    Fed: {len(full_df[full_df['state']=='fed'])}, Fasted: {len(full_df[full_df['state']=='fasted'])}")
    print(f"    Exploration: {len(full_df[full_df['phase']=='exploration'])}, Foraging: {len(full_df[full_df['phase']=='foraging'])}")

    # ========== STATE COMPARISON ==========
    print(f"\n  --- STATE COMPARISON (Fed vs Fasted) ---")
    state_results = []
    for lag in [10, 50, 100]:
        col = f'correlation_{lag}ms'
        fed = full_df[full_df['state']=='fed'][col].dropna().values
        fas = full_df[full_df['state']=='fasted'][col].dropna().values

        if len(fed) < 2 or len(fas) < 2:
            print(f"    Lag {lag}ms: insufficient data")
            continue

        mw_s, mw_p = stats.mannwhitneyu(fed, fas)
        t_s, t_p = stats.ttest_ind(fed, fas)
        d = compute_cohens_d(fed, fas)
        fm, fl, fh = compute_ci_95(fed)
        am, al, ah = compute_ci_95(fas)
        fold = ((am - fm) / abs(fm)) * 100 if fm != 0 else 0

        state_results.append({
            'Lag_ms': lag, 'Fed_Mean': fm, 'Fed_CI_Low': fl, 'Fed_CI_High': fh,
            'Fasted_Mean': am, 'Fasted_CI_Low': al, 'Fasted_CI_High': ah,
            'MannWhitney_Statistic': mw_s, 'MannWhitney_Pvalue': mw_p,
            'Ttest_Statistic': t_s, 'Ttest_Pvalue': t_p,
            'Cohens_d': d, 'N_Fed': len(fed), 'N_Fasted': len(fas)
        })

        print(f"    Lag {lag}ms: Fed={fm:.6f}, Fasted={am:.6f}, Change={fold:+.1f}%, {fmt_p(mw_p)}, d={d:.4f}")

    state_df = pd.DataFrame(state_results)

    # ========== PHASE COMPARISON ==========
    print(f"\n  --- PHASE COMPARISON (Exploration vs Foraging) ---")
    phase_results = []
    for lag in [10, 50, 100]:
        col = f'correlation_{lag}ms'
        exp = full_df[full_df['phase']=='exploration'][col].dropna().values
        forg = full_df[full_df['phase']=='foraging'][col].dropna().values

        if len(exp) < 2 or len(forg) < 2:
            print(f"    Lag {lag}ms: insufficient data")
            continue

        mw_s, mw_p = stats.mannwhitneyu(exp, forg)
        t_s, t_p = stats.ttest_ind(exp, forg)
        d = compute_cohens_d(exp, forg)
        em, el, eh = compute_ci_95(exp)
        fm, fl, fh = compute_ci_95(forg)

        phase_results.append({
            'Lag_ms': lag, 'Exploration_Mean': em, 'Exploration_CI_Low': el, 'Exploration_CI_High': eh,
            'Foraging_Mean': fm, 'Foraging_CI_Low': fl, 'Foraging_CI_High': fh,
            'MannWhitney_Statistic': mw_s, 'MannWhitney_Pvalue': mw_p,
            'Ttest_Statistic': t_s, 'Ttest_Pvalue': t_p,
            'Cohens_d': d, 'N_Exploration': len(exp), 'N_Foraging': len(forg)
        })

        fold = ((fm - em) / abs(em)) * 100 if em != 0 else 0
        print(f"    Lag {lag}ms: Exp={em:.6f}, For={fm:.6f}, Change={fold:+.1f}%, {fmt_p(mw_p)}, d={d:.4f}")

    phase_df = pd.DataFrame(phase_results)

    # ========== SAVE CSVs ==========
    if len(state_df) > 0:
        state_df.to_csv(f"data/{nt_prefix}_stats_state_comparison.csv", index=False)
        print(f"  [OK] Saved {nt_prefix}_stats_state_comparison.csv")
    if len(phase_df) > 0:
        phase_df.to_csv(f"data/{nt_prefix}_stats_phase_comparison.csv", index=False)
        print(f"  [OK] Saved {nt_prefix}_stats_phase_comparison.csv")

    # ========== FIGURES ==========

    # Figure 1: State comparison
    if len(state_df) > 0:
        fig, ax = plt.subplots(figsize=(12, 7))
        x = np.arange(len(state_df))
        w = 0.35

        fm = state_df['Fed_Mean'].values
        fc = (state_df['Fed_CI_High'].values - state_df['Fed_CI_Low'].values) / 2
        am = state_df['Fasted_Mean'].values
        ac = (state_df['Fasted_CI_High'].values - state_df['Fasted_CI_Low'].values) / 2

        ax.bar(x-w/2, fm, w, yerr=fc, label='Fed', capsize=8, color='#3498db', alpha=0.8, error_kw={'linewidth': 2})
        ax.bar(x+w/2, am, w, yerr=ac, label='Fasted', capsize=8, color='#e74c3c', alpha=0.8, error_kw={'linewidth': 2})

        ax.set_xlabel('Time Lag (ms)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Mean Cross-Correlation +/- 95% CI', fontsize=12, fontweight='bold')
        ax.set_title(f'{nt_label} Network (Probe-1): State Comparison (Fed vs Fasted)', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(state_df['Lag_ms'].values.astype(int))
        ax.legend(fontsize=11, loc='upper left')
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')

        mv = max(max(fm+fc), max(am+ac))
        ax.set_ylim(0, mv * 1.5)

        for i, row in state_df.iterrows():
            yp = max(fm[i]+fc[i], am[i]+ac[i])
            ax.text(i, yp + mv*0.08, fmt_p(row['MannWhitney_Pvalue']), ha='center', fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', alpha=0.7))
            ax.text(i, yp + mv*0.18, f"d = {row['Cohens_d']:.4f}", ha='center', fontsize=9, style='italic')

        plt.tight_layout()
        f = Path(f"figures/{nt_prefix}_state_comparison_with_pvalues.png")
        plt.savefig(f, dpi=150, bbox_inches='tight')
        print(f"  [OK] Saved {f}")
        plt.close()

    # Figure 2: Phase comparison
    if len(phase_df) > 0:
        fig, ax = plt.subplots(figsize=(12, 7))
        x = np.arange(len(phase_df))

        em = phase_df['Exploration_Mean'].values
        ec = (phase_df['Exploration_CI_High'].values - phase_df['Exploration_CI_Low'].values) / 2
        fm = phase_df['Foraging_Mean'].values
        fc = (phase_df['Foraging_CI_High'].values - phase_df['Foraging_CI_Low'].values) / 2

        ax.bar(x-w/2, em, w, yerr=ec, label='Exploration', capsize=8, color='#2ecc71', alpha=0.8, error_kw={'linewidth': 2})
        ax.bar(x+w/2, fm, w, yerr=fc, label='Foraging', capsize=8, color='#f39c12', alpha=0.8, error_kw={'linewidth': 2})

        ax.set_xlabel('Time Lag (ms)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Mean Cross-Correlation +/- 95% CI', fontsize=12, fontweight='bold')
        ax.set_title(f'{nt_label} Network (Probe-1): Phase Comparison (Exploration vs Foraging)', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(phase_df['Lag_ms'].values.astype(int))
        ax.legend(fontsize=11, loc='upper left')
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')

        mv = max(max(em+ec), max(fm+fc))
        ax.set_ylim(0, mv * 1.5)

        for i, row in phase_df.iterrows():
            yp = max(em[i]+ec[i], fm[i]+fc[i])
            ax.text(i, yp + mv*0.08, fmt_p(row['MannWhitney_Pvalue']), ha='center', fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', alpha=0.7))
            ax.text(i, yp + mv*0.18, f"d = {row['Cohens_d']:.4f}", ha='center', fontsize=9, style='italic')

        plt.tight_layout()
        f = Path(f"figures/{nt_prefix}_phase_comparison_with_pvalues.png")
        plt.savefig(f, dpi=150, bbox_inches='tight')
        print(f"  [OK] Saved {f}")
        plt.close()

    # Figure 3: Summary table
    if len(state_df) > 0 and len(phase_df) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(18, 5))

        ax = axes[0]
        ax.axis('off')
        tbl = [['Lag', 'Fed', 'Fasted', 'p-value', "Cohen's d"]]
        for _, r in state_df.iterrows():
            tbl.append([f"{int(r['Lag_ms'])}ms",
                        f"{r['Fed_Mean']:.4f}", f"{r['Fasted_Mean']:.4f}",
                        fmt_p(r['MannWhitney_Pvalue']), f"{r['Cohens_d']:.4f}"])
        t = ax.table(cellText=tbl, cellLoc='center', loc='center', colWidths=[0.1,0.2,0.2,0.25,0.15])
        t.auto_set_font_size(False); t.set_fontsize(10); t.scale(1,2.5)
        for i in range(5): t[(0,i)].set_facecolor('#3498db'); t[(0,i)].set_text_props(weight='bold',color='white')
        for i in range(1,len(tbl)):
            for j in range(5): t[(i,j)].set_facecolor('#ecf0f1' if i%2==0 else 'white')
        ax.set_title(f'{nt_label}: Fed vs Fasted', fontsize=12, fontweight='bold', pad=20)

        ax = axes[1]
        ax.axis('off')
        tbl = [['Lag', 'Exploration', 'Foraging', 'p-value', "Cohen's d"]]
        for _, r in phase_df.iterrows():
            tbl.append([f"{int(r['Lag_ms'])}ms",
                        f"{r['Exploration_Mean']:.4f}", f"{r['Foraging_Mean']:.4f}",
                        fmt_p(r['MannWhitney_Pvalue']), f"{r['Cohens_d']:.4f}"])
        t = ax.table(cellText=tbl, cellLoc='center', loc='center', colWidths=[0.1,0.2,0.2,0.25,0.15])
        t.auto_set_font_size(False); t.set_fontsize(10); t.scale(1,2.5)
        for i in range(5): t[(0,i)].set_facecolor('#2ecc71'); t[(0,i)].set_text_props(weight='bold',color='white')
        for i in range(1,len(tbl)):
            for j in range(5): t[(i,j)].set_facecolor('#ecf0f1' if i%2==0 else 'white')
        ax.set_title(f'{nt_label}: Exploration vs Foraging', fontsize=12, fontweight='bold', pad=20)

        plt.tight_layout()
        f = Path(f"figures/{nt_prefix}_summary_tables.png")
        plt.savefig(f, dpi=150, bbox_inches='tight')
        print(f"  [OK] Saved {f}")
        plt.close()

# =============================================================================
# GRAND COMPARISON TABLE
# =============================================================================

print(f"\n{'='*70}")
print("GRAND COMPARISON: ALL PROBE-1 NETWORKS (10ms lag)")
print("="*70)

print(f"\n{'Network':<12} {'Fed Mean':>12} {'Fasted Mean':>14} {'Fold Change':>14} {'Cohen d':>10} {'p-value':>20}")
print("-" * 82)

for nt_label, nt_prefix in network_types.items():
    try:
        sdf = pd.read_csv(f"data/{nt_prefix}_stats_state_comparison.csv")
        r = sdf[sdf['Lag_ms'] == 10].iloc[0]
        fold = ((r['Fasted_Mean'] - r['Fed_Mean']) / abs(r['Fed_Mean'])) * 100 if r['Fed_Mean'] != 0 else 0
        print(f"{nt_label:<12} {r['Fed_Mean']:>12.6f} {r['Fasted_Mean']:>14.6f} {fold:>+13.1f}% {r['Cohens_d']:>10.4f} {fmt_p(r['MannWhitney_Pvalue']):>20}")
    except:
        print(f"{nt_label:<12} -- data unavailable --")

print(f"\n[DONE] All probe-1 statistical analyses complete!")
