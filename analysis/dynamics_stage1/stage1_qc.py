"""
Stage 1 cross-session QC.

Inputs: data/dynamics_stage1/all_sessions_summary.csv (315 phase rows, 20 sessions)

Drop: S13 (1 phase, recording too short), S23/S24 (NEW_PARADIGM, also short).
Final analyzed set: 17 sessions  fed=8, fasted=5, HFD=4.

Tests
-----
1. Peak vs trough speed (paired within session, per region):
     Wilcoxon signed-rank on session-level means per phase_type.
2. Peak vs trough curvature (same).
3. Cross-condition (fed/fasted/HFD) on session-level peak speed and trough speed:
     Kruskal-Wallis 3-way + pairwise Mann-Whitney U with BH correction.
4. Behavioral covariation: per-phase-type mean behavior fractions; chi-square on
     frac > 0 vs phase_type for each of the 5 target behaviors.
5. Region asymmetry: ACA vs LHA speed/curvature paired across phases.

Outputs
-------
data/dynamics_stage1/qc_session_means.csv
data/dynamics_stage1/qc_peak_vs_trough.csv
data/dynamics_stage1/qc_cross_condition.csv
data/dynamics_stage1/qc_region_asymmetry.csv
data/dynamics_stage1/qc_behavior_by_phase_type.csv
figures/dynamics_stage1/qc_speed_by_phase.png
figures/dynamics_stage1/qc_curvature_by_phase.png
figures/dynamics_stage1/qc_behavior_heatmap.png
data/dynamics_stage1/stage1_qc_report.md
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "analysis" / "cycles_partD"))
from dp_cycles_lib import bh_correct

OUTDIR = REPO / "data" / "dynamics_stage1"
FIGDIR = REPO / "figures" / "dynamics_stage1"
FIGDIR.mkdir(parents=True, exist_ok=True)

DROP = [13, 23, 24]
TARGET_BEHAVIORS = [
    'feeding', 'digging_sand', 'incomplete_home_returns',
    'quick_one_loop_at_home', 'transition_wall_exploration',
]
PHASE_ORDER = ['trough', 'rising', 'peak', 'falling']
STATE_ORDER = ['fed', 'fasted', 'fed-HFD']

REGIONS = ['ACA', 'LHA']


def load_clean():
    df = pd.read_csv(OUTDIR / 'all_sessions_summary.csv')
    df = df[~df['session'].isin(DROP)].copy()
    return df


def session_means(df):
    """Session × phase_type mean of every numeric metric."""
    metrics = [c for c in df.columns
               if any(p in c for p in
                      ['mean_speed', 'mean_curv', 'peak_speed', 'peak_curv',
                       'mean_fr', 'mean_pc1', 'mean_entropy', 'mean_velocity',
                       'mean_dist_nearest_pot', 'frac_'])]
    g = df.groupby(['session', 'state', 'exp_phase', 'phase_type'])[metrics].mean().reset_index()
    return g


def paired_peak_trough(sm):
    """For each metric × region, paired Wilcoxon between peak and trough phases
    using session-level means. Returns rows of (metric, n_sessions, stat, p)."""
    out = []
    metric_cols = [c for c in sm.columns
                   if c.startswith(('mean_speed_', 'mean_curv_',
                                    'peak_speed_', 'peak_curv_',
                                    'mean_fr_', 'mean_pc1_'))]
    for m in metric_cols:
        peak = sm[sm.phase_type == 'peak'].set_index('session')[m]
        trough = sm[sm.phase_type == 'trough'].set_index('session')[m]
        common = peak.index.intersection(trough.index)
        a, b = peak.loc[common].dropna(), trough.loc[common].dropna()
        cm = a.index.intersection(b.index)
        if len(cm) < 5:
            continue
        a, b = a.loc[cm], b.loc[cm]
        try:
            stat, p = stats.wilcoxon(a, b)
        except ValueError:
            stat, p = np.nan, np.nan
        out.append(dict(metric=m, n=len(cm),
                        peak_mean=float(a.mean()), trough_mean=float(b.mean()),
                        diff=float((a - b).mean()),
                        wilcoxon_stat=float(stat) if np.isfinite(stat) else np.nan,
                        p=float(p) if np.isfinite(p) else np.nan))
    df = pd.DataFrame(out)
    df['q'] = bh_correct(df['p'].fillna(1.0).values) if len(df) else []
    return df


def cross_condition(sm):
    """Kruskal-Wallis across fed / fasted / HFD on session-level peak and trough
    means; pairwise Mann-Whitney U with BH correction across all metric*pair tests."""
    rows_kw = []
    rows_pair = []
    metric_cols = [c for c in sm.columns
                   if c.startswith(('mean_speed_', 'mean_curv_',
                                    'mean_fr_', 'mean_pc1_'))]
    for m in metric_cols:
        for pt in ['peak', 'trough', 'rising', 'falling']:
            sub = sm[sm.phase_type == pt]
            groups = [sub[sub.state == s][m].dropna().values for s in STATE_ORDER]
            if any(len(g) < 3 for g in groups):
                continue
            try:
                kw_stat, kw_p = stats.kruskal(*groups)
            except ValueError:
                kw_stat, kw_p = np.nan, np.nan
            rows_kw.append(dict(metric=m, phase_type=pt,
                                fed_mean=float(np.mean(groups[0])),
                                fasted_mean=float(np.mean(groups[1])),
                                hfd_mean=float(np.mean(groups[2])),
                                fed_n=len(groups[0]), fasted_n=len(groups[1]),
                                hfd_n=len(groups[2]),
                                kw_stat=float(kw_stat) if np.isfinite(kw_stat) else np.nan,
                                kw_p=float(kw_p) if np.isfinite(kw_p) else np.nan))
            for i, j in [(0, 1), (0, 2), (1, 2)]:
                a, b = groups[i], groups[j]
                if len(a) < 3 or len(b) < 3:
                    continue
                try:
                    u, p = stats.mannwhitneyu(a, b, alternative='two-sided')
                except ValueError:
                    u, p = np.nan, np.nan
                rows_pair.append(dict(metric=m, phase_type=pt,
                                      pair=f"{STATE_ORDER[i]} vs {STATE_ORDER[j]}",
                                      u=float(u) if np.isfinite(u) else np.nan,
                                      p=float(p) if np.isfinite(p) else np.nan))
    kw_df = pd.DataFrame(rows_kw)
    pair_df = pd.DataFrame(rows_pair)
    if len(kw_df):
        kw_df['kw_q'] = bh_correct(kw_df['kw_p'].fillna(1.0).values)
    if len(pair_df):
        pair_df['q'] = bh_correct(pair_df['p'].fillna(1.0).values)
    return kw_df, pair_df


def region_asymmetry(sm):
    """Paired Wilcoxon ACA vs LHA on session-level means per phase type."""
    out = []
    for stat_name in ['mean_speed', 'mean_curv', 'mean_fr', 'mean_pc1']:
        for pt in PHASE_ORDER:
            sub = sm[sm.phase_type == pt]
            a = sub[f'{stat_name}_ACA'].dropna().values
            b = sub[f'{stat_name}_LHA'].dropna().values
            common_idx = sub[[f'{stat_name}_ACA', f'{stat_name}_LHA']].dropna()
            if len(common_idx) < 5:
                continue
            a = common_idx[f'{stat_name}_ACA'].values
            b = common_idx[f'{stat_name}_LHA'].values
            try:
                w, p = stats.wilcoxon(a, b)
            except ValueError:
                w, p = np.nan, np.nan
            out.append(dict(stat=stat_name, phase_type=pt, n=len(a),
                            aca_mean=float(a.mean()), lha_mean=float(b.mean()),
                            wilcoxon_stat=float(w) if np.isfinite(w) else np.nan,
                            p=float(p) if np.isfinite(p) else np.nan))
    df = pd.DataFrame(out)
    if len(df):
        df['q'] = bh_correct(df['p'].fillna(1.0).values)
    return df


def behavior_by_phase(df):
    """For each behavior, compute mean fraction by phase_type; KW across phase_type
    on raw per-phase fractions."""
    out = []
    for b in TARGET_BEHAVIORS:
        col = f'frac_{b}'
        groups = [df[df.phase_type == pt][col].values for pt in PHASE_ORDER]
        try:
            kw, p = stats.kruskal(*groups)
        except ValueError:
            kw, p = np.nan, np.nan
        means = {pt: float(g.mean()) for pt, g in zip(PHASE_ORDER, groups)}
        out.append(dict(behavior=b, **{f'mean_{pt}': means[pt] for pt in PHASE_ORDER},
                        kw_stat=float(kw) if np.isfinite(kw) else np.nan,
                        kw_p=float(p) if np.isfinite(p) else np.nan))
    df_out = pd.DataFrame(out)
    if len(df_out):
        df_out['q'] = bh_correct(df_out['kw_p'].fillna(1.0).values)
    return df_out


def fig_speed_by_phase(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, region in zip(axes, REGIONS):
        col = f'mean_speed_{region}'
        sns.boxplot(data=df, x='phase_type', y=col, hue='state',
                    hue_order=STATE_ORDER, order=PHASE_ORDER, ax=ax,
                    palette={'fed': '#1f77b4', 'fasted': '#ff7f0e',
                             'fed-HFD': '#d62728'},
                    showfliers=False)
        sns.stripplot(data=df, x='phase_type', y=col, hue='state',
                      hue_order=STATE_ORDER, order=PHASE_ORDER, ax=ax,
                      dodge=True, size=2, alpha=0.4, color='black', legend=False)
        ax.set_title(f'{region} speed by phase × state (per-phase rows)')
        ax.set_xlabel('phase type')
        ax.set_ylabel('mean speed')
        ax.grid(alpha=0.3)
        if ax is not axes[0]:
            ax.legend_.remove() if ax.legend_ else None
    plt.tight_layout()
    plt.savefig(FIGDIR / 'qc_speed_by_phase.png', dpi=120)
    plt.close()


def fig_curv_by_phase(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, region in zip(axes, REGIONS):
        col = f'mean_curv_{region}'
        sns.boxplot(data=df, x='phase_type', y=col, hue='state',
                    hue_order=STATE_ORDER, order=PHASE_ORDER, ax=ax,
                    palette={'fed': '#1f77b4', 'fasted': '#ff7f0e',
                             'fed-HFD': '#d62728'},
                    showfliers=False)
        ax.set_title(f'{region} curvature by phase × state')
        ax.set_xlabel('phase type')
        ax.set_ylabel('mean 1-cos(theta)')
        ax.grid(alpha=0.3)
        if ax is not axes[0]:
            ax.legend_.remove() if ax.legend_ else None
    plt.tight_layout()
    plt.savefig(FIGDIR / 'qc_curvature_by_phase.png', dpi=120)
    plt.close()


def fig_behavior_heatmap(df):
    rows = []
    for b in TARGET_BEHAVIORS:
        col = f'frac_{b}'
        for pt in PHASE_ORDER:
            for st in STATE_ORDER:
                sub = df[(df.phase_type == pt) & (df.state == st)]
                rows.append(dict(behavior=b, phase_type=pt, state=st,
                                 mean=float(sub[col].mean()) if len(sub) else np.nan))
    rdf = pd.DataFrame(rows)
    pivot = rdf.pivot_table(index='behavior', columns=['state', 'phase_type'],
                             values='mean')
    pivot = pivot.reindex(columns=pd.MultiIndex.from_product([STATE_ORDER, PHASE_ORDER]))
    fig, ax = plt.subplots(figsize=(12, 4))
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='viridis', ax=ax,
                cbar_kws={'label': 'mean fraction in phase'})
    ax.set_title('Behavior fraction by phase type × state')
    plt.tight_layout()
    plt.savefig(FIGDIR / 'qc_behavior_heatmap.png', dpi=120)
    plt.close()


def _md_table(d, max_rows=None):
    """Tabulate-free markdown writer for small DataFrames."""
    if max_rows is not None:
        d = d.head(max_rows)
    cols = list(d.columns)
    header = '| ' + ' | '.join(str(c) for c in cols) + ' |'
    sep = '| ' + ' | '.join('---' for _ in cols) + ' |'
    lines = [header, sep]
    for _, r in d.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                cells.append(f'{v:.4g}')
            else:
                cells.append(str(v))
        lines.append('| ' + ' | '.join(cells) + ' |')
    return "\n".join(lines)


def write_report(df, sm, pt_df, kw_df, pair_df, ra_df, beh_df):
    n_sess = sm['session'].nunique()
    state_counts = sm.groupby('state')['session'].nunique().to_dict()
    lines = []
    lines.append('# Stage 1 dynamics — cross-session QC')
    lines.append('')
    lines.append(f'- Sessions analyzed: **{n_sess}**  '
                 f'fed={state_counts.get("fed", 0)}, '
                 f'fasted={state_counts.get("fasted", 0)}, '
                 f'fed-HFD={state_counts.get("fed-HFD", 0)}')
    lines.append(f'- Excluded: S{DROP} (S13/S23 single-phase, S24 NEW_PARADIGM)')
    lines.append(f'- Total phase rows: {len(df)}')
    lines.append('')

    lines.append('## 1. Peak vs trough (paired Wilcoxon, session-level means)')
    lines.append('')
    lines.append(pt_df.sort_values('p').pipe(_md_table))
    lines.append('')

    lines.append('## 2. Cross-condition Kruskal-Wallis (per phase type)')
    lines.append('')
    lines.append(kw_df.sort_values('kw_p').pipe(_md_table))
    lines.append('')

    lines.append('## 3. Pairwise Mann-Whitney U (BH-corrected across all rows)')
    lines.append('')
    sig_pairs = pair_df[pair_df['q'] < 0.10].sort_values('q')
    if len(sig_pairs):
        lines.append('Significant pairs at q<0.10:')
        lines.append('')
        lines.append(sig_pairs.pipe(_md_table))
    else:
        lines.append('No pairs survive q<0.10.')
    lines.append('')

    lines.append('## 4. Region asymmetry ACA vs LHA (paired Wilcoxon per phase type)')
    lines.append('')
    lines.append(ra_df.sort_values('p').pipe(_md_table))
    lines.append('')

    lines.append('## 5. Behavior × phase type (KW across phases, all 17 sessions pooled)')
    lines.append('')
    lines.append(beh_df.sort_values('kw_p').pipe(_md_table))
    lines.append('')

    out = OUTDIR / 'stage1_qc_report.md'
    out.write_text("\n".join(lines), encoding='utf-8')
    print(f"  wrote {out}")


def main():
    print('Loading clean csv...')
    df = load_clean()
    print(f'  rows: {len(df)}, sessions: {df["session"].nunique()}')

    print('Session × phase_type means...')
    sm = session_means(df)
    sm.to_csv(OUTDIR / 'qc_session_means.csv', index=False)
    print(f'  rows: {len(sm)}')

    print('Peak vs trough...')
    pt_df = paired_peak_trough(sm)
    pt_df.to_csv(OUTDIR / 'qc_peak_vs_trough.csv', index=False)
    sig = pt_df[pt_df['q'] < 0.10]
    print(f'  q<0.10: {len(sig)} / {len(pt_df)}')

    print('Cross-condition...')
    kw_df, pair_df = cross_condition(sm)
    kw_df.to_csv(OUTDIR / 'qc_cross_condition_kw.csv', index=False)
    pair_df.to_csv(OUTDIR / 'qc_cross_condition_pairwise.csv', index=False)
    if len(kw_df):
        sig_kw = kw_df[kw_df['kw_q'] < 0.10]
        print(f'  KW q<0.10: {len(sig_kw)} / {len(kw_df)}')
    if len(pair_df):
        sig_pair = pair_df[pair_df['q'] < 0.10]
        print(f'  pairwise q<0.10: {len(sig_pair)} / {len(pair_df)}')

    print('Region asymmetry...')
    ra_df = region_asymmetry(sm)
    ra_df.to_csv(OUTDIR / 'qc_region_asymmetry.csv', index=False)
    sig_ra = ra_df[ra_df['q'] < 0.10]
    print(f'  q<0.10: {len(sig_ra)} / {len(ra_df)}')

    print('Behavior by phase type...')
    beh_df = behavior_by_phase(df)
    beh_df.to_csv(OUTDIR / 'qc_behavior_by_phase_type.csv', index=False)
    sig_b = beh_df[beh_df['q'] < 0.10]
    print(f'  q<0.10: {len(sig_b)} / {len(beh_df)}')

    print('Figures...')
    fig_speed_by_phase(df)
    fig_curv_by_phase(df)
    fig_behavior_heatmap(df)

    print('Report...')
    write_report(df, sm, pt_df, kw_df, pair_df, ra_df, beh_df)

    print('\nDone.')


if __name__ == '__main__':
    main()
