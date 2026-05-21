"""Step 5: cross-metric consistency check.

If ACA curvature shows a state effect during rising/falling phases, do other ACA dynamics
metrics show consistent state patterns?

Metrics tested (per phase, session-level mean):
  - mean_speed_ACA      (translational speed of neural trajectory)
  - mean_curv_ACA       (reference — already known)
  - mean_fr_ACA         (mean firing rate, z-scored)
  - mean_pc1_ACA        (mean PC1 of ACA population)
  - within-phase variance of ACA curvature (variability proxy, computed from per-bin)
  - within-phase variance of ACA speed

The first 4 are already in the per-phase summary. The variance metrics need recomputation
from saved per-bin curvature/speed.

Also include LHA equivalents as a cross-region sanity check.

Outputs
-------
- data/drilldown_curvature/step5_metric_table.csv
- figures/drilldown_curvature/curvature_metric_consistency.png
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

REPO = Path(r'H:/NPX ANALYSIS REPO')
DATA = REPO / 'data' / 'dynamics_stage1'
OUT_DATA = REPO / 'data' / 'drilldown_curvature'
OUT_FIG = REPO / 'figures' / 'drilldown_curvature'

EXCLUDE = {13, 23, 24}
N_BOOT = 5000
RNG = np.random.default_rng(4)
PAIRS = [('fed', 'fasted'), ('fed', 'fed-HFD')]
PHASE_TYPES = ['rising', 'falling']


def cohens_d(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return np.nan
    s = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
                / (len(a) + len(b) - 2))
    return (a.mean() - b.mean()) / s if s > 0 else np.nan


def boot_diff(a, b, n_boot=N_BOOT, rng=None):
    rng = rng or RNG
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan, np.nan, np.nan
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        ai = rng.choice(a, size=len(a), replace=True)
        bi = rng.choice(b, size=len(b), replace=True)
        diffs[i] = ai.mean() - bi.mean()
    return a.mean() - b.mean(), np.percentile(diffs, 2.5), np.percentile(diffs, 97.5), cohens_d(a, b)


def compute_within_phase_var(df):
    """For each phase row, recompute within-phase variance of ACA curvature, speed."""
    rows = []
    for session, sdf in df.groupby('session'):
        cv = np.load(DATA / f'session_{session}_curvature.npy', allow_pickle=True).item()
        sp = np.load(DATA / f'session_{session}_speed.npy', allow_pickle=True).item()
        cv_aca = cv['ACA']; cv_lha = cv['LHA']
        sp_aca = sp['ACA']; sp_lha = sp['LHA']
        for _, r in sdf.iterrows():
            sb, eb = int(r.start_bin), int(r.end_bin)
            eb_cv = min(eb, len(cv_aca))
            eb_sp = min(eb, len(sp_aca))
            row = dict(session=session, phase_id=r.phase_id, phase_type=r.phase_type,
                       state=r.state)
            seg = cv_aca[sb:eb_cv]; row['var_curv_ACA'] = float(np.var(seg)) if len(seg) > 1 else np.nan
            seg = sp_aca[sb:eb_sp]; row['var_speed_ACA'] = float(np.var(seg)) if len(seg) > 1 else np.nan
            seg = cv_lha[sb:eb_cv]; row['var_curv_LHA'] = float(np.var(seg)) if len(seg) > 1 else np.nan
            seg = sp_lha[sb:eb_sp]; row['var_speed_LHA'] = float(np.var(seg)) if len(seg) > 1 else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    df = pd.read_csv(DATA / 'all_sessions_summary.csv')
    df = df[~df.session.isin(EXCLUDE)].copy()
    df = df[df.phase_type.isin(PHASE_TYPES)].copy()

    print('Recomputing within-phase variance metrics...')
    var_df = compute_within_phase_var(df)
    df = df.merge(var_df[['session', 'phase_id', 'phase_type',
                          'var_curv_ACA', 'var_speed_ACA',
                          'var_curv_LHA', 'var_speed_LHA']],
                  on=['session', 'phase_id', 'phase_type'], how='left')

    metrics = [
        'mean_curv_ACA', 'mean_speed_ACA', 'mean_fr_ACA', 'mean_pc1_ACA',
        'var_curv_ACA', 'var_speed_ACA',
        'mean_curv_LHA', 'mean_speed_LHA', 'mean_fr_LHA', 'mean_pc1_LHA',
    ]

    rows = []
    for ptype in PHASE_TYPES:
        sub = df[df.phase_type == ptype]
        for metric in metrics:
            sm = sub.groupby(['session', 'state'])[metric].mean().reset_index()
            for sa, sb in PAIRS:
                a = sm[sm.state == sa][metric].values
                b = sm[sm.state == sb][metric].values
                a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
                obs, lo, hi, d = boot_diff(a, b)
                try:
                    _, p = mannwhitneyu(a, b, alternative='two-sided')
                except (ValueError, IndexError):
                    p = np.nan
                rows.append(dict(metric=metric, phase_type=ptype, contrast=f'{sa}_vs_{sb}',
                                 n_a=len(a), n_b=len(b),
                                 mean_a=a.mean() if len(a) else np.nan,
                                 mean_b=b.mean() if len(b) else np.nan,
                                 mean_diff=obs, ci_lo=lo, ci_hi=hi, cohens_d=d,
                                 mw_p=p,
                                 ci_excludes_zero=((lo > 0) or (hi < 0)) if not np.isnan(lo) else False))

    out = pd.DataFrame(rows)
    out.to_csv(OUT_DATA / 'step5_metric_table.csv', index=False)

    print('\n=== Step 5: cross-metric consistency ===')
    cols = ['metric', 'phase_type', 'contrast', 'mean_a', 'mean_b',
            'mean_diff', 'ci_lo', 'ci_hi', 'cohens_d', 'mw_p', 'ci_excludes_zero']
    print(out[cols].to_string(index=False))

    # Convergence summary: how many ACA metrics show a state effect in same direction?
    print('\n=== ACA metric convergence ===')
    for ptype in PHASE_TYPES:
        for sa, sb in PAIRS:
            aca = out[(out.phase_type == ptype) & (out.contrast == f'{sa}_vs_{sb}')
                      & out.metric.str.contains('_ACA')]
            n_sig = aca.ci_excludes_zero.sum()
            n_neg = ((aca.mean_diff < 0) & aca.ci_excludes_zero).sum()
            n_pos = ((aca.mean_diff > 0) & aca.ci_excludes_zero).sum()
            print(f'  {ptype:8s} {sa:7s} vs {sb:8s}: {n_sig}/{len(aca)} ACA metrics significant '
                  f'({n_neg} negative diff, {n_pos} positive)')

    # Figure: heatmap of cohens_d per metric x contrast
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for c, ptype in enumerate(PHASE_TYPES):
        ax = axes[c]
        sub = out[out.phase_type == ptype]
        contrasts = ['fed_vs_fasted', 'fed_vs_fed-HFD']
        mat = np.full((len(metrics), len(contrasts)), np.nan)
        sig_mat = np.zeros_like(mat, dtype=bool)
        for i, m in enumerate(metrics):
            for j, ctr in enumerate(contrasts):
                row = sub[(sub.metric == m) & (sub.contrast == ctr)]
                if len(row):
                    mat[i, j] = row.cohens_d.iloc[0]
                    sig_mat[i, j] = bool(row.ci_excludes_zero.iloc[0])

        im = ax.imshow(mat, cmap='RdBu_r', vmin=-4, vmax=4, aspect='auto')
        for i in range(len(metrics)):
            for j in range(len(contrasts)):
                if np.isnan(mat[i, j]):
                    continue
                txt = f'{mat[i, j]:+.2f}'
                if sig_mat[i, j]:
                    txt = txt + '*'
                ax.text(j, i, txt, ha='center', va='center',
                        color='white' if abs(mat[i, j]) > 2 else 'black', fontsize=9)
        ax.set_xticks(range(len(contrasts)))
        ax.set_xticklabels(contrasts, rotation=15)
        ax.set_yticks(range(len(metrics)))
        ax.set_yticklabels(metrics)
        ax.set_title(f'{ptype} — Cohen\'s d (* = CI excludes 0)')
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label='Cohen\'s d')
    fig.suptitle('Step 5 — cross-metric consistency for state contrasts (rising/falling)')
    fig.tight_layout()
    fig.savefig(OUT_FIG / 'curvature_metric_consistency.png', dpi=150)
    plt.close(fig)

    print(f'\nWrote {OUT_DATA / "step5_metric_table.csv"}')
    print(f'Wrote {OUT_FIG / "curvature_metric_consistency.png"}')


if __name__ == '__main__':
    main()
