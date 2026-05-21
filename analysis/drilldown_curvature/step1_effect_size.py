"""Step 1: bootstrap effect size for ACA curvature x diet state on rising/falling phases.

Session-level bootstrap throughout: resample sessions (not phases) so the dependency
structure within sessions is preserved.

Stop-condition flag: if 95% CI on the mean difference brackets zero, OR Cohen's d < 0.2,
print STOP and write a flag file. Stop-condition is informational; downstream steps still
read this output.

Outputs
-------
- data/drilldown_curvature/step1_effect_sizes.csv
- figures/drilldown_curvature/curvature_state_distribution.png
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt

REPO = Path(r'H:/NPX ANALYSIS REPO')
DATA = REPO / 'data' / 'dynamics_stage1'
OUT_DATA = REPO / 'data' / 'drilldown_curvature'
OUT_FIG = REPO / 'figures' / 'drilldown_curvature'
OUT_DATA.mkdir(parents=True, exist_ok=True)
OUT_FIG.mkdir(parents=True, exist_ok=True)

EXCLUDE = {13, 23, 24}
N_BOOT = 10000
RNG = np.random.default_rng(0)

STATE_PAIRS = [('fed', 'fasted'), ('fed', 'fed-HFD'), ('fasted', 'fed-HFD')]
PHASE_TYPES = ['rising', 'falling']
METRIC = 'mean_curv_ACA'


def session_means(df, metric, phase_type):
    """Return DataFrame: session, state, mean(metric) over `phase_type` rows in that session."""
    sub = df[df.phase_type == phase_type]
    g = sub.groupby(['session', 'state'])[metric].mean().reset_index()
    return g


def cohens_d(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    s = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return (a.mean() - b.mean()) / s if s > 0 else np.nan


def bootstrap_diff(a, b, n_boot=N_BOOT, rng=None):
    """Session-level resample with replacement; return (mean_diff_obs, ci_lo, ci_hi, d_obs, d_lo, d_hi)."""
    rng = rng or RNG
    a = np.asarray(a, float); b = np.asarray(b, float)
    diffs = np.empty(n_boot)
    ds = np.empty(n_boot)
    for i in range(n_boot):
        ai = rng.choice(a, size=len(a), replace=True)
        bi = rng.choice(b, size=len(b), replace=True)
        diffs[i] = ai.mean() - bi.mean()
        ds[i] = cohens_d(ai, bi)
    return (
        a.mean() - b.mean(),
        np.nanpercentile(diffs, 2.5), np.nanpercentile(diffs, 97.5),
        cohens_d(a, b),
        np.nanpercentile(ds, 2.5), np.nanpercentile(ds, 97.5),
    )


def main():
    df = pd.read_csv(DATA / 'all_sessions_summary.csv')
    df = df[~df.session.isin(EXCLUDE)].copy()
    print(f'Loaded {len(df)} phase rows from {df.session.nunique()} sessions')
    print(f'  state counts: {df.groupby("state").session.nunique().to_dict()}')

    rows = []
    for ptype in PHASE_TYPES:
        sm = session_means(df, METRIC, ptype)
        print(f'\n[{ptype}] session means by state:')
        for st in ['fed', 'fasted', 'fed-HFD']:
            v = sm[sm.state == st][METRIC].values
            print(f'  {st:8s} n={len(v)}  mean={v.mean():.5f}  median={np.median(v):.5f}  IQR=[{np.percentile(v,25):.5f}, {np.percentile(v,75):.5f}]')

        for sa, sb in STATE_PAIRS:
            a = sm[sm.state == sa][METRIC].values
            b = sm[sm.state == sb][METRIC].values
            if len(a) < 2 or len(b) < 2:
                continue
            mdiff, lo, hi, d, dlo, dhi = bootstrap_diff(a, b)
            try:
                u, p = mannwhitneyu(a, b, alternative='two-sided')
            except ValueError:
                u, p = np.nan, np.nan

            ci_brackets_zero = (lo <= 0 <= hi)
            d_small = abs(d) < 0.2

            rows.append(dict(
                metric=METRIC, phase_type=ptype, contrast=f'{sa}_vs_{sb}',
                n_a=len(a), n_b=len(b),
                mean_a=a.mean(), mean_b=b.mean(),
                median_a=np.median(a), median_b=np.median(b),
                iqr_a_lo=np.percentile(a, 25), iqr_a_hi=np.percentile(a, 75),
                iqr_b_lo=np.percentile(b, 25), iqr_b_hi=np.percentile(b, 75),
                mean_diff=mdiff, ci_lo=lo, ci_hi=hi,
                cohens_d=d, d_ci_lo=dlo, d_ci_hi=dhi,
                mw_u=u, mw_p=p,
                ci_brackets_zero=ci_brackets_zero,
                cohens_d_below_0p2=d_small,
                fragile=ci_brackets_zero or d_small,
            ))

    out = pd.DataFrame(rows)
    out.to_csv(OUT_DATA / 'step1_effect_sizes.csv', index=False)
    print('\n=== Step 1 results ===')
    cols = ['phase_type', 'contrast', 'n_a', 'n_b', 'mean_diff', 'ci_lo', 'ci_hi',
            'cohens_d', 'd_ci_lo', 'd_ci_hi', 'mw_p', 'fragile']
    print(out[cols].to_string(index=False))

    # Figure: per phase_type, per state distribution of session means
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    state_order = ['fed', 'fasted', 'fed-HFD']
    state_color = {'fed': '#1b9e77', 'fasted': '#d95f02', 'fed-HFD': '#7570b3'}
    for ax, ptype in zip(axes, PHASE_TYPES):
        sm = session_means(df, METRIC, ptype)
        for i, st in enumerate(state_order):
            v = sm[sm.state == st][METRIC].values
            x = np.full_like(v, i, dtype=float) + RNG.uniform(-0.08, 0.08, size=len(v))
            ax.scatter(x, v, color=state_color[st], s=42, edgecolor='k', linewidth=0.5, zorder=3)
            ax.hlines(np.mean(v), i - 0.25, i + 0.25, colors='k', linewidth=2, zorder=4)
            ax.hlines(np.median(v), i - 0.18, i + 0.18, colors='gray', linewidth=1, linestyle=':', zorder=4)
        ax.set_xticks(range(len(state_order)))
        ax.set_xticklabels(state_order)
        ax.set_title(f'{ptype} phases')
        ax.set_ylabel('session-mean ACA curvature')
        ax.grid(axis='y', alpha=0.25)
    fig.suptitle('Step 1 — session-level ACA curvature distributions per state')
    fig.tight_layout()
    fig.savefig(OUT_FIG / 'curvature_state_distribution.png', dpi=150)
    plt.close(fig)
    print(f'\nWrote {OUT_DATA / "step1_effect_sizes.csv"}')
    print(f'Wrote {OUT_FIG / "curvature_state_distribution.png"}')

    n_fragile = int(out.fragile.sum())
    if n_fragile == len(out):
        print('\nSTOP CONDITION: ALL contrasts are fragile (CI brackets zero or d<0.2). Finding too weak to pursue.')
    elif n_fragile > 0:
        print(f'\nPARTIAL: {n_fragile}/{len(out)} contrasts fragile. Specific contrasts that survived:')
        print(out.loc[~out.fragile, ['phase_type', 'contrast', 'mean_diff', 'cohens_d']].to_string(index=False))
    else:
        print('\nALL contrasts pass: every CI excludes zero AND |d| >= 0.2.')


if __name__ == '__main__':
    main()
