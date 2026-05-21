"""Step 3: behavior-conditioned state contrast on ACA curvature.

Subset rising/falling phases by dominant compartment and dominant_action. For each subset
with sufficient n, recompute fed-vs-fasted and fed-vs-HFD effect sizes (bootstrap CI).
Report ALL subsets including those that don't survive.

Subsets:
  by dominant_compartment: Arena, AtPot, Home, Ladder
  by dominant_action: feeding, digging_sand, quick_one_loop_at_home,
                      transition_wall_exploration, incomplete_home_returns, none
Per phase_type: rising, falling.

Outputs
-------
- data/drilldown_curvature/step3_behavior_subsets.csv
- figures/drilldown_curvature/curvature_behavior_conditioned.png
"""
from pathlib import Path
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
RNG = np.random.default_rng(2)
PAIRS = [('fed', 'fasted'), ('fed', 'fed-HFD')]
PHASE_TYPES = ['rising', 'falling']
METRIC = 'mean_curv_ACA'
MIN_PHASES_PER_STATE = 5      # report below this with a flag
MIN_SESSIONS_PER_STATE = 3    # absolute floor; below this we report but don't bootstrap


def cohens_d(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    s = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return (a.mean() - b.mean()) / s if s > 0 else np.nan


def session_resample_diff(df_a, df_b, n_boot=N_BOOT, rng=None):
    """Bootstrap session-level diff. df_a/df_b: phase rows for one state in one subset.
    Strategy: resample session IDs; within each resampled session, take all that session's
    phases in the subset (preserves within-session structure).
    """
    rng = rng or RNG
    sess_a = df_a.session.unique()
    sess_b = df_b.session.unique()
    if len(sess_a) < 2 or len(sess_b) < 2:
        return None
    diffs = np.empty(n_boot)
    ds = np.empty(n_boot)
    for i in range(n_boot):
        sa = rng.choice(sess_a, size=len(sess_a), replace=True)
        sb = rng.choice(sess_b, size=len(sess_b), replace=True)
        # session-level mean per resampled session, then mean of those
        a_means = np.array([df_a[df_a.session == s][METRIC].mean() for s in sa])
        b_means = np.array([df_b[df_b.session == s][METRIC].mean() for s in sb])
        diffs[i] = a_means.mean() - b_means.mean()
        ds[i] = cohens_d(a_means, b_means)
    a_sm = df_a.groupby('session')[METRIC].mean().values
    b_sm = df_b.groupby('session')[METRIC].mean().values
    obs = a_sm.mean() - b_sm.mean()
    obs_d = cohens_d(a_sm, b_sm)
    return dict(
        mean_diff=obs,
        ci_lo=np.nanpercentile(diffs, 2.5),
        ci_hi=np.nanpercentile(diffs, 97.5),
        cohens_d=obs_d,
        d_ci_lo=np.nanpercentile(ds, 2.5),
        d_ci_hi=np.nanpercentile(ds, 97.5),
    )


def main():
    df = pd.read_csv(DATA / 'all_sessions_summary.csv')
    df = df[~df.session.isin(EXCLUDE)].copy()
    df = df[df.phase_type.isin(PHASE_TYPES)].copy()
    print(f'Working with {len(df)} rising/falling phase rows from {df.session.nunique()} sessions')

    # Per-subset breakdown for compartment + action
    rows = []
    for ptype in PHASE_TYPES:
        sub = df[df.phase_type == ptype]
        # by compartment
        for comp in sub.dominant_compartment.dropna().unique():
            sset = sub[sub.dominant_compartment == comp]
            for sa, sb in PAIRS:
                a = sset[sset.state == sa]
                b = sset[sset.state == sb]
                n_a, n_b = len(a), len(b)
                ns_a, ns_b = a.session.nunique(), b.session.nunique()
                row = dict(phase_type=ptype, subset_kind='compartment', subset=comp,
                           contrast=f'{sa}_vs_{sb}',
                           n_phases_a=n_a, n_phases_b=n_b,
                           n_sessions_a=ns_a, n_sessions_b=ns_b,
                           low_power_flag=(n_a < MIN_PHASES_PER_STATE) or (n_b < MIN_PHASES_PER_STATE)
                                          or (ns_a < MIN_SESSIONS_PER_STATE) or (ns_b < MIN_SESSIONS_PER_STATE))
                if ns_a >= 2 and ns_b >= 2:
                    res = session_resample_diff(a, b)
                    if res:
                        row.update(res)
                        a_sm = a.groupby('session')[METRIC].mean().values
                        b_sm = b.groupby('session')[METRIC].mean().values
                        if len(a_sm) >= 2 and len(b_sm) >= 2:
                            try:
                                _, p = mannwhitneyu(a_sm, b_sm, alternative='two-sided')
                            except ValueError:
                                p = np.nan
                            row['mw_p'] = p
                rows.append(row)

        # by dominant_action
        for act in sub.dominant_action.dropna().unique():
            sset = sub[sub.dominant_action == act]
            for sa, sb in PAIRS:
                a = sset[sset.state == sa]
                b = sset[sset.state == sb]
                n_a, n_b = len(a), len(b)
                ns_a, ns_b = a.session.nunique(), b.session.nunique()
                row = dict(phase_type=ptype, subset_kind='action', subset=act,
                           contrast=f'{sa}_vs_{sb}',
                           n_phases_a=n_a, n_phases_b=n_b,
                           n_sessions_a=ns_a, n_sessions_b=ns_b,
                           low_power_flag=(n_a < MIN_PHASES_PER_STATE) or (n_b < MIN_PHASES_PER_STATE)
                                          or (ns_a < MIN_SESSIONS_PER_STATE) or (ns_b < MIN_SESSIONS_PER_STATE))
                if ns_a >= 2 and ns_b >= 2:
                    res = session_resample_diff(a, b)
                    if res:
                        row.update(res)
                        a_sm = a.groupby('session')[METRIC].mean().values
                        b_sm = b.groupby('session')[METRIC].mean().values
                        try:
                            _, p = mannwhitneyu(a_sm, b_sm, alternative='two-sided')
                        except ValueError:
                            p = np.nan
                        row['mw_p'] = p
                rows.append(row)

    out = pd.DataFrame(rows)

    # BH correction across non-low-power rows with valid p
    valid = out[(~out.low_power_flag) & out['mw_p'].notna()].copy()
    if len(valid):
        from scipy.stats import false_discovery_control
        valid = valid.sort_values('mw_p').reset_index(drop=True)
        valid['mw_q'] = false_discovery_control(valid['mw_p'].values, method='bh')
        out = out.merge(valid[['phase_type', 'subset_kind', 'subset', 'contrast', 'mw_q']],
                        on=['phase_type', 'subset_kind', 'subset', 'contrast'], how='left')

    out.to_csv(OUT_DATA / 'step3_behavior_subsets.csv', index=False)

    # Print summary
    print('\n=== Step 3: by compartment ===')
    cols = ['phase_type', 'subset', 'contrast', 'n_phases_a', 'n_phases_b',
            'n_sessions_a', 'n_sessions_b', 'mean_diff', 'ci_lo', 'ci_hi',
            'cohens_d', 'mw_p', 'low_power_flag']
    if 'mw_q' in out.columns:
        cols.append('mw_q')
    comp = out[out.subset_kind == 'compartment'][cols].copy()
    print(comp.to_string(index=False))

    print('\n=== Step 3: by dominant_action ===')
    act = out[out.subset_kind == 'action'][cols].copy()
    print(act.to_string(index=False))

    # Figure: forest plot of mean_diff w/ CIs per subset for fed_vs_fasted and fed_vs_HFD
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    for r, ptype in enumerate(PHASE_TYPES):
        for c, contrast in enumerate(['fed_vs_fasted', 'fed_vs_fed-HFD']):
            ax = axes[r, c]
            sub = out[(out.phase_type == ptype) & (out.contrast == contrast)
                      & out.mean_diff.notna()].copy()
            sub = sub.sort_values(['subset_kind', 'subset']).reset_index(drop=True)
            ys = np.arange(len(sub))
            for y, (_, row) in zip(ys, sub.iterrows()):
                color = '0.65' if row.low_power_flag else ('#d95f02' if row.subset_kind == 'compartment' else '#1b9e77')
                ax.plot([row.ci_lo, row.ci_hi], [y, y], color=color, lw=2)
                ax.plot(row.mean_diff, y, 'o', color=color, ms=6)
                lab = f'{row.subset_kind}: {row.subset} (a={row.n_phases_a}/b={row.n_phases_b})'
                if row.low_power_flag:
                    lab += ' †'
                ax.text(ax.get_xlim()[0] if c == 0 else 0, y, lab,
                        ha='right' if c == 0 else 'left', va='center', fontsize=7)
            ax.axvline(0, color='k', lw=0.7)
            ax.set_yticks([])
            ax.set_title(f'{ptype} — {contrast}\n(negative = a < b)')
            ax.set_xlabel('mean_diff (curvature)')
            ax.grid(axis='x', alpha=0.25)
    fig.suptitle('Step 3 — behavior-conditioned state contrasts (95% session-level bootstrap CI)\n† = low power (<5 phases or <3 sessions per state)')
    fig.tight_layout()
    fig.savefig(OUT_FIG / 'curvature_behavior_conditioned.png', dpi=150)
    plt.close(fig)

    # Did the effect survive in adequately-powered subsets?
    print('\n=== Survival summary ===')
    powered = out[(~out.low_power_flag) & out.mean_diff.notna()].copy()
    print(f'Adequately-powered subsets: {len(powered)}')
    powered['ci_excludes_zero'] = (powered.ci_lo > 0) | (powered.ci_hi < 0)
    n_survive = int(powered.ci_excludes_zero.sum())
    print(f'  CI excludes zero: {n_survive}/{len(powered)}')
    if n_survive < len(powered):
        print('\n  Did NOT survive in:')
        print(powered.loc[~powered.ci_excludes_zero,
                          ['phase_type', 'subset_kind', 'subset', 'contrast',
                           'mean_diff', 'ci_lo', 'ci_hi']].to_string(index=False))
        print('\n  DID survive in:')
        print(powered.loc[powered.ci_excludes_zero,
                          ['phase_type', 'subset_kind', 'subset', 'contrast',
                           'mean_diff', 'ci_lo', 'ci_hi']].to_string(index=False))
    print(f'\nWrote {OUT_DATA / "step3_behavior_subsets.csv"}')
    print(f'Wrote {OUT_FIG / "curvature_behavior_conditioned.png"}')


if __name__ == '__main__':
    main()
