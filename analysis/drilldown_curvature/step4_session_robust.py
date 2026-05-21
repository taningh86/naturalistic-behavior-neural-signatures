"""Step 4: session-level robustness check.

Per state, plot session-level mean ACA curvature distributions in rising/falling phases.
Leave-one-session-out (LOSO): drop each session in turn, recompute fed-vs-fasted and
fed-vs-HFD effect sizes. If removing the most extreme session per group preserves the
contrast (CI still excludes zero), the effect is not outlier-driven.

Outputs
-------
- data/drilldown_curvature/step4_loso.csv
- figures/drilldown_curvature/curvature_session_level.png
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPO = Path(r'H:/NPX ANALYSIS REPO')
DATA = REPO / 'data' / 'dynamics_stage1'
OUT_DATA = REPO / 'data' / 'drilldown_curvature'
OUT_FIG = REPO / 'figures' / 'drilldown_curvature'

EXCLUDE = {13, 23, 24}
N_BOOT = 5000
RNG = np.random.default_rng(3)
PAIRS = [('fed', 'fasted'), ('fed', 'fed-HFD'), ('fasted', 'fed-HFD')]
PHASE_TYPES = ['rising', 'falling']
METRIC = 'mean_curv_ACA'
STATE_ORDER = ['fed', 'fasted', 'fed-HFD']
STATE_COLOR = {'fed': '#1b9e77', 'fasted': '#d95f02', 'fed-HFD': '#7570b3'}


def cohens_d(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    s = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return (a.mean() - b.mean()) / s if s > 0 else np.nan


def boot_diff(a, b, n_boot=N_BOOT, rng=None):
    rng = rng or RNG
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return None
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        ai = rng.choice(a, size=len(a), replace=True)
        bi = rng.choice(b, size=len(b), replace=True)
        diffs[i] = ai.mean() - bi.mean()
    return a.mean() - b.mean(), np.percentile(diffs, 2.5), np.percentile(diffs, 97.5), cohens_d(a, b)


def main():
    df = pd.read_csv(DATA / 'all_sessions_summary.csv')
    df = df[~df.session.isin(EXCLUDE)].copy()
    df = df[df.phase_type.isin(PHASE_TYPES)].copy()

    sm = df.groupby(['session', 'state', 'phase_type'])[METRIC].mean().reset_index()
    print('Session-level means (rising/falling):')
    print(sm.to_string(index=False))

    # LOSO
    rows = []
    for ptype in PHASE_TYPES:
        for sa, sb in PAIRS:
            tab = sm[(sm.phase_type == ptype) & (sm.state.isin([sa, sb]))].copy()
            sessions_to_drop = tab.session.unique()

            # full dataset baseline
            a_full = tab[tab.state == sa][METRIC].values
            b_full = tab[tab.state == sb][METRIC].values
            res = boot_diff(a_full, b_full)
            if res:
                obs, lo, hi, d = res
                rows.append(dict(phase_type=ptype, contrast=f'{sa}_vs_{sb}',
                                 dropped='NONE', dropped_state='-',
                                 mean_diff=obs, ci_lo=lo, ci_hi=hi, cohens_d=d,
                                 ci_excludes_zero=(lo > 0) or (hi < 0),
                                 n_a=len(a_full), n_b=len(b_full)))

            for ds in sessions_to_drop:
                drop_state = tab[tab.session == ds].state.iloc[0]
                t2 = tab[tab.session != ds]
                a = t2[t2.state == sa][METRIC].values
                b = t2[t2.state == sb][METRIC].values
                res = boot_diff(a, b)
                if res is None:
                    continue
                obs, lo, hi, d = res
                rows.append(dict(phase_type=ptype, contrast=f'{sa}_vs_{sb}',
                                 dropped=int(ds), dropped_state=drop_state,
                                 mean_diff=obs, ci_lo=lo, ci_hi=hi, cohens_d=d,
                                 ci_excludes_zero=(lo > 0) or (hi < 0),
                                 n_a=len(a), n_b=len(b)))

    out = pd.DataFrame(rows)
    out.to_csv(OUT_DATA / 'step4_loso.csv', index=False)

    print('\n=== Step 4: LOSO summary (does any single dropped session flip CI?) ===')
    for (ptype, contrast), grp in out.groupby(['phase_type', 'contrast']):
        full = grp[grp.dropped == 'NONE'].iloc[0]
        loso = grp[grp.dropped != 'NONE']
        n_flip = (~loso.ci_excludes_zero).sum() if full.ci_excludes_zero else (loso.ci_excludes_zero.sum())
        flippers = loso[loso.ci_excludes_zero != full.ci_excludes_zero]
        print(f'  {ptype:8s} {contrast:18s} full mean_diff={full.mean_diff:+.5f} '
              f'CI=[{full.ci_lo:+.5f}, {full.ci_hi:+.5f}] excludes0={full.ci_excludes_zero}')
        if len(flippers):
            print(f'    flipped by dropping: {flippers.dropped.tolist()}')
        else:
            print(f'    no single-session drop flips the verdict')

    # Figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    # Top row: session-level points per state (rising, falling)
    for c, ptype in enumerate(PHASE_TYPES):
        ax = axes[0, c]
        s2 = sm[sm.phase_type == ptype]
        for i, st in enumerate(STATE_ORDER):
            v = s2[s2.state == st][METRIC].values
            sess = s2[s2.state == st]['session'].values
            x = np.full_like(v, i, dtype=float) + np.random.default_rng(i).uniform(-0.08, 0.08, size=len(v))
            ax.scatter(x, v, color=STATE_COLOR[st], s=70, edgecolor='k', linewidth=0.6, zorder=3)
            for xi, yi, sn in zip(x, v, sess):
                ax.text(xi + 0.04, yi, f'S{sn}', fontsize=7, va='center')
            ax.hlines(np.mean(v), i - 0.25, i + 0.25, colors='k', linewidth=2, zorder=4)
        ax.set_xticks(range(len(STATE_ORDER)))
        ax.set_xticklabels(STATE_ORDER)
        ax.set_title(f'{ptype} — session means w/ session IDs')
        ax.set_ylabel('mean ACA curvature')
        ax.grid(axis='y', alpha=0.25)

    # Bottom row: LOSO mean_diff for fed_vs_fasted, fed_vs_HFD per ptype
    for c, contrast in enumerate(['fed_vs_fasted', 'fed_vs_fed-HFD']):
        ax = axes[1, c]
        for ptype in PHASE_TYPES:
            sub = out[(out.phase_type == ptype) & (out.contrast == contrast)
                      & (out.dropped != 'NONE')].copy()
            if sub.empty:
                continue
            sub = sub.sort_values('dropped').reset_index(drop=True)
            ys = np.arange(len(sub))
            offset = -0.15 if ptype == 'rising' else 0.15
            for y, (_, row) in zip(ys, sub.iterrows()):
                color = '#1b9e77' if ptype == 'rising' else '#d95f02'
                ax.plot([row.ci_lo, row.ci_hi], [y + offset, y + offset], color=color, lw=2,
                        alpha=0.6 if row.ci_excludes_zero else 0.3)
                ax.plot(row.mean_diff, y + offset, 'o', color=color, ms=4)
            full = out[(out.phase_type == ptype) & (out.contrast == contrast)
                       & (out.dropped == 'NONE')].iloc[0]
            ax.axvline(full.mean_diff, color='#1b9e77' if ptype == 'rising' else '#d95f02',
                       linestyle='--', alpha=0.7, label=f'{ptype} full')
            ax.set_yticks(ys)
            ax.set_yticklabels([f'drop S{int(d)}' for d in sub.dropped], fontsize=7)
        ax.axvline(0, color='k', lw=0.7)
        ax.set_xlabel('mean_diff (curvature)')
        ax.set_title(f'{contrast} — LOSO bootstrap CIs')
        ax.legend(loc='best', fontsize=8)
        ax.grid(axis='x', alpha=0.25)

    fig.suptitle('Step 4 — session-level robustness check')
    fig.tight_layout()
    fig.savefig(OUT_FIG / 'curvature_session_level.png', dpi=150)
    plt.close(fig)
    print(f'\nWrote {OUT_DATA / "step4_loso.csv"}')
    print(f'Wrote {OUT_FIG / "curvature_session_level.png"}')


if __name__ == '__main__':
    main()
