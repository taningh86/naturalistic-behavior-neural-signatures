"""Stage 2 Analysis 2: cross-region pairing (ACA curvature x LHA speed).

For each phase across all 17 sessions, compute the within-phase Pearson correlation
between ACA curvature and LHA speed at 50 ms bins. Pool by state and phase type.

Tests
-----
1. Distribution per state: is the correlation different from zero in each state?
2. State difference in correlation: bootstrap CIs.
3. Lagged correlation: -2 s to +2 s in 100 ms steps. Where is the peak?
4. Shuffled null: within-session pair-shuffle (use a different phase's LHA speed).

Parameters
----------
MIN_PHASE_BINS = 20    # 1 s minimum
LAG_RANGE_BINS = 40    # +/- 2 s
LAG_STEP_BINS = 2      # 100 ms steps
N_BOOT = 5000
EXCLUDE = {13, 23, 24}

Outputs
-------
data/stage2_temporal_pairing/
  crossregion_correlation.csv         # per phase
  crossregion_state_summary.csv       # per state x phase_type, mean+CI vs null
  crossregion_lagged.csv              # per state x lag, mean correlation
figures/stage2_temporal_pairing/
  crossregion_correlation_distribution.png
  crossregion_lag.png
  crossregion_state_difference.png
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

REPO = Path(r'H:/NPX ANALYSIS REPO')
S1D = REPO / 'data' / 'dynamics_stage1'
OUT_DATA = REPO / 'data' / 'stage2_temporal_pairing'
OUT_FIG = REPO / 'figures' / 'stage2_temporal_pairing'
OUT_DATA.mkdir(parents=True, exist_ok=True)
OUT_FIG.mkdir(parents=True, exist_ok=True)

EXCLUDE = {13, 23, 24}
MIN_PHASE_BINS = 20
LAG_RANGE_BINS = 40
LAG_STEP_BINS = 2
N_BOOT = 5000
RNG = np.random.default_rng(20)
PAIRS = [('fed', 'fasted'), ('fed', 'fed-HFD'), ('fasted', 'fed-HFD')]
STATES = ['fed', 'fasted', 'fed-HFD']
PHASE_TYPES = ['rising', 'peak', 'falling', 'trough']
STATE_COLOR = {'fed': '#1b9e77', 'fasted': '#d95f02', 'fed-HFD': '#7570b3'}


def load_session(session):
    cv = np.load(S1D / f'session_{session}_curvature.npy', allow_pickle=True).item()
    sp = np.load(S1D / f'session_{session}_speed.npy', allow_pickle=True).item()
    with open(S1D / f'session_{session}_phases.json') as f:
        phases = json.load(f)
    return cv, sp, phases


def session_state_lookup():
    df = pd.read_csv(S1D / 'all_sessions_summary.csv')
    df = df[~df.session.isin(EXCLUDE)]
    return df.groupby('session').state.first().to_dict()


def safe_corr(x, y):
    if len(x) < MIN_PHASE_BINS or np.any(np.isnan(x)) or np.any(np.isnan(y)):
        return np.nan
    if x.std() == 0 or y.std() == 0:
        return np.nan
    try:
        r, _ = pearsonr(x, y)
        return r
    except Exception:
        return np.nan


def lagged_corr(x, y, lag):
    """Pearson r between x and y shifted by `lag` bins (positive lag => y leads x: y[t+lag] vs x[t])."""
    if lag == 0:
        return safe_corr(x, y)
    if lag > 0:
        x_seg, y_seg = x[:-lag], y[lag:]
    else:
        x_seg, y_seg = x[-lag:], y[:lag]
    if len(x_seg) < MIN_PHASE_BINS:
        return np.nan
    return safe_corr(x_seg, y_seg)


def main():
    state_lookup = session_state_lookup()
    rows = []
    for session, state in state_lookup.items():
        cv, sp, phases = load_session(session)
        aca_curv = cv['ACA']
        lha_speed = sp['LHA']
        # ACA curvature length is T-2; LHA speed length is T-1. Align by truncating to common length.
        T = min(len(aca_curv), len(lha_speed))
        for ph in phases:
            sb, eb = int(ph['start_bin']), min(int(ph['end_bin']), T)
            if eb - sb < MIN_PHASE_BINS:
                continue
            x = aca_curv[sb:eb]
            y = lha_speed[sb:eb]
            r = safe_corr(x, y)
            row = dict(session=session, state=state, phase_id=ph['phase_id'],
                       phase_type=ph['phase_type'], start_bin=sb, end_bin=eb,
                       n_bins=eb - sb, r=r)
            # lagged
            for lag in range(-LAG_RANGE_BINS, LAG_RANGE_BINS + 1, LAG_STEP_BINS):
                row[f'r_lag{lag:+d}'] = lagged_corr(x, y, lag)
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DATA / 'crossregion_correlation.csv', index=False)
    print(f'Phases passing length filter: {len(df)}')
    print(f'  by state:      {df.state.value_counts().to_dict()}')
    print(f'  by phase_type: {df.phase_type.value_counts().to_dict()}')

    # ---- Within-session shuffle null
    print('\nGenerating shuffle null...')
    null_rows = []
    for session, state in state_lookup.items():
        sub = df[df.session == session]
        if len(sub) < 2:
            continue
        cv, sp, phases = load_session(session)
        aca_curv = cv['ACA']; lha_speed = sp['LHA']
        T = min(len(aca_curv), len(lha_speed))
        # For each phase, swap LHA speed segment with another random phase's LHA speed segment from same session
        for _, ph_row in sub.iterrows():
            sb, eb = ph_row.start_bin, ph_row.end_bin
            x = aca_curv[sb:eb]
            # random alternate phase from same session, different phase_id
            others = sub[sub.phase_id != ph_row.phase_id]
            if len(others) == 0:
                continue
            alt = others.iloc[RNG.integers(0, len(others))]
            asb, aeb = alt.start_bin, alt.end_bin
            # truncate or repeat-pad to match length
            n_target = eb - sb
            seg = lha_speed[asb:aeb]
            if len(seg) >= n_target:
                seg = seg[:n_target]
            else:
                # pad by tiling
                reps = int(np.ceil(n_target / max(len(seg), 1)))
                seg = np.tile(seg, reps)[:n_target]
            r_null = safe_corr(x, seg)
            null_rows.append(dict(session=session, state=state, phase_id=ph_row.phase_id,
                                  phase_type=ph_row.phase_type, r_null=r_null))
    null_df = pd.DataFrame(null_rows)

    # ---- Per state x phase_type summary with bootstrap CIs (session-level)
    summary_rows = []
    for state in STATES:
        for ptype in PHASE_TYPES:
            real_sub = df[(df.state == state) & (df.phase_type == ptype)].dropna(subset=['r'])
            null_sub = null_df[(null_df.state == state) & (null_df.phase_type == ptype)].dropna(subset=['r_null'])

            # session-level: mean per session, then bootstrap on session means
            real_sm = real_sub.groupby('session')['r'].mean().values
            null_sm = null_sub.groupby('session')['r_null'].mean().values
            if len(real_sm) < 2:
                continue
            mean_real = real_sm.mean()
            boots_real = np.array([np.random.default_rng(s).choice(real_sm, len(real_sm), replace=True).mean()
                                   for s in range(N_BOOT)])
            real_lo = np.percentile(boots_real, 2.5); real_hi = np.percentile(boots_real, 97.5)

            mean_null = null_sm.mean() if len(null_sm) else np.nan
            null_lo = null_hi = np.nan
            if len(null_sm) >= 2:
                boots_null = np.array([np.random.default_rng(s + 1000).choice(null_sm, len(null_sm), replace=True).mean()
                                       for s in range(N_BOOT)])
                null_lo = np.percentile(boots_null, 2.5); null_hi = np.percentile(boots_null, 97.5)

            # diff vs null
            mean_diff_null = mean_real - mean_null
            summary_rows.append(dict(state=state, phase_type=ptype,
                                     n_phases=len(real_sub), n_sessions=len(real_sm),
                                     mean_r_real=mean_real, real_ci_lo=real_lo, real_ci_hi=real_hi,
                                     real_excludes_zero=(real_lo > 0) or (real_hi < 0),
                                     mean_r_null=mean_null, null_ci_lo=null_lo, null_ci_hi=null_hi,
                                     real_minus_null=mean_diff_null))
    pd.DataFrame(summary_rows).to_csv(OUT_DATA / 'crossregion_state_summary.csv', index=False)

    # ---- State pairwise diff in mean r (rising+falling pooled, peak+trough pooled separately)
    pair_rows = []
    for ptype_label, ptypes in [('rising_falling', ['rising', 'falling']),
                                ('peak_trough', ['peak', 'trough']),
                                ('all_phases', PHASE_TYPES)]:
        sub = df[df.phase_type.isin(ptypes)].dropna(subset=['r'])
        for sa, sb in PAIRS:
            a_sm = sub[sub.state == sa].groupby('session')['r'].mean().values
            b_sm = sub[sub.state == sb].groupby('session')['r'].mean().values
            if len(a_sm) < 2 or len(b_sm) < 2:
                continue
            obs = a_sm.mean() - b_sm.mean()
            diffs = np.empty(N_BOOT)
            for i in range(N_BOOT):
                ai = np.random.default_rng(i + 5000).choice(a_sm, len(a_sm), replace=True)
                bi = np.random.default_rng(i + 6000).choice(b_sm, len(b_sm), replace=True)
                diffs[i] = ai.mean() - bi.mean()
            lo, hi = np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)
            pair_rows.append(dict(group=ptype_label, contrast=f'{sa}_vs_{sb}',
                                  n_a=len(a_sm), n_b=len(b_sm),
                                  mean_a=a_sm.mean(), mean_b=b_sm.mean(),
                                  mean_diff=obs, ci_lo=lo, ci_hi=hi,
                                  ci_excludes_zero=(lo > 0) or (hi < 0)))
    pair_df = pd.DataFrame(pair_rows)
    pair_df.to_csv(OUT_DATA / 'crossregion_state_pairwise.csv', index=False)

    # ---- Lagged correlation per state (averaged over phases, then sessions)
    lag_rows = []
    lag_cols = [c for c in df.columns if c.startswith('r_lag')]
    lag_values = sorted([int(c.replace('r_lag', '')) for c in lag_cols])
    for state in STATES:
        sub = df[df.state == state]
        for lag in lag_values:
            col = f'r_lag{lag:+d}'
            sm = sub.dropna(subset=[col]).groupby('session')[col].mean().values
            if len(sm) < 2:
                continue
            obs = sm.mean()
            boots = np.array([np.random.default_rng(s + 7000).choice(sm, len(sm), replace=True).mean()
                              for s in range(N_BOOT)])
            lo, hi = np.percentile(boots, 2.5), np.percentile(boots, 97.5)
            lag_rows.append(dict(state=state, lag_bins=lag, lag_s=lag * 0.05,
                                 mean_r=obs, ci_lo=lo, ci_hi=hi,
                                 n_sessions=len(sm)))
    lag_df = pd.DataFrame(lag_rows)
    lag_df.to_csv(OUT_DATA / 'crossregion_lagged.csv', index=False)

    # ---- Leverage check: are correlations dominated by long phases?
    print('\nLeverage check: phase length vs |r|')
    long_thresh = df.n_bins.quantile(0.9)
    print(f'  90th percentile phase length: {long_thresh:.0f} bins ({long_thresh*0.05:.1f} s)')
    print(f'  mean |r| in top decile: {df[df.n_bins >= long_thresh].r.abs().mean():.4f}')
    print(f'  mean |r| in bottom 90%: {df[df.n_bins < long_thresh].r.abs().mean():.4f}')

    # ---- Figures
    # Distribution per state
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, group_label, ptypes in [(axes[0], 'rising/falling', ['rising', 'falling']),
                                     (axes[1], 'peak/trough', ['peak', 'trough'])]:
        for i, state in enumerate(STATES):
            sub = df[(df.state == state) & (df.phase_type.isin(ptypes))]
            v = sub.r.dropna().values
            x = np.full_like(v, i, dtype=float) + np.random.default_rng(i).uniform(-0.1, 0.1, size=len(v))
            ax.scatter(x, v, color=STATE_COLOR[state], s=12, alpha=0.4)
            sm = sub.groupby('session')['r'].mean().values
            xm = np.full_like(sm, i, dtype=float) + np.random.default_rng(i + 100).uniform(-0.05, 0.05, size=len(sm))
            ax.scatter(xm, sm, color=STATE_COLOR[state], s=70, edgecolor='k', linewidth=0.6)
            ax.hlines(np.nanmean(sm), i - 0.25, i + 0.25, colors='k', linewidth=2)
        ax.axhline(0, color='k', lw=0.7, linestyle='--')
        ax.set_xticks(range(len(STATES))); ax.set_xticklabels(STATES)
        ax.set_title(f'ACA curv × LHA speed corr — {group_label}')
        ax.set_ylabel('Pearson r')
        ax.grid(axis='y', alpha=0.25)
    fig.suptitle('Within-phase ACA curvature × LHA speed correlation per state')
    fig.tight_layout()
    fig.savefig(OUT_FIG / 'crossregion_correlation_distribution.png', dpi=150)
    plt.close(fig)

    # Lagged correlation
    fig, ax = plt.subplots(figsize=(10, 5))
    for state in STATES:
        s2 = lag_df[lag_df.state == state].sort_values('lag_s')
        if s2.empty:
            continue
        ax.plot(s2.lag_s, s2.mean_r, color=STATE_COLOR[state], lw=2, label=state)
        ax.fill_between(s2.lag_s, s2.ci_lo, s2.ci_hi, color=STATE_COLOR[state], alpha=0.2)
    ax.axvline(0, color='k', lw=0.7, linestyle='--')
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel('lag (s, positive = LHA speed leads)')
    ax.set_ylabel('Pearson r')
    ax.set_title('Lagged ACA curvature × LHA speed correlation')
    ax.legend(); ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_FIG / 'crossregion_lag.png', dpi=150)
    plt.close(fig)

    # State difference
    fig, ax = plt.subplots(figsize=(10, 5))
    pair_plot = pair_df.copy()
    pair_plot['label'] = pair_plot.contrast + ' (' + pair_plot.group + ')'
    pair_plot = pair_plot.sort_values(['group', 'contrast'])
    ys = np.arange(len(pair_plot))
    for y, (_, r) in zip(ys, pair_plot.iterrows()):
        c = '#d95f02' if r.ci_excludes_zero else '#888'
        ax.plot([r.ci_lo, r.ci_hi], [y, y], color=c, lw=2)
        ax.plot(r.mean_diff, y, 'o', color=c, ms=7)
    ax.axvline(0, color='k', lw=0.7, linestyle='--')
    ax.set_yticks(ys); ax.set_yticklabels(pair_plot.label.values, fontsize=9)
    ax.set_xlabel('mean_diff in within-phase r')
    ax.set_title('State pairwise differences in ACA-curv × LHA-speed correlation')
    ax.grid(axis='x', alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_FIG / 'crossregion_state_difference.png', dpi=150)
    plt.close(fig)

    # Print summary
    print('\n=== Per-state summary ===')
    print(pd.read_csv(OUT_DATA / 'crossregion_state_summary.csv').to_string(index=False))
    print('\n=== Pairwise state contrasts ===')
    print(pair_df.to_string(index=False))
    print('\n=== Lag peak per state ===')
    for state in STATES:
        s2 = lag_df[lag_df.state == state].dropna(subset=['mean_r'])
        if s2.empty:
            continue
        peak = s2.iloc[s2.mean_r.abs().idxmax() - s2.index[0]]
        print(f'  {state}: peak |r|={peak.mean_r:.4f} at lag={peak.lag_s:+.2f} s '
              f'(boundary={"YES" if abs(peak.lag_s) >= 1.95 else "no"})')

    print(f'\nWrote {OUT_DATA}')
    print(f'Wrote {OUT_FIG}')


if __name__ == '__main__':
    main()
