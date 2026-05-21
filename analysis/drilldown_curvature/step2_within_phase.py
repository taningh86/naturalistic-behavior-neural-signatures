"""Step 2: time-resolved ACA curvature within rising/falling phases.

Per phase, extract curvature across the phase, resample to 50 normalized time bins.
Pool across phases within each state. Bootstrap session-level CIs at each time bin.

Outputs
-------
- data/drilldown_curvature/step2_timecourse.csv (per state, per ptype, per nbin: mean + CI)
- data/drilldown_curvature/step2_state_diff.csv (per ptype, per nbin: pairwise diff + CI)
- figures/drilldown_curvature/curvature_within_phase_trajectory.png
"""
import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPO = Path(r'H:/NPX ANALYSIS REPO')
DATA = REPO / 'data' / 'dynamics_stage1'
OUT_DATA = REPO / 'data' / 'drilldown_curvature'
OUT_FIG = REPO / 'figures' / 'drilldown_curvature'
OUT_DATA.mkdir(parents=True, exist_ok=True)
OUT_FIG.mkdir(parents=True, exist_ok=True)

EXCLUDE = {13, 23, 24}
N_BINS_NORM = 50
MIN_PHASE_BINS = 5  # need at least 5 neural bins (250 ms) to resample meaningfully
N_BOOT = 5000
RNG = np.random.default_rng(1)
PAIRS = [('fed', 'fasted'), ('fed', 'fed-HFD'), ('fasted', 'fed-HFD')]
PHASE_TYPES = ['rising', 'falling']


def load_curvature(session):
    obj = np.load(DATA / f'session_{session}_curvature.npy', allow_pickle=True).item()
    return obj['ACA']


def load_phases(session):
    with open(DATA / f'session_{session}_phases.json') as f:
        return json.load(f)


def resample_phase(curv_phase, n_bins=N_BINS_NORM):
    """Linear interpolate phase-curvature to n_bins normalized timeline."""
    if len(curv_phase) < MIN_PHASE_BINS:
        return None
    t_orig = np.linspace(0, 1, len(curv_phase))
    t_target = np.linspace(0, 1, n_bins)
    return np.interp(t_target, t_orig, curv_phase)


def collect_phase_traces():
    """Return dict[(state, ptype)] -> list of (session, trace[N_BINS_NORM])."""
    summary = pd.read_csv(DATA / 'all_sessions_summary.csv')
    summary = summary[~summary.session.isin(EXCLUDE)].copy()
    state_lookup = summary.groupby('session').state.first().to_dict()

    out = {(s, p): [] for s in ['fed', 'fasted', 'fed-HFD'] for p in PHASE_TYPES}
    for session, state in state_lookup.items():
        curv = load_curvature(session)
        phases = load_phases(session)
        for ph in phases:
            ptype = ph['phase_type']
            if ptype not in PHASE_TYPES:
                continue
            sb, eb = ph['start_bin'], ph['end_bin']
            eb = min(eb, len(curv))
            if eb - sb < MIN_PHASE_BINS:
                continue
            seg = curv[sb:eb]
            tr = resample_phase(seg)
            if tr is None:
                continue
            out[(state, ptype)].append((session, tr))
    return out


def session_pool_per_state(traces, state, ptype):
    """Compute per-session mean trajectory; return DataFrame with 1 row per session."""
    rows = [(s, tr) for s, tr in traces[(state, ptype)]]
    if not rows:
        return None, None
    df = pd.DataFrame([[s] + list(tr) for s, tr in rows],
                      columns=['session'] + [f'b{i}' for i in range(N_BINS_NORM)])
    g = df.groupby('session').mean()
    sessions = g.index.values
    arr = g.values  # (n_sessions, N_BINS_NORM)
    return sessions, arr


def bootstrap_mean_trajectory(arr, n_boot=N_BOOT, rng=None):
    """Resample sessions with replacement; return mean and 2.5/97.5 CIs at each timepoint."""
    rng = rng or RNG
    n = arr.shape[0]
    if n == 0:
        return None
    means = np.empty((n_boot, arr.shape[1]))
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means[i] = arr[idx].mean(axis=0)
    return arr.mean(axis=0), np.percentile(means, 2.5, axis=0), np.percentile(means, 97.5, axis=0)


def bootstrap_diff_trajectory(arr_a, arr_b, n_boot=N_BOOT, rng=None):
    """Bootstrap pairwise mean difference at each timepoint."""
    rng = rng or RNG
    na, nb = arr_a.shape[0], arr_b.shape[0]
    diffs = np.empty((n_boot, arr_a.shape[1]))
    for i in range(n_boot):
        ai = rng.integers(0, na, size=na)
        bi = rng.integers(0, nb, size=nb)
        diffs[i] = arr_a[ai].mean(axis=0) - arr_b[bi].mean(axis=0)
    obs = arr_a.mean(axis=0) - arr_b.mean(axis=0)
    lo = np.percentile(diffs, 2.5, axis=0)
    hi = np.percentile(diffs, 97.5, axis=0)
    return obs, lo, hi


def main():
    print('Collecting phase traces...')
    traces = collect_phase_traces()
    for (s, p), v in traces.items():
        print(f'  {s:8s} {p:8s}: {len(v):3d} phases from {len(set(x[0] for x in v))} sessions')

    # Save mean + CI per state per phase_type
    rows_tc = []
    state_arrays = {}
    for ptype in PHASE_TYPES:
        for state in ['fed', 'fasted', 'fed-HFD']:
            sessions, arr = session_pool_per_state(traces, state, ptype)
            if arr is None or arr.shape[0] < 2:
                continue
            state_arrays[(state, ptype)] = arr
            mean, lo, hi = bootstrap_mean_trajectory(arr)
            for i in range(N_BINS_NORM):
                rows_tc.append(dict(state=state, phase_type=ptype, nbin=i,
                                    mean=mean[i], ci_lo=lo[i], ci_hi=hi[i],
                                    n_sessions=arr.shape[0]))
    pd.DataFrame(rows_tc).to_csv(OUT_DATA / 'step2_timecourse.csv', index=False)

    # Pairwise diffs
    rows_d = []
    for ptype in PHASE_TYPES:
        for sa, sb in PAIRS:
            if (sa, ptype) not in state_arrays or (sb, ptype) not in state_arrays:
                continue
            arr_a = state_arrays[(sa, ptype)]
            arr_b = state_arrays[(sb, ptype)]
            obs, lo, hi = bootstrap_diff_trajectory(arr_a, arr_b)
            sig = (lo > 0) | (hi < 0)
            for i in range(N_BINS_NORM):
                rows_d.append(dict(phase_type=ptype, contrast=f'{sa}_vs_{sb}',
                                   nbin=i, diff=obs[i], ci_lo=lo[i], ci_hi=hi[i],
                                   sig_excludes_zero=bool(sig[i])))
    pd.DataFrame(rows_d).to_csv(OUT_DATA / 'step2_state_diff.csv', index=False)

    # Figure
    state_color = {'fed': '#1b9e77', 'fasted': '#d95f02', 'fed-HFD': '#7570b3'}
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey='row')
    t_norm = np.linspace(0, 1, N_BINS_NORM)
    for col, ptype in enumerate(PHASE_TYPES):
        ax = axes[0, col]
        for state in ['fed', 'fasted', 'fed-HFD']:
            if (state, ptype) not in state_arrays:
                continue
            arr = state_arrays[(state, ptype)]
            mean, lo, hi = bootstrap_mean_trajectory(arr)
            c = state_color[state]
            ax.plot(t_norm, mean, color=c, lw=2, label=f'{state} (n={arr.shape[0]})')
            ax.fill_between(t_norm, lo, hi, color=c, alpha=0.2)
        ax.set_title(f'{ptype} — mean trajectory ± 95% bootstrap CI')
        ax.set_xlabel('normalized phase time (0=start, 1=end)')
        ax.set_ylabel('mean ACA curvature')
        ax.legend(loc='best', fontsize=8)
        ax.grid(alpha=0.25)

    # Diff panel
    diff_df = pd.DataFrame(rows_d)
    for col, ptype in enumerate(PHASE_TYPES):
        ax = axes[1, col]
        for sa, sb in PAIRS:
            d = diff_df[(diff_df.phase_type == ptype) & (diff_df.contrast == f'{sa}_vs_{sb}')]
            if d.empty:
                continue
            ax.plot(d.nbin / (N_BINS_NORM - 1), d['diff'], lw=1.5, label=f'{sa}-{sb}')
            ax.fill_between(d.nbin / (N_BINS_NORM - 1), d.ci_lo, d.ci_hi, alpha=0.15)
            sig_x = d[d.sig_excludes_zero].nbin.values / (N_BINS_NORM - 1)
            sig_y = np.full_like(sig_x, ax.get_ylim()[0] if d['diff'].min() < 0 else 0)
            if len(sig_x):
                ax.scatter(sig_x, np.full_like(sig_x, np.nan), s=12)  # placeholder
        ax.axhline(0, color='k', lw=0.5)
        ax.set_xlabel('normalized phase time')
        ax.set_ylabel('curvature difference')
        ax.set_title(f'{ptype} — pairwise differences (95% bootstrap CI)')
        ax.legend(loc='best', fontsize=8)
        ax.grid(alpha=0.25)

    # Extra panel: significance shading per ptype
    for col, ptype in enumerate(PHASE_TYPES):
        ax = axes[1, col]
        for sa, sb in PAIRS:
            d = diff_df[(diff_df.phase_type == ptype) & (diff_df.contrast == f'{sa}_vs_{sb}')]
            if d.empty:
                continue
            sig = d[d.sig_excludes_zero]
            for _, row in sig.iterrows():
                ax.axvspan(row.nbin / (N_BINS_NORM - 1) - 0.005,
                           row.nbin / (N_BINS_NORM - 1) + 0.005,
                           color='0.85', zorder=0)

    # Hide third column on top row (only 2 phase types)
    axes[0, 2].axis('off')
    axes[1, 2].axis('off')

    fig.suptitle('Step 2 — time-resolved ACA curvature within phases')
    fig.tight_layout()
    fig.savefig(OUT_FIG / 'curvature_within_phase_trajectory.png', dpi=150)
    plt.close(fig)

    # Summary report: how much of each phase shows sig diff
    print('\n=== Step 2 sig-fraction per contrast ===')
    for ptype in PHASE_TYPES:
        for sa, sb in PAIRS:
            d = diff_df[(diff_df.phase_type == ptype) & (diff_df.contrast == f'{sa}_vs_{sb}')]
            if d.empty:
                continue
            frac = d.sig_excludes_zero.mean()
            mean_abs_diff = d['diff'].abs().mean()
            print(f'  {ptype:8s} {sa:8s} vs {sb:8s}: {frac*100:5.1f}% of phase sig (mean |diff|={mean_abs_diff:.4f})')

    print(f'\nWrote {OUT_DATA / "step2_timecourse.csv"}')
    print(f'Wrote {OUT_DATA / "step2_state_diff.csv"}')
    print(f'Wrote {OUT_FIG / "curvature_within_phase_trajectory.png"}')


if __name__ == '__main__':
    main()
