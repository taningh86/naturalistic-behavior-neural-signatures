"""Stage 2 Analysis 1: peri-inflection time-locking.

For each entropy peak and trough across all 17 sessions, define a ±60 s peri-inflection
window (±1200 bins at 50 ms). Extract aligned signals, average per state per inflection
type, and compute the divergence-onset timepoint where bootstrap CI on the state diff
first excludes zero (with a contiguous-run criterion to avoid single-bin noise).

Parameters (single source of truth)
-----------------------------------
WINDOW_SEC = 60        # +/- 60 s window
BIN_S = 0.05           # 50 ms neural bins (Stage 1 standard)
WINDOW_BINS = 1200     # +/- 1200 bins
N_BOOT = 2000          # bootstrap iterations (session-level)
MIN_RUN_BINS = 20      # 1 s contiguous-run threshold for divergence-onset
EXCLUDE = {13, 23, 24} # same as Stage 1 QC

Edge handling: inflections within WINDOW_BINS of session start/end are kept but the
truncated end is filled with NaN. Per-timepoint averaging uses nanmean and tracks the
contributing-window count per timepoint (saved in csv).

Outputs
-------
data/stage2_temporal_pairing/
  periinflection_curvature_aca.csv
  periinflection_speed_aca.csv
  periinflection_curvature_lha.csv
  periinflection_speed_lha.csv
  periinflection_state_diff_<metric>.csv  (per metric)
  divergence_onset_summary.csv
figures/stage2_temporal_pairing/
  periinflection_curvature_aca.png
  periinflection_speed_lha.png
  divergence_onset.png
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPO = Path(r'H:/NPX ANALYSIS REPO')
S1D = REPO / 'data' / 'dynamics_stage1'
OUT_DATA = REPO / 'data' / 'stage2_temporal_pairing'
OUT_FIG = REPO / 'figures' / 'stage2_temporal_pairing'
OUT_DATA.mkdir(parents=True, exist_ok=True)
OUT_FIG.mkdir(parents=True, exist_ok=True)

EXCLUDE = {13, 23, 24}
WINDOW_SEC = 60
BIN_S = 0.05
WINDOW_BINS = int(WINDOW_SEC / BIN_S)  # 1200
N_BOOT = 2000
MIN_RUN_BINS = 20  # 1 s
RNG = np.random.default_rng(10)

INFLECTION_TYPES = ['peak', 'trough']
STATES = ['fed', 'fasted', 'fed-HFD']
PAIRS = [('fed', 'fasted'), ('fed', 'fed-HFD'), ('fasted', 'fed-HFD')]
STATE_COLOR = {'fed': '#1b9e77', 'fasted': '#d95f02', 'fed-HFD': '#7570b3'}

METRICS = {
    'curv_ACA': ('curvature', 'ACA'),
    'speed_ACA': ('speed', 'ACA'),
    'curv_LHA': ('curvature', 'LHA'),
    'speed_LHA': ('speed', 'LHA'),
}


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


def signal_for(metric_key, cv, sp):
    kind, region = METRICS[metric_key]
    if kind == 'curvature':
        return cv[region]
    return sp[region]


def extract_window(signal, center_bin, window_bins=WINDOW_BINS):
    """Return length 2*window_bins+1 array centered on center_bin; NaN-pad edges."""
    n = len(signal)
    out = np.full(2 * window_bins + 1, np.nan)
    s_lo = max(0, center_bin - window_bins)
    s_hi = min(n, center_bin + window_bins + 1)
    if s_hi <= s_lo:
        return out
    o_lo = window_bins - (center_bin - s_lo)
    o_hi = o_lo + (s_hi - s_lo)
    out[o_lo:o_hi] = signal[s_lo:s_hi]
    return out


def collect_windows():
    """Return dict[(metric, state, inflection_type)] -> list of (session, window).

    Also returns inflection_count per (state, inflection_type).
    """
    state_lookup = session_state_lookup()
    out = {(m, s, it): [] for m in METRICS for s in STATES for it in INFLECTION_TYPES}
    counts = {(s, it): 0 for s in STATES for it in INFLECTION_TYPES}

    for session, state in state_lookup.items():
        cv, sp, phases = load_session(session)
        for ph in phases:
            if ph['phase_type'] not in INFLECTION_TYPES:
                continue
            ib = ph.get('inflection_bin')
            if ib is None:
                continue
            counts[(state, ph['phase_type'])] += 1
            for mkey in METRICS:
                w = extract_window(signal_for(mkey, cv, sp), int(ib))
                out[(mkey, state, ph['phase_type'])].append((session, w))
    return out, counts


def session_pool(records):
    """records: list of (session, window). Return (sessions_arr, n_sessions x time)."""
    if not records:
        return None, None
    df = pd.DataFrame(records, columns=['session', 'window'])
    grouped = df.groupby('session')['window'].apply(lambda g: np.nanmean(np.stack(g.values), axis=0))
    sessions = grouped.index.values
    arr = np.stack(grouped.values)
    return sessions, arr


def bootstrap_mean_traj(arr, n_boot=N_BOOT, rng=None):
    """arr: (n_sess, T). Return mean (nanmean), 2.5/97.5 percentile arrays."""
    rng = rng or RNG
    n = arr.shape[0]
    if n == 0:
        return None, None, None
    obs = np.nanmean(arr, axis=0)
    boots = np.empty((n_boot, arr.shape[1]))
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = np.nanmean(arr[idx], axis=0)
    lo = np.nanpercentile(boots, 2.5, axis=0)
    hi = np.nanpercentile(boots, 97.5, axis=0)
    return obs, lo, hi


def bootstrap_diff_traj(arr_a, arr_b, n_boot=N_BOOT, rng=None):
    """Pairwise mean diff with bootstrap CI per timepoint."""
    rng = rng or RNG
    na, nb = arr_a.shape[0], arr_b.shape[0]
    obs = np.nanmean(arr_a, axis=0) - np.nanmean(arr_b, axis=0)
    boots = np.empty((n_boot, arr_a.shape[1]))
    for i in range(n_boot):
        ai = rng.integers(0, na, size=na)
        bi = rng.integers(0, nb, size=nb)
        boots[i] = np.nanmean(arr_a[ai], axis=0) - np.nanmean(arr_b[bi], axis=0)
    lo = np.nanpercentile(boots, 2.5, axis=0)
    hi = np.nanpercentile(boots, 97.5, axis=0)
    return obs, lo, hi


def find_divergence_onset(t_axis, ci_lo, ci_hi, min_run=MIN_RUN_BINS):
    """Earliest contiguous run of `min_run` bins where CI excludes zero. Returns time (s) or NaN."""
    sig = (ci_lo > 0) | (ci_hi < 0)
    n = len(sig)
    run = 0
    for i in range(n):
        if sig[i]:
            run += 1
            if run >= min_run:
                # onset is at the START of this run
                return t_axis[i - min_run + 1], i - min_run + 1
        else:
            run = 0
    return np.nan, -1


def main():
    print('Collecting peri-inflection windows...')
    windows, counts = collect_windows()
    print('Inflection counts (state, type):')
    for k, v in counts.items():
        print(f'  {k}: {v}')

    t_axis = np.arange(-WINDOW_BINS, WINDOW_BINS + 1) * BIN_S  # seconds, length 2401

    # ---- Per-state mean + CI per metric per inflection type
    onset_rows = []
    for mkey in METRICS:
        all_rows = []
        # Pool: per state x inflection type
        state_arrays = {}
        for it in INFLECTION_TYPES:
            for state in STATES:
                _, arr = session_pool(windows[(mkey, state, it)])
                if arr is None:
                    continue
                state_arrays[(state, it)] = arr
                obs, lo, hi = bootstrap_mean_traj(arr)
                for k, t in enumerate(t_axis):
                    all_rows.append(dict(metric=mkey, state=state, inflection_type=it,
                                         t_s=t, mean=obs[k], ci_lo=lo[k], ci_hi=hi[k],
                                         n_sessions=arr.shape[0]))
        pd.DataFrame(all_rows).to_csv(OUT_DATA / f'periinflection_{mkey}.csv', index=False)

        # Pairwise state diffs (pooled across inflection types as well as separately)
        diff_rows = []
        for it_label, inflection_filter in [('peak', ['peak']),
                                            ('trough', ['trough']),
                                            ('pooled', ['peak', 'trough'])]:
            # Construct per-state pooled arrays for this filter
            pooled = {}
            for state in STATES:
                segs = []
                for it in inflection_filter:
                    if (state, it) in state_arrays:
                        segs.append(state_arrays[(state, it)])
                if not segs:
                    continue
                # Stack per-session means across inflection types: simply concatenate sessions
                # (each session contributes its own per-inflection-type session mean)
                pooled[state] = np.vstack(segs)
            for sa, sb in PAIRS:
                if sa not in pooled or sb not in pooled:
                    continue
                obs, lo, hi = bootstrap_diff_traj(pooled[sa], pooled[sb])
                onset_t, onset_idx = find_divergence_onset(t_axis, lo, hi)
                onset_rows.append(dict(metric=mkey, contrast=f'{sa}_vs_{sb}',
                                       inflection_type=it_label,
                                       divergence_onset_s=onset_t,
                                       divergence_onset_bin=onset_idx,
                                       n_sessions_a=pooled[sa].shape[0],
                                       n_sessions_b=pooled[sb].shape[0]))
                for k, t in enumerate(t_axis):
                    diff_rows.append(dict(metric=mkey, contrast=f'{sa}_vs_{sb}',
                                          inflection_type=it_label, t_s=t,
                                          diff=obs[k], ci_lo=lo[k], ci_hi=hi[k],
                                          sig=bool((lo[k] > 0) or (hi[k] < 0))))
        pd.DataFrame(diff_rows).to_csv(OUT_DATA / f'periinflection_state_diff_{mkey}.csv', index=False)

    pd.DataFrame(onset_rows).to_csv(OUT_DATA / 'divergence_onset_summary.csv', index=False)

    print('\n=== Divergence-onset summary ===')
    onset_df = pd.DataFrame(onset_rows)
    print(onset_df.to_string(index=False))

    # ---- Figure: ACA curvature peri-inflection per state
    def make_periinflection_figure(mkey, title, fig_path):
        fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
        # rows: peak, trough; cols: trajectories, fed-vs-others diff
        for r, it in enumerate(INFLECTION_TYPES):
            ax = axes[r, 0]
            for state in STATES:
                _, arr = session_pool(windows[(mkey, state, it)])
                if arr is None:
                    continue
                obs, lo, hi = bootstrap_mean_traj(arr)
                ax.plot(t_axis, obs, color=STATE_COLOR[state], lw=1.5,
                        label=f'{state} (n={arr.shape[0]})')
                ax.fill_between(t_axis, lo, hi, color=STATE_COLOR[state], alpha=0.18)
            ax.axvline(0, color='k', lw=0.7, linestyle='--')
            ax.set_title(f'{it} — peri-inflection {mkey}')
            ax.set_ylabel(mkey)
            ax.legend(loc='best', fontsize=8)
            ax.grid(alpha=0.25)

            ax = axes[r, 1]
            diff_csv = pd.read_csv(OUT_DATA / f'periinflection_state_diff_{mkey}.csv')
            for sa, sb in [('fed', 'fasted'), ('fed', 'fed-HFD')]:
                d = diff_csv[(diff_csv.contrast == f'{sa}_vs_{sb}')
                             & (diff_csv.inflection_type == it)]
                if d.empty:
                    continue
                color = '#d95f02' if sb == 'fasted' else '#7570b3'
                ax.plot(d.t_s, d['diff'], color=color, lw=1.4, label=f'{sa}-{sb}')
                ax.fill_between(d.t_s, d.ci_lo, d.ci_hi, color=color, alpha=0.18)
                # mark divergence onset for this contrast/inflection_type
                onset_match = onset_df[(onset_df.metric == mkey)
                                       & (onset_df.contrast == f'{sa}_vs_{sb}')
                                       & (onset_df.inflection_type == it)]
                if not onset_match.empty and not np.isnan(onset_match.divergence_onset_s.iloc[0]):
                    ax.axvline(onset_match.divergence_onset_s.iloc[0],
                               color=color, lw=1, linestyle=':', alpha=0.8)
            ax.axvline(0, color='k', lw=0.7, linestyle='--')
            ax.axhline(0, color='k', lw=0.5)
            ax.set_title(f'{it} — pairwise difference (95% bootstrap CI)')
            ax.set_ylabel('mean_diff')
            ax.legend(loc='best', fontsize=8)
            ax.grid(alpha=0.25)
        axes[1, 0].set_xlabel('time relative to inflection (s)')
        axes[1, 1].set_xlabel('time relative to inflection (s)')
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)

    make_periinflection_figure('curv_ACA',
                               'Peri-inflection ACA curvature per diet state',
                               OUT_FIG / 'periinflection_curvature_aca.png')
    make_periinflection_figure('speed_LHA',
                               'Peri-inflection LHA speed per diet state',
                               OUT_FIG / 'periinflection_speed_lha.png')
    make_periinflection_figure('speed_ACA',
                               'Peri-inflection ACA speed per diet state',
                               OUT_FIG / 'periinflection_speed_aca.png')
    make_periinflection_figure('curv_LHA',
                               'Peri-inflection LHA curvature per diet state',
                               OUT_FIG / 'periinflection_curvature_lha.png')

    # Divergence-onset overview figure: one bar per (metric, contrast, inflection_type)
    fig, ax = plt.subplots(figsize=(10, 6))
    onset_plot = onset_df.copy()
    onset_plot['label'] = (onset_plot.metric + ' / ' + onset_plot.inflection_type
                           + ' / ' + onset_plot.contrast)
    onset_plot = onset_plot.sort_values(['metric', 'inflection_type', 'contrast'])
    finite = onset_plot[~onset_plot.divergence_onset_s.isna()]
    nan_rows = onset_plot[onset_plot.divergence_onset_s.isna()]
    ys = np.arange(len(onset_plot))
    for y, (_, r) in zip(ys, onset_plot.iterrows()):
        if np.isnan(r.divergence_onset_s):
            ax.text(0, y, '  no onset (CI never excludes 0 for ≥1s run)',
                    fontsize=8, va='center', color='gray')
        else:
            ax.barh(y, r.divergence_onset_s, color='#1b9e77' if r.divergence_onset_s < 0 else '#d95f02')
            ax.text(r.divergence_onset_s, y, f' {r.divergence_onset_s:+.1f}s',
                    fontsize=8, va='center')
    ax.set_yticks(ys)
    ax.set_yticklabels(onset_plot.label.values, fontsize=8)
    ax.axvline(0, color='k', lw=1, linestyle='--')
    ax.set_xlabel('divergence onset (s relative to inflection)')
    ax.set_title('Divergence onset per metric / inflection type / state contrast\n'
                 'green = pre-inflection, orange = post-inflection')
    ax.grid(axis='x', alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_FIG / 'divergence_onset.png', dpi=150)
    plt.close(fig)

    print(f'\nWrote {OUT_DATA}')
    print(f'Wrote {OUT_FIG}')


if __name__ == '__main__':
    main()
