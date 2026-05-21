"""
Step 8: Replicate the peri-inflection event-locked analysis from
dp_entropy_neural_signatures.py on speed and curvature, asking whether
HFD also shows a flat / wrong-direction signature on these metrics
(supporting the FR/PC1 finding from a non-FR angle).

Procedure (matches dp_entropy_neural_signatures.py windowing):
  - PERI_WINDOW = 12 entropy steps (+/- 120 s).
  - For each peak/trough at entropy step idx:
      pre window  = idx - 6 .. idx (excl)         [60 s before]
      at  window  = idx                              [event]
      post window = idx + 1 .. idx + 7              [60 s after]
  - Metrics: ACA speed, LHA speed, ACA curvature, LHA curvature.
  - Z-score each metric within session before computing window means.
  - delta_z = post_mean_z - pre_mean_z.

Pool by state. Test:
  (i)  fed, fasted, HFD each: is delta_z significantly different from 0?
  (ii) HFD vs fed and HFD vs fasted: is HFD's |delta_z| smaller (i.e., HFD is flatter)?
"""
import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, mannwhitneyu
from scipy.ndimage import gaussian_filter1d

REPO = Path(__file__).resolve().parent.parent.parent
S1D = REPO / 'data' / 'dynamics_stage1'
OUTDIR = REPO / 'data' / 'drilldown_curvature'
FIGDIR = REPO / 'figures' / 'drilldown_curvature'
OUTDIR.mkdir(parents=True, exist_ok=True)
FIGDIR.mkdir(parents=True, exist_ok=True)

PERI_WINDOW = 12          # entropy steps (+/- 120 s)
PRE_SL = slice(PERI_WINDOW - 6, PERI_WINDOW)        # 60 s pre
POST_SL = slice(PERI_WINDOW + 1, PERI_WINDOW + 7)   # 60 s post
EXCLUDE_SESSIONS = {13, 23, 24}


def load_session(session):
    phz = np.load(S1D / f'session_{session}_phase_data.npz', allow_pickle=True)
    spd = np.load(S1D / f'session_{session}_speed.npy', allow_pickle=True).item()
    crv = np.load(S1D / f'session_{session}_curvature.npy', allow_pickle=True).item()
    return phz, spd, crv


def aggregate_to_entropy_steps(neural_trace, bin_centers, ent_t):
    """Aggregate a neural-bin (50 ms) trace into entropy-step time points by
    nearest-neighbor / interpolation. Output length matches len(ent_t)."""
    # The neural trace is at bin_centers; entropy is at ent_t (10 s spacing).
    # Linear interp captures the slow envelope.
    return np.interp(ent_t, bin_centers, neural_trace,
                     left=neural_trace[0], right=neural_trace[-1])


def zscore(x):
    m = np.nanmean(x); s = np.nanstd(x)
    if s == 0 or not np.isfinite(s):
        return np.zeros_like(x)
    return (x - m) / s


def per_event_windows(metric_z, peaks, troughs):
    """Return dict of (n_events, 2*PERI+1) arrays for peaks and troughs,
    plus pre/at/post values."""
    out = {}
    for label, idxs in [('peak', peaks), ('trough', troughs)]:
        rows = []
        for i in idxs:
            if i < PERI_WINDOW or i >= len(metric_z) - PERI_WINDOW:
                continue
            rows.append(metric_z[i - PERI_WINDOW: i + PERI_WINDOW + 1])
        if rows:
            arr = np.array(rows)
            out[label] = dict(
                windows=arr,
                at=arr[:, PERI_WINDOW],
                pre=np.nanmean(arr[:, PRE_SL], axis=1),
                post=np.nanmean(arr[:, POST_SL], axis=1),
            )
        else:
            out[label] = None
    return out


def main():
    summary = pd.read_csv(S1D / 'all_sessions_summary.csv')
    state_lookup = summary.groupby('session').state.first().to_dict()
    sessions = sorted(s for s in summary.session.unique()
                      if s not in EXCLUDE_SESSIONS)
    print(f'Processing {len(sessions)} sessions...')

    rows = []   # one row per (session, inflection, metric, event)
    for snum in sessions:
        try:
            phz, spd, crv = load_session(snum)
        except FileNotFoundError as e:
            print(f'  S{snum}: cache missing ({e}), skip')
            continue
        peaks = list(map(int, phz['peaks']))
        troughs = list(map(int, phz['troughs']))
        ent_t = phz['ent_t']
        bin_centers = phz['bin_centers']

        if len(peaks) == 0 and len(troughs) == 0:
            continue

        # speed/curv arrays are length T-1 / T-2 (one/two less than n_neural_bins).
        # Pad to match bin_centers length by repeating last value.
        def pad_to_bins(a, target_len):
            if len(a) == target_len:
                return a
            pad = np.repeat(a[-1], target_len - len(a))
            return np.concatenate([a, pad])

        T = len(bin_centers)
        speed_aca = pad_to_bins(spd['ACA'], T)
        speed_lha = pad_to_bins(spd['LHA'], T)
        curv_aca = pad_to_bins(crv['ACA'], T)
        curv_lha = pad_to_bins(crv['LHA'], T)

        # Aggregate to entropy step resolution
        metrics_at_ent = {
            'ACA speed': aggregate_to_entropy_steps(speed_aca, bin_centers, ent_t),
            'LHA speed': aggregate_to_entropy_steps(speed_lha, bin_centers, ent_t),
            'ACA curv':  aggregate_to_entropy_steps(curv_aca, bin_centers, ent_t),
            'LHA curv':  aggregate_to_entropy_steps(curv_lha, bin_centers, ent_t),
        }

        state = state_lookup.get(snum, 'unknown')
        for mname, mvals in metrics_at_ent.items():
            mz = zscore(mvals)
            wins = per_event_windows(mz, peaks, troughs)
            for inflection in ['peak', 'trough']:
                w = wins.get(inflection)
                if w is None:
                    continue
                for ev_i in range(len(w['at'])):
                    rows.append(dict(
                        session=int(snum), state=state, metric=mname,
                        inflection=inflection,
                        pre_z=float(w['pre'][ev_i]),
                        at_z=float(w['at'][ev_i]),
                        post_z=float(w['post'][ev_i]),
                        delta_z=float(w['post'][ev_i] - w['pre'][ev_i]),
                    ))
        print(f"  S{snum} ({state}): n_peaks={len(peaks)}, n_troughs={len(troughs)}")

    df = pd.DataFrame(rows)
    df.to_csv(OUTDIR / 'step8_periinflection_speed_curv_events.csv', index=False)
    print(f"Wrote {OUTDIR / 'step8_periinflection_speed_curv_events.csv'}")
    print(f'Total events: {len(df)}')

    # ---- Pooled stats (matches dp_entropy_inflection_pooled_stats format) ----
    pooled_rows = []
    for state in ['fed', 'fasted', 'fed-HFD', 'all']:
        sub = df if state == 'all' else df[df.state == state]
        if len(sub) == 0:
            continue
        for inflection in ['peak', 'trough']:
            for metric in ['ACA speed', 'LHA speed', 'ACA curv', 'LHA curv']:
                events = sub[(sub.inflection == inflection) & (sub.metric == metric)]
                if len(events) < 5:
                    continue
                # MWU: at_z vs zero (one-sample equiv via vs zero array)
                # Wilcoxon: paired pre vs post equiv to delta_z vs zero
                try:
                    wilc_p = wilcoxon(events.pre_z.values, events.post_z.values).pvalue
                except ValueError:
                    wilc_p = np.nan
                try:
                    mwu_p = mannwhitneyu(events.at_z.values,
                                          np.zeros(len(events))).pvalue
                except ValueError:
                    mwu_p = np.nan
                pooled_rows.append(dict(
                    group=state, inflection=inflection, metric=metric,
                    n_events=int(len(events)),
                    pre_mean_z=float(np.mean(events.pre_z)),
                    at_mean_z=float(np.mean(events.at_z)),
                    post_mean_z=float(np.mean(events.post_z)),
                    delta_z=float(np.mean(events.delta_z)),
                    p_mwu=float(mwu_p) if np.isfinite(mwu_p) else np.nan,
                    p_wilcoxon=float(wilc_p) if np.isfinite(wilc_p) else np.nan,
                ))
    pooled = pd.DataFrame(pooled_rows)
    pooled.to_csv(OUTDIR / 'step8_periinflection_speed_curv_pooled.csv', index=False)
    print(f"Wrote {OUTDIR / 'step8_periinflection_speed_curv_pooled.csv'}")

    # ---- Between-state contrasts on delta_z ----
    print('\n========== HFD-vs-fed and HFD-vs-fasted MWU on delta_z ==========')
    contrast_rows = []
    for inflection in ['peak', 'trough']:
        print(f'\n  -- {inflection.upper()}S --')
        for metric in ['ACA speed', 'LHA speed', 'ACA curv', 'LHA curv']:
            print(f'\n    {metric}:')
            sub = df[(df.inflection == inflection) & (df.metric == metric)]
            for s_a, s_b in [('fed-HFD', 'fed'), ('fed-HFD', 'fasted'),
                              ('fed', 'fasted')]:
                a = sub[sub.state == s_a].delta_z.values
                b = sub[sub.state == s_b].delta_z.values
                if len(a) < 5 or len(b) < 5:
                    continue
                try:
                    p = mannwhitneyu(a, b).pvalue
                except ValueError:
                    p = np.nan
                tag = '***' if (np.isfinite(p) and p < 0.05) else \
                      ('+' if (np.isfinite(p) and p < 0.10) else 'ns')
                print(f"      {s_a:8s} (mean {np.mean(a):+.2f}) vs "
                      f"{s_b:8s} (mean {np.mean(b):+.2f})  "
                      f"MWU p={p:.3g}  {tag}")
                contrast_rows.append(dict(
                    inflection=inflection, metric=metric,
                    state_a=s_a, state_b=s_b,
                    n_a=int(len(a)), n_b=int(len(b)),
                    mean_a=float(np.mean(a)), mean_b=float(np.mean(b)),
                    mwu_p=float(p) if np.isfinite(p) else np.nan,
                ))
    pd.DataFrame(contrast_rows).to_csv(
        OUTDIR / 'step8_periinflection_state_contrasts.csv', index=False)
    print(f"Wrote {OUTDIR / 'step8_periinflection_state_contrasts.csv'}")

    # ---- Print results ----
    print('\n========== Pooled peri-inflection delta_z by state ==========')
    for inflection in ['peak', 'trough']:
        print(f'\n  -- {inflection.upper()}S --')
        sub = pooled[pooled.inflection == inflection]
        for metric in ['ACA speed', 'LHA speed', 'ACA curv', 'LHA curv']:
            print(f'\n    {metric}:')
            for state in ['fed', 'fasted', 'fed-HFD']:
                row = sub[(sub.metric == metric) & (sub.group == state)]
                if len(row) == 0:
                    continue
                r = row.iloc[0]
                tag = '***' if (np.isfinite(r.p_wilcoxon) and r.p_wilcoxon < 0.05) else \
                      ('+' if (np.isfinite(r.p_wilcoxon) and r.p_wilcoxon < 0.10) else 'ns')
                print(f"      {state:8s}  n={int(r.n_events):3d}  "
                      f"pre={r.pre_mean_z:+.2f}  at={r.at_mean_z:+.2f}  "
                      f"post={r.post_mean_z:+.2f}  delta={r.delta_z:+.2f}  "
                      f"p_wilc={r.p_wilcoxon:.3g}  {tag}")


if __name__ == '__main__':
    main()
