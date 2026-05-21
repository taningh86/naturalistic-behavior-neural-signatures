"""
Per-session view of the tight ±5 s peri-inflection analysis.

For each session, compute the mean z-scored neural metric at each peak/trough
(bin nearest t=0), then plot session-level peak vs trough and report
directional consistency with the expected opposition:

    SP: RSP ↑ at peaks, LHA ↑ at troughs
    DP: ACA ↑ at peaks, LHA ↑ at troughs

Produces:
  data/entropy_inflection_tight_per_session_sp.csv
  data/entropy_inflection_tight_per_session_dp.csv
  figures/entropy_inflection_tight_per_session_sp.png
  figures/entropy_inflection_tight_per_session_dp.png
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, binomtest

# Reuse everything from the main tight-window script
import entropy_inflection_tight_window as T

REPO = Path(r"H:\NPX ANALYSIS REPO")
FIG = REPO / "figures"
DAT = REPO / "data"


def _per_session_values(trace, centers, peak_times, trough_times):
    """Return lists of at-inflection z-values for peaks and troughs."""
    pwin = T.extract_windows(trace, centers, peak_times)
    twin = T.extract_windows(trace, centers, trough_times)
    n_pre = int(T.PRE_SEC / T.BIN_SEC)
    pv = [w[n_pre] for w in pwin if not np.isnan(w[n_pre])]
    tv = [w[n_pre] for w in twin if not np.isnan(w[n_pre])]
    return pv, tv


def run_sp_per_session():
    print("\n=== SP per-session ===")
    sp_cfg = T.cfg["single_probe"]["coordinates_1"]["mouse01"]["sessions"]
    sp_meta = {
        1: ('fed', 'exploration'), 2: ('fed', 'foraging'),
        3: ('fed', 'exploration'), 4: ('fed', 'foraging'),
        5: ('fasted', 'exploration'), 6: ('fasted', 'foraging'),
        7: ('fasted', 'exploration'), 8: ('fasted', 'foraging'),
    }
    metrics = ['Velocity', 'LHA FR', 'RSP FR', 'LHA PC1', 'RSP PC1']
    rows = []

    for snum in range(1, 9):
        state, phase = sp_meta[snum]
        sc = sp_cfg[f"session_{snum}"]
        bp = sc.get('behavior')
        if not bp or not Path(bp).exists():
            continue
        time_vals, vel, zones = T.load_behavior_sp(bp)
        ent_t, ent_v, _ = T.compute_entropy(zones, time_vals, vel,
                                            T.ENTROPY_WINDOW_SEC, T.ENTROPY_STEP_SEC)
        if len(ent_v) < 15:
            continue
        peaks_i, troughs_i = T.find_inflections(ent_v, T.SMOOTH_SIGMA, T.MIN_AMPLITUDE)
        peak_times = ent_t[peaks_i] if peaks_i else np.array([])
        trough_times = ent_t[troughs_i] if troughs_i else np.array([])

        lha_ids, rsp_ids = T.sp_units(sc['sorted'])
        import spikeinterface.extractors as se
        try:
            sorting = se.read_kilosort(sc['sorted'])
        except Exception as e:
            print(f"  S{snum}: sort err {e}")
            continue
        avail = set(sorting.get_unit_ids())
        lha_ids = np.array([u for u in lha_ids if u in avail])
        rsp_ids = np.array([u for u in rsp_ids if u in avail])
        if len(lha_ids) < 2 or len(rsp_ids) < 2:
            continue

        max_t = float(time_vals[-1])
        centers, lha_pop, lha_pc1 = T.compute_fine_neural(sorting, lha_ids, max_t)
        _, rsp_pop, rsp_pc1 = T.compute_fine_neural(sorting, rsp_ids, max_t)
        vel_interp = np.interp(centers, time_vals, vel)

        tracer = {
            'Velocity': vel_interp, 'LHA FR': lha_pop, 'RSP FR': rsp_pop,
            'LHA PC1': lha_pc1, 'RSP PC1': rsp_pc1,
        }
        print(f"  S{snum} {state}/{phase}: {len(peak_times)} peaks, {len(trough_times)} troughs")
        for m, tr in tracer.items():
            pv, tv = _per_session_values(tr, centers, peak_times, trough_times)
            rows.append({
                'session': snum, 'state': state, 'phase': phase, 'metric': m,
                'n_peak': len(pv), 'n_trough': len(tv),
                'peak_mean': float(np.mean(pv)) if pv else np.nan,
                'peak_sem': float(np.std(pv) / np.sqrt(len(pv))) if pv else np.nan,
                'trough_mean': float(np.mean(tv)) if tv else np.nan,
                'trough_sem': float(np.std(tv) / np.sqrt(len(tv))) if tv else np.nan,
            })
    df = pd.DataFrame(rows)
    df.to_csv(DAT / 'entropy_inflection_tight_per_session_sp.csv', index=False)
    return df


def run_dp_per_session():
    print("\n=== DP per-session ===")
    dp_cfg = T.cfg["double_probe"]["coordinates_1"]["mouse01"]["sessions"]
    metrics = ['Velocity', 'ACA FR', 'LHA FR', 'ACA PC1', 'LHA PC1']
    rows = []

    for skey, sval in dp_cfg.items():
        snum = int(skey.split('_')[1])
        if snum in T.DP_SKIP:
            continue
        state = sval.get('state')
        phase = sval.get('phase')
        p0 = sval.get('probe_0_aca', {}).get('sorted')
        p1 = sval.get('probe_1_lha_rsp', {}).get('sorted')
        beh = sval.get('behavior')
        if not (p0 and p1 and beh):
            continue
        if not (Path(p0).exists() and Path(beh).exists()):
            continue

        try:
            time_vals, vel, zones = T.load_behavior_dp(beh)
        except Exception as e:
            print(f"  S{snum}: beh err {e}")
            continue

        ent_t, ent_v, _ = T.compute_entropy(zones, time_vals, vel,
                                            T.ENTROPY_WINDOW_SEC, T.ENTROPY_STEP_SEC)
        if len(ent_v) < 15:
            continue
        peaks_i, troughs_i = T.find_inflections(ent_v, T.SMOOTH_SIGMA, T.MIN_AMPLITUDE)
        peak_times = ent_t[peaks_i] if peaks_i else np.array([])
        trough_times = ent_t[troughs_i] if troughs_i else np.array([])

        aca_ids = T.dp_units_aca(p0)
        lha_ids = T.dp_units_lha(p1)
        import spikeinterface.extractors as se
        try:
            sorting_p0 = se.read_kilosort(p0)
            aca_ids = np.array([u for u in aca_ids if u in set(sorting_p0.get_unit_ids())])
        except Exception:
            aca_ids = np.array([]); sorting_p0 = None
        try:
            sorting_p1 = se.read_kilosort(p1)
            lha_ids = np.array([u for u in lha_ids if u in set(sorting_p1.get_unit_ids())])
        except Exception:
            lha_ids = np.array([]); sorting_p1 = None

        if (sorting_p0 is None or len(aca_ids) < 2) and \
           (sorting_p1 is None or len(lha_ids) < 2):
            continue

        max_t = float(time_vals[-1])
        centers_p0, aca_pop, aca_pc1 = (None, None, None)
        if sorting_p0 is not None and len(aca_ids) >= 2:
            centers_p0, aca_pop, aca_pc1 = T.compute_fine_neural(sorting_p0, aca_ids, max_t)
        centers_p1, lha_pop, lha_pc1 = (None, None, None)
        if sorting_p1 is not None and len(lha_ids) >= 2:
            centers_p1, lha_pop, lha_pc1 = T.compute_fine_neural(sorting_p1, lha_ids, max_t)

        centers = centers_p0 if centers_p0 is not None else centers_p1
        vel_interp = np.interp(centers, time_vals, vel)
        tracer = {'Velocity': vel_interp}
        if aca_pop is not None:
            tracer['ACA FR'] = aca_pop
            tracer['ACA PC1'] = aca_pc1
        if lha_pop is not None:
            if centers_p1 is not None and not np.array_equal(centers, centers_p1):
                lha_pop = np.interp(centers, centers_p1, lha_pop)
                lha_pc1 = np.interp(centers, centers_p1, lha_pc1)
            tracer['LHA FR'] = lha_pop
            tracer['LHA PC1'] = lha_pc1

        print(f"  S{snum} {state}/{phase}: {len(peak_times)} peaks, {len(trough_times)} troughs")
        for m in metrics:
            if m not in tracer:
                rows.append({'session': snum, 'state': state, 'phase': phase,
                             'metric': m, 'n_peak': 0, 'n_trough': 0,
                             'peak_mean': np.nan, 'peak_sem': np.nan,
                             'trough_mean': np.nan, 'trough_sem': np.nan})
                continue
            pv, tv = _per_session_values(tracer[m], centers, peak_times, trough_times)
            rows.append({
                'session': snum, 'state': state, 'phase': phase, 'metric': m,
                'n_peak': len(pv), 'n_trough': len(tv),
                'peak_mean': float(np.mean(pv)) if pv else np.nan,
                'peak_sem': float(np.std(pv) / np.sqrt(len(pv))) if pv else np.nan,
                'trough_mean': float(np.mean(tv)) if tv else np.nan,
                'trough_sem': float(np.std(tv) / np.sqrt(len(tv))) if tv else np.nan,
            })
    df = pd.DataFrame(rows)
    df.to_csv(DAT / 'entropy_inflection_tight_per_session_dp.csv', index=False)
    return df


# Expected directional sign for opposition:
# value = peak_mean - trough_mean at a given metric
# If opposition holds:
#   Cortical (RSP, ACA) should RISE at peaks (high entropy) → peak > trough → sign > 0
#   LHA should RISE at troughs (low entropy) → trough > peak → sign < 0
EXPECTED_SIGN = {
    'RSP FR': +1, 'RSP PC1': +1,
    'ACA FR': +1, 'ACA PC1': +1,
    'LHA FR': -1, 'LHA PC1': -1,
    'Velocity': +1,  # varied movement at peaks = movement
}


def _direction_stats(df, metric):
    sub = df[df['metric'] == metric].dropna(subset=['peak_mean', 'trough_mean'])
    if len(sub) == 0:
        return None
    diff = sub['peak_mean'] - sub['trough_mean']
    expected = EXPECTED_SIGN.get(metric, +1)
    consistent = ((diff * expected) > 0).sum()
    total = len(diff)
    # Wilcoxon signed-rank on paired peak vs trough
    try:
        w = wilcoxon(sub['peak_mean'].values, sub['trough_mean'].values).pvalue
    except ValueError:
        w = np.nan
    # Binomial test: probability of getting `consistent` out of `total` under p=0.5
    b = binomtest(consistent, total, p=0.5, alternative='two-sided').pvalue if total > 0 else np.nan
    return dict(n=total, consistent=int(consistent),
                frac=float(consistent / total) if total else np.nan,
                wilcoxon_paired_p=float(w),
                binomial_p=float(b),
                mean_peak=float(sub['peak_mean'].mean()),
                mean_trough=float(sub['trough_mean'].mean()),
                mean_diff=float(diff.mean()))


STATE_COLOR = {'fed': '#1F77B4', 'fasted': '#D62728', 'fed-HFD': '#9467BD'}


def plot_per_session(df, metrics, title, out_path):
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 5.2),
                             constrained_layout=True)
    for ax, m in zip(axes, metrics):
        sub = df[df['metric'] == m].dropna(subset=['peak_mean', 'trough_mean'])
        if len(sub) == 0:
            ax.set_title(f"{m}\n(no data)")
            continue
        # Paired lines per session
        for _, row in sub.iterrows():
            c = STATE_COLOR.get(row['state'], '#888888')
            ax.plot([0, 1], [row['peak_mean'], row['trough_mean']],
                    color=c, alpha=0.8, lw=1.6, marker='o', markersize=6,
                    markerfacecolor=c, markeredgecolor='k', markeredgewidth=0.4)
            # session label at right
            ax.annotate(f"S{int(row['session'])}", xy=(1.03, row['trough_mean']),
                        fontsize=8, color=c, va='center')
        # Means
        ax.plot([0, 1], [sub['peak_mean'].mean(), sub['trough_mean'].mean()],
                color='k', lw=2.6, marker='D', markersize=9, alpha=0.9, zorder=5)
        ax.axhline(0, color='gray', lw=0.6, alpha=0.5)
        ax.set_xticks([0, 1]); ax.set_xticklabels(['peak', 'trough'], fontsize=12)
        ax.set_xlim(-0.25, 1.35)
        ax.set_ylabel('at-inflection z-score', fontsize=11)
        stats = _direction_stats(df, m)
        if stats:
            txt = (f"n={stats['n']} sessions\n"
                   f"expected dir: {'peak>trough' if EXPECTED_SIGN.get(m,1) > 0 else 'trough>peak'}\n"
                   f"consistent: {stats['consistent']}/{stats['n']}"
                   f" ({100*stats['frac']:.0f}%)\n"
                   f"Wilcoxon p={stats['wilcoxon_paired_p']:.3f}\n"
                   f"binomial p={stats['binomial_p']:.3f}")
        else:
            txt = "no data"
        ax.set_title(f"{m}\n{txt}", fontsize=10)
        ax.grid(alpha=0.2)

    # Legend for states
    handles = [plt.Line2D([0], [0], color=STATE_COLOR[s], lw=2, marker='o',
                          label=s) for s in STATE_COLOR if (df['state'] == s).any()]
    if handles:
        fig.legend(handles=handles, loc='upper right', fontsize=10, frameon=True)
    fig.suptitle(title, fontsize=15, fontweight='bold')
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    sp_df = run_sp_per_session()
    plot_per_session(sp_df,
                     ['Velocity', 'LHA FR', 'RSP FR', 'LHA PC1', 'RSP PC1'],
                     'Single-probe per-session peak vs trough — tight ±5 s window',
                     FIG / 'entropy_inflection_tight_per_session_sp.png')

    dp_df = run_dp_per_session()
    plot_per_session(dp_df,
                     ['Velocity', 'ACA FR', 'LHA FR', 'ACA PC1', 'LHA PC1'],
                     'Dual-probe per-session peak vs trough — tight ±5 s window',
                     FIG / 'entropy_inflection_tight_per_session_dp.png')

    # Print summary
    print("\n" + "=" * 80)
    print("DIRECTIONAL CONSISTENCY SUMMARY")
    print("=" * 80)
    for df, name, metrics in [
        (sp_df, 'SP', ['Velocity', 'LHA FR', 'RSP FR', 'LHA PC1', 'RSP PC1']),
        (dp_df, 'DP', ['Velocity', 'ACA FR', 'LHA FR', 'ACA PC1', 'LHA PC1']),
    ]:
        print(f"\n--- {name} ---")
        for m in metrics:
            s = _direction_stats(df, m)
            if s is None:
                continue
            expected = '>' if EXPECTED_SIGN.get(m, 1) > 0 else '<'
            print(f"  {m:10s} (expect peak{expected}trough): "
                  f"consistent {s['consistent']}/{s['n']} "
                  f"({100*s['frac']:.0f}%)  "
                  f"peak mean={s['mean_peak']:+.2f}  trough mean={s['mean_trough']:+.2f}  "
                  f"paired Wilcoxon p={s['wilcoxon_paired_p']:.3f}  "
                  f"binom p={s['binomial_p']:.3f}")


if __name__ == '__main__':
    main()
