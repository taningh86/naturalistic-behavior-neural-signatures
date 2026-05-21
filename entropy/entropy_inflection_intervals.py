"""
Measure peak <-> trough intervals per session to check whether a +/-120 s window
around each inflection would overlap neighboring inflections.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import entropy_inflection_tight_window as T

FIG = Path(r"H:\NPX ANALYSIS REPO\figures")
DAT = Path(r"H:\NPX ANALYSIS REPO\data")


def intervals_for(peak_times, trough_times):
    """Given arrays of peak and trough times, return:
      - nearest-opposite: for each peak, distance to nearest trough (and vice versa)
      - successive: consecutive |diff| after merging+sorting all inflections
    """
    nearest_p_to_t = []
    if len(peak_times) and len(trough_times):
        for p in peak_times:
            nearest_p_to_t.append(np.min(np.abs(trough_times - p)))
    nearest_t_to_p = []
    if len(peak_times) and len(trough_times):
        for t in trough_times:
            nearest_t_to_p.append(np.min(np.abs(peak_times - t)))
    # Merge and compute consecutive (any-type) intervals
    all_inf = np.sort(np.concatenate([peak_times, trough_times]))
    cons = np.diff(all_inf) if len(all_inf) > 1 else np.array([])
    return (np.array(nearest_p_to_t), np.array(nearest_t_to_p), cons)


def run_sp():
    sp_cfg = T.cfg["single_probe"]["coordinates_1"]["mouse01"]["sessions"]
    sp_meta = {1:'fed/exp',2:'fed/for',3:'fed/exp',4:'fed/for',
               5:'fasted/exp',6:'fasted/for',7:'fasted/exp',8:'fasted/for'}
    rows = []
    for snum in range(1, 9):
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
        ppt, ptp, cons = intervals_for(peak_times, trough_times)
        rows.append({
            'dataset': 'SP', 'session': snum, 'label': sp_meta[snum],
            'n_peak': len(peak_times), 'n_trough': len(trough_times),
            'median_consecutive_s': float(np.median(cons)) if len(cons) else np.nan,
            'min_consecutive_s': float(np.min(cons)) if len(cons) else np.nan,
            'pct25_consecutive_s': float(np.percentile(cons, 25)) if len(cons) else np.nan,
            'pct75_consecutive_s': float(np.percentile(cons, 75)) if len(cons) else np.nan,
            'median_nearest_opposite_s': float(np.median(np.concatenate([ppt, ptp])))
                if (len(ppt) + len(ptp)) else np.nan,
            'min_nearest_opposite_s': float(np.min(np.concatenate([ppt, ptp])))
                if (len(ppt) + len(ptp)) else np.nan,
        })
    return pd.DataFrame(rows), peak_times, trough_times


def run_dp():
    dp_cfg = T.cfg["double_probe"]["coordinates_1"]["mouse01"]["sessions"]
    rows = []
    for skey, sval in dp_cfg.items():
        snum = int(skey.split('_')[1])
        if snum in T.DP_SKIP:
            continue
        beh = sval.get('behavior')
        if not beh or not Path(beh).exists():
            continue
        state = sval.get('state'); phase = sval.get('phase')
        try:
            time_vals, vel, zones = T.load_behavior_dp(beh)
        except Exception:
            continue
        ent_t, ent_v, _ = T.compute_entropy(zones, time_vals, vel,
                                            T.ENTROPY_WINDOW_SEC, T.ENTROPY_STEP_SEC)
        if len(ent_v) < 15:
            continue
        peaks_i, troughs_i = T.find_inflections(ent_v, T.SMOOTH_SIGMA, T.MIN_AMPLITUDE)
        peak_times = ent_t[peaks_i] if peaks_i else np.array([])
        trough_times = ent_t[troughs_i] if troughs_i else np.array([])
        ppt, ptp, cons = intervals_for(peak_times, trough_times)
        rows.append({
            'dataset': 'DP', 'session': snum, 'label': f"{state}/{phase}",
            'n_peak': len(peak_times), 'n_trough': len(trough_times),
            'median_consecutive_s': float(np.median(cons)) if len(cons) else np.nan,
            'min_consecutive_s': float(np.min(cons)) if len(cons) else np.nan,
            'pct25_consecutive_s': float(np.percentile(cons, 25)) if len(cons) else np.nan,
            'pct75_consecutive_s': float(np.percentile(cons, 75)) if len(cons) else np.nan,
            'median_nearest_opposite_s': float(np.median(np.concatenate([ppt, ptp])))
                if (len(ppt) + len(ptp)) else np.nan,
            'min_nearest_opposite_s': float(np.min(np.concatenate([ppt, ptp])))
                if (len(ppt) + len(ptp)) else np.nan,
        })
    return pd.DataFrame(rows)


def pooled_intervals_for_hist(df_rows_maker):
    """Re-runs to get raw consecutive + nearest-opposite arrays pooled."""
    all_cons = []
    all_nearest = []
    return all_cons, all_nearest


def collect_raw(cfg_block, loader, skip=None):
    cons_all = []
    nearest_all = []
    for skey, sval in cfg_block.items():
        snum = int(skey.split('_')[1])
        if skip and snum in skip:
            continue
        beh = sval.get('behavior') if isinstance(sval, dict) else None
        if beh is None:
            continue
        if not Path(beh).exists():
            continue
        try:
            time_vals, vel, zones = loader(beh)
        except Exception:
            continue
        ent_t, ent_v, _ = T.compute_entropy(zones, time_vals, vel,
                                            T.ENTROPY_WINDOW_SEC, T.ENTROPY_STEP_SEC)
        if len(ent_v) < 15:
            continue
        pi, ti = T.find_inflections(ent_v, T.SMOOTH_SIGMA, T.MIN_AMPLITUDE)
        pt = ent_t[pi] if pi else np.array([])
        tt = ent_t[ti] if ti else np.array([])
        ppt, ptp, cons = intervals_for(pt, tt)
        if len(cons):
            cons_all.append(cons)
        if len(ppt) + len(ptp):
            nearest_all.append(np.concatenate([ppt, ptp]))
    cons = np.concatenate(cons_all) if cons_all else np.array([])
    nearest = np.concatenate(nearest_all) if nearest_all else np.array([])
    return cons, nearest


def main():
    sp_df, _, _ = run_sp()
    dp_df = run_dp()

    sp_df.to_csv(DAT / 'entropy_inflection_intervals_sp.csv', index=False)
    dp_df.to_csv(DAT / 'entropy_inflection_intervals_dp.csv', index=False)

    # Pool raw intervals for histograms
    sp_cfg = T.cfg["single_probe"]["coordinates_1"]["mouse01"]["sessions"]
    dp_cfg = T.cfg["double_probe"]["coordinates_1"]["mouse01"]["sessions"]
    sp_cons, sp_near = collect_raw(sp_cfg, T.load_behavior_sp)
    dp_cons, dp_near = collect_raw(dp_cfg, T.load_behavior_dp, skip=T.DP_SKIP)

    print("\nSP per-session:")
    print(sp_df.to_string(index=False))
    print("\nDP per-session:")
    print(dp_df.to_string(index=False))

    print("\n" + "=" * 60)
    print("POOLED INTERVAL STATS (seconds)")
    print("=" * 60)
    for name, cons, near in [('SP', sp_cons, sp_near), ('DP', dp_cons, dp_near)]:
        print(f"\n{name}:")
        print(f"  Consecutive inflections (any-type): n={len(cons)}, "
              f"median={np.median(cons):.1f}, "
              f"IQR=[{np.percentile(cons, 25):.1f}, {np.percentile(cons, 75):.1f}], "
              f"min={np.min(cons):.1f}, max={np.max(cons):.1f}")
        print(f"  Nearest opposite-type: n={len(near)}, "
              f"median={np.median(near):.1f}, "
              f"IQR=[{np.percentile(near, 25):.1f}, {np.percentile(near, 75):.1f}], "
              f"min={np.min(near):.1f}, max={np.max(near):.1f}")
        # Fraction of events with nearest-opposite <= 120 s
        f120 = np.mean(near <= 120) * 100
        f60 = np.mean(near <= 60) * 100
        f5 = np.mean(near <= 5) * 100
        print(f"  % events w/ opposite <= 5 s: {f5:.1f}%, <= 60 s: {f60:.1f}%, <= 120 s: {f120:.1f}%")

    # Histogram plot
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    bins = np.arange(0, 600, 20)
    for col, (name, cons, near) in enumerate([('SP', sp_cons, sp_near),
                                               ('DP', dp_cons, dp_near)]):
        ax = axes[0, col]
        ax.hist(cons, bins=bins, color='#2471A3', edgecolor='white')
        ax.axvline(120, color='red', lw=2, ls='--', label='+/-120 s window')
        ax.axvline(5, color='orange', lw=2, ls='--', label='+/-5 s window')
        ax.set_xlabel('Interval between consecutive inflections (s)')
        ax.set_ylabel('count')
        ax.set_title(f"{name}: consecutive inflection intervals\n"
                     f"n={len(cons)}, median={np.median(cons):.0f} s" if len(cons) else name)
        ax.legend(); ax.grid(alpha=0.2)

        ax = axes[1, col]
        ax.hist(near, bins=bins, color='#C0392B', edgecolor='white')
        ax.axvline(120, color='black', lw=2, ls='--', label='+/-120 s window')
        ax.axvline(5, color='orange', lw=2, ls='--', label='+/-5 s window')
        ax.set_xlabel('Distance to nearest opposite-type inflection (s)')
        ax.set_ylabel('count')
        ax.set_title(f"{name}: nearest opposite-type\n"
                     f"n={len(near)}, median={np.median(near):.0f} s" if len(near) else name)
        ax.legend(); ax.grid(alpha=0.2)

    fig.suptitle('Inter-inflection intervals: is a +/-120 s window clean?',
                 fontsize=14, fontweight='bold')
    fig.savefig(FIG / 'entropy_inflection_intervals.png', dpi=140)
    plt.close(fig)
    print(f"\nSaved {FIG / 'entropy_inflection_intervals.png'}")


if __name__ == '__main__':
    main()
