"""
Tight-Window Peri-Inflection Analysis
======================================
Re-run of peri-inflection analysis with a ±5 s focal window at 500 ms resolution,
z-scored against a –10 s to –5 s baseline.

Rationale: the previous ±120 s window could not distinguish event-locked neural
changes from slow drift. This version asks whether LHA / cortical populations
shift within seconds of an entropy inflection.

Inflection detection is unchanged: causal 60 s / 10 s entropy trace, Gaussian
smoothing sigma=3, minimum amplitude 0.3 bits. Only the peri-event alignment
resolution is tightened.

Outputs:
  data/entropy_inflection_tight_sp.csv               # single-probe pooled stats
  data/dp_entropy_inflection_tight.csv               # dual-probe pooled stats
  data/dp_entropy_inflection_tight_by_state.csv      # dual-probe per-state stats
  figures/entropy_inflection_tight_sp.png
  figures/dp_entropy_inflection_tight.png
  figures/dp_entropy_inflection_tight_by_state.png
"""
from __future__ import annotations

import yaml
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
from scipy.stats import (
    entropy as sp_entropy, mannwhitneyu, wilcoxon,
)
from scipy.ndimage import gaussian_filter1d
from scipy.signal import argrelextrema
from sklearn.decomposition import PCA
import spikeinterface.extractors as se
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# Constants
# =============================================================================
FS = 30_000
ENTROPY_WINDOW_SEC = 60
ENTROPY_STEP_SEC = 10
SMOOTH_SIGMA = 3
MIN_AMPLITUDE = 0.3

# Tight window
BIN_SEC = 0.5                     # 500 ms neural bin
PRE_SEC = 10.0                    # look back 10 s
POST_SEC = 5.0                    # look forward 5 s
BASELINE_START_SEC = -10.0        # -10 s to -5 s = 10 bins
BASELINE_END_SEC = -5.0
FOCAL_START_SEC = -5.0            # -5 s to +5 s = 20 bins (the "tight" display)
FOCAL_END_SEC = 5.0
SMOOTH_BINS = 3                   # Gaussian smooth applied to 500 ms FR trace

# Single-probe unit selection
SP_LHA_MAX_DEPTH = 1300
SP_MIN_FR = 0.3
SP_MIN_AMP = 48

# Dual-probe unit selection (per feedback_dual_probe_thresholds.md)
DP_LHA_DEPTH_MIN = 0
DP_LHA_DEPTH_MAX = 345
DP_P0_MIN_FR = 0.2                # ACA: no AMP filter
DP_P1_MIN_FR = 0.2
DP_P1_MIN_AMP = 43

DP_SKIP = {23, 24}                # new paradigm

# =============================================================================
# Load config
# =============================================================================
with open("paths.yaml") as f:
    cfg = yaml.safe_load(f)

# =============================================================================
# Behavior loaders
# =============================================================================
SP_PRIORITY = [
    'Right corner', 'Left corner', 'Arna center', 'Foraging arena',
    'Home', 'Ladder', 'Transition zone',
    'Pot-1 zone', 'Pot-2 zone', 'Pot-3 zone', 'Pot-4 zone',
    'Pot-1', 'Pot-2', 'Pot-3', 'Pot-4',
]
SP_SHORT = {
    'Home': 'H', 'Ladder': 'L', 'Transition zone': 'T',
    'Foraging arena': 'FA', 'Arna center': 'AC',
    'Pot-1': 'P1', 'Pot-2': 'P2', 'Pot-3': 'P3', 'Pot-4': 'P4',
    'Pot-1 zone': 'P1z', 'Pot-2 zone': 'P2z',
    'Pot-3 zone': 'P3z', 'Pot-4 zone': 'P4z',
    'Right corner': 'RC', 'Left corner': 'LC',
}


def load_behavior_sp(csv_path):
    """Single-probe behavior CSV loader."""
    df_raw = pd.read_csv(csv_path, header=None)
    var_names = df_raw.iloc[:, 0].values
    time_vals = df_raw.iloc[1, 1:].astype(float).values
    data = df_raw.iloc[:, 1:].values
    behav = {}
    for i, name in enumerate(var_names):
        if isinstance(name, str):
            behav[name.strip()] = data[i].astype(float)
    vel = np.nan_to_num(behav.get('Velocity', np.zeros(len(time_vals))), nan=0.0)
    vel = np.clip(vel, 0, None)
    zones = np.full(len(time_vals), 'O', dtype=object)
    for v in SP_PRIORITY:
        if v in behav:
            mask = behav[v] > 0.5
            zones[mask] = SP_SHORT.get(v, v)
    return time_vals, vel, zones


DP_PRIORITY = [
    'Home corner left', 'Home corner right', 'Central Arena Zone',
    'Foraging arena', 'Home', 'ladder to Arena', 'Transition Zone',
    'Pot-1 zone', 'Pot-2 Zone', 'Pot-3 zone', 'Pot-4 zone',
    'Pot-1', 'Pot-2', 'Pot-3', 'Pot-4',
]
DP_SHORT = {
    'Home': 'H', 'ladder to Arena': 'L', 'Transition Zone': 'T',
    'Foraging arena': 'FA', 'Central Arena Zone': 'CA',
    'Pot-1': 'P1', 'Pot-2': 'P2', 'Pot-3': 'P3', 'Pot-4': 'P4',
    'Pot-1 zone': 'P1z', 'Pot-2 Zone': 'P2z',
    'Pot-3 zone': 'P3z', 'Pot-4 zone': 'P4z',
    'Home corner left': 'HCL', 'Home corner right': 'HCR',
}


def load_behavior_dp(xlsx_path):
    df = pd.read_excel(xlsx_path, header=None)
    col_names = df.iloc[34].tolist()
    data = df.iloc[36:].reset_index(drop=True)
    data.columns = col_names
    time_vals = pd.to_numeric(data['Recording time'], errors='coerce').values
    vel = pd.to_numeric(data['Velocity(Center-point)'], errors='coerce').values
    vel = np.nan_to_num(vel, nan=0.0)
    vel = np.clip(vel, 0, None)
    zones = np.full(len(time_vals), 'O', dtype=object)
    for z in DP_PRIORITY:
        match = [c for c in col_names if isinstance(c, str) and c.startswith('Zone(') and z in c]
        if match:
            vals = pd.to_numeric(data[match[0]], errors='coerce').values
            zones[vals > 0.5] = DP_SHORT.get(z, z[:3])
    return time_vals, vel, zones


# =============================================================================
# Entropy (causal, end-assigned)
# =============================================================================
def compute_entropy(zones, time_vals, vel, win_sec, step_sec):
    dt = np.median(np.diff(time_vals))
    win_bins = int(win_sec / dt)
    step_bins = int(step_sec / dt)
    ent_t, ent_v, vel_t = [], [], []
    for s in range(0, len(zones) - win_bins, step_bins):
        wz = zones[s:s + win_bins]
        trans = [f"{wz[j-1]}->{wz[j]}" for j in range(1, len(wz)) if wz[j] != wz[j-1]]
        if len(trans) < 3:
            continue
        probs = np.array(list(Counter(trans).values()), float)
        probs /= probs.sum()
        ent_t.append(time_vals[s + win_bins - 1])
        ent_v.append(sp_entropy(probs, base=2))
        vel_t.append(np.nanmean(vel[s:s + win_bins]))
    return np.array(ent_t), np.array(ent_v), np.array(vel_t)


def find_inflections(vals, sigma, min_amp, order=3):
    sm = gaussian_filter1d(vals, sigma)
    peaks = list(argrelextrema(sm, np.greater, order=order)[0])
    troughs = list(argrelextrema(sm, np.less, order=order)[0])
    merged = sorted(
        [(p, 'peak', sm[p]) for p in peaks] + [(t, 'trough', sm[t]) for t in troughs],
        key=lambda x: x[0]
    )
    if len(merged) < 2:
        return [], []
    filt = [merged[0]]
    for item in merged[1:]:
        if item[1] == filt[-1][1]:
            if (item[1] == 'peak' and item[2] > filt[-1][2]) or \
               (item[1] == 'trough' and item[2] < filt[-1][2]):
                filt[-1] = item
        else:
            if abs(item[2] - filt[-1][2]) >= min_amp:
                filt.append(item)
    return [f[0] for f in filt if f[1] == 'peak'], [f[0] for f in filt if f[1] == 'trough']


# =============================================================================
# Unit selection
# =============================================================================
def sp_units(sorted_path):
    ci = Path(sorted_path) / "cluster_info.tsv"
    if not ci.exists():
        return np.array([]), np.array([])
    df = pd.read_csv(ci, sep='\t')
    label = 'group' if ('group' in df.columns and df['group'].eq('good').any()) else 'KSLabel'
    g = df[(df[label] == 'good') & (df['fr'] > SP_MIN_FR) & (df['amp'] > SP_MIN_AMP)]
    lha = g[g['depth'] < SP_LHA_MAX_DEPTH]['cluster_id'].values
    rsp = g[g['depth'] >= SP_LHA_MAX_DEPTH]['cluster_id'].values
    return lha, rsp


def dp_units_aca(sorted_path):
    ci = Path(sorted_path) / "cluster_info.tsv"
    if not ci.exists():
        return np.array([])
    df = pd.read_csv(ci, sep='\t')
    label = 'group' if ('group' in df.columns and df['group'].eq('good').any()) else 'KSLabel'
    if label not in df.columns:
        return np.array([])
    return df[(df[label] == 'good') & (df['fr'] > DP_P0_MIN_FR)]['cluster_id'].values


def dp_units_lha(sorted_path):
    ci = Path(sorted_path) / "cluster_info.tsv"
    if not ci.exists():
        return np.array([])
    df = pd.read_csv(ci, sep='\t')
    label = 'group' if ('group' in df.columns and df['group'].eq('good').any()) else 'KSLabel'
    if label not in df.columns or 'depth' not in df.columns:
        return np.array([])
    g = df[(df[label] == 'good') &
           (df['fr'] > DP_P1_MIN_FR) &
           (df['amp'] > DP_P1_MIN_AMP) &
           (df['depth'] >= DP_LHA_DEPTH_MIN) &
           (df['depth'] <= DP_LHA_DEPTH_MAX)]
    return g['cluster_id'].values


# =============================================================================
# Tight peri-inflection extraction
# =============================================================================
def compute_fine_neural(sorting, unit_ids, max_time, bin_sec=BIN_SEC):
    """Return (bin_centers, pop_fr_z, pc1_z) at `bin_sec` resolution."""
    if len(unit_ids) < 2:
        return None, None, None
    edges = np.arange(0, max_time + 2 * bin_sec, bin_sec)
    centers = (edges[:-1] + edges[1:]) / 2
    fr = np.array([np.histogram(sorting.get_unit_spike_train(u) / FS, bins=edges)[0]
                   for u in unit_ids], dtype=float)
    fr /= bin_sec  # convert to Hz
    fr_z = np.array([(r - r.mean()) / max(r.std(), 1e-6) for r in fr])
    pop = gaussian_filter1d(np.mean(fr_z, axis=0), SMOOTH_BINS)
    pc1 = gaussian_filter1d(
        PCA(n_components=min(3, len(unit_ids))).fit_transform(fr_z.T)[:, 0],
        SMOOTH_BINS,
    )
    return centers, pop, pc1


def extract_windows(trace, time_grid, event_times):
    """Pull ±PRE/POST_SEC windows around each event_time (seconds).
    Returns (n_events, n_bins) array z-scored to [-10,-5] s baseline.
    n_bins = int((PRE+POST) / BIN_SEC).
    """
    n_pre = int(PRE_SEC / BIN_SEC)
    n_post = int(POST_SEC / BIN_SEC)
    windows = []
    for t_evt in event_times:
        i0 = int(round((t_evt - PRE_SEC) / BIN_SEC))
        i1 = i0 + n_pre + n_post
        if i0 < 0 or i1 > len(trace):
            windows.append(None)
            continue
        w = trace[i0:i1].copy()
        # Baseline: first 5 s of the window = bins [0 : n_pre - n_post_cue]
        n_base = int((BASELINE_END_SEC - BASELINE_START_SEC) / BIN_SEC)  # 10 bins
        base = w[:n_base]
        if np.all(np.isnan(base)) or np.nanstd(base) < 1e-9:
            w_z = w - np.nanmean(base)
        else:
            w_z = (w - np.nanmean(base)) / np.nanstd(base)
        windows.append(w_z)
    return [w for w in windows if w is not None]


# =============================================================================
# Stats helpers
# =============================================================================
def summarize_windows(windows):
    """Given list of (n_bins,) arrays, return n x n_bins stack."""
    if not windows:
        return np.zeros((0, int((PRE_SEC + POST_SEC) / BIN_SEC)))
    return np.vstack(windows)


def at_inflection_values(stack):
    """Pick bin nearest t=0 (which is index n_pre)."""
    n_pre = int(PRE_SEC / BIN_SEC)
    if stack.shape[0] == 0:
        return np.array([])
    return stack[:, n_pre]


def stats_peak_vs_trough(peak_stack, trough_stack):
    """Wilcoxon vs 0 and MWU peak vs trough on 'at' values."""
    pv = at_inflection_values(peak_stack)
    tv = at_inflection_values(trough_stack)
    out = {}
    for name, arr in [('peak', pv), ('trough', tv)]:
        if len(arr) >= 5:
            try:
                out[f'{name}_wilcoxon_p'] = float(wilcoxon(arr).pvalue)
            except ValueError:
                out[f'{name}_wilcoxon_p'] = np.nan
            out[f'{name}_mean_z'] = float(np.mean(arr))
            out[f'{name}_n'] = len(arr)
        else:
            out[f'{name}_wilcoxon_p'] = np.nan
            out[f'{name}_mean_z'] = np.nan
            out[f'{name}_n'] = len(arr)
    if len(pv) >= 5 and len(tv) >= 5:
        try:
            out['mwu_p'] = float(mannwhitneyu(pv, tv).pvalue)
        except ValueError:
            out['mwu_p'] = np.nan
    else:
        out['mwu_p'] = np.nan
    return out


# =============================================================================
# Plotting
# =============================================================================
def plot_pooled_panel(ax, stack_peak, stack_trough, title, ylabel='z-score'):
    """Mean ± SEM traces for peaks and troughs in the focal window only."""
    n_pre = int(PRE_SEC / BIN_SEC)
    n_focal_start = n_pre + int(FOCAL_START_SEC / BIN_SEC)  # = n_pre - 10
    n_focal_end = n_pre + int(FOCAL_END_SEC / BIN_SEC)      # = n_pre + 10
    t_axis = (np.arange(n_focal_start, n_focal_end) - n_pre) * BIN_SEC

    for stack, color, label in [
        (stack_peak, '#C0392B', f'peaks (n={len(stack_peak)})'),
        (stack_trough, '#2471A3', f'troughs (n={len(stack_trough)})'),
    ]:
        if stack.shape[0] == 0:
            continue
        seg = stack[:, n_focal_start:n_focal_end]
        m = np.nanmean(seg, axis=0)
        se = np.nanstd(seg, axis=0) / np.sqrt(max(seg.shape[0], 1))
        ax.plot(t_axis, m, color=color, label=label, linewidth=2.2)
        ax.fill_between(t_axis, m - se, m + se, color=color, alpha=0.22)
    ax.axvline(0, color='k', linestyle=':', alpha=0.5)
    ax.axhline(0, color='gray', linestyle='-', alpha=0.3, linewidth=0.7)
    ax.set_xlabel('time from inflection (s)', fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(alpha=0.2)


# =============================================================================
# RUN — single-probe
# =============================================================================
def run_single_probe():
    print("\n" + "=" * 88)
    print("SINGLE-PROBE (M01 Coor1) — TIGHT PERI-INFLECTION")
    print("=" * 88)
    sp_cfg = cfg["single_probe"]["coordinates_1"]["mouse01"]["sessions"]
    sp_meta = {
        1: ('fed', 'exploration'), 2: ('fed', 'foraging'),
        3: ('fed', 'exploration'), 4: ('fed', 'foraging'),
        5: ('fasted', 'exploration'), 6: ('fasted', 'foraging'),
        7: ('fasted', 'exploration'), 8: ('fasted', 'foraging'),
    }

    metrics = ['Velocity', 'LHA FR', 'RSP FR', 'LHA PC1', 'RSP PC1']
    pool_peaks = {m: [] for m in metrics}
    pool_troughs = {m: [] for m in metrics}
    n_pre = int(PRE_SEC / BIN_SEC)
    n_post = int(POST_SEC / BIN_SEC)
    n_bins = n_pre + n_post

    for snum in range(1, 9):
        state, phase = sp_meta[snum]
        sc = sp_cfg[f"session_{snum}"]
        sp_sorted = sc['sorted']
        bp = sc.get('behavior')
        if not bp or not Path(bp).exists():
            print(f"  S{snum}: no behavior, skipping")
            continue

        # Behavior
        time_vals, vel, zones = load_behavior_sp(bp)
        ent_t, ent_v, _ = compute_entropy(
            zones, time_vals, vel, ENTROPY_WINDOW_SEC, ENTROPY_STEP_SEC)
        if len(ent_v) < 15:
            print(f"  S{snum}: too few entropy pts, skipping")
            continue
        peaks_i, troughs_i = find_inflections(ent_v, SMOOTH_SIGMA, MIN_AMPLITUDE)
        peak_times = ent_t[peaks_i] if peaks_i else np.array([])
        trough_times = ent_t[troughs_i] if troughs_i else np.array([])

        # Neural
        lha_ids, rsp_ids = sp_units(sp_sorted)
        try:
            sorting = se.read_kilosort(sp_sorted)
        except Exception as e:
            print(f"  S{snum}: sorting load err {e}, skip")
            continue
        avail = set(sorting.get_unit_ids())
        lha_ids = np.array([u for u in lha_ids if u in avail])
        rsp_ids = np.array([u for u in rsp_ids if u in avail])
        if len(lha_ids) < 2 or len(rsp_ids) < 2:
            print(f"  S{snum}: low units L={len(lha_ids)} R={len(rsp_ids)}, skip")
            continue

        max_t = float(time_vals[-1])
        lha_centers, lha_pop, lha_pc1 = compute_fine_neural(sorting, lha_ids, max_t)
        _, rsp_pop, rsp_pc1 = compute_fine_neural(sorting, rsp_ids, max_t)
        centers = lha_centers

        # Velocity at same 500 ms grid
        dt_beh = np.median(np.diff(time_vals))
        vel_interp = np.interp(centers, time_vals, vel)

        tracer = {
            'Velocity': vel_interp,
            'LHA FR': lha_pop,
            'RSP FR': rsp_pop,
            'LHA PC1': lha_pc1,
            'RSP PC1': rsp_pc1,
        }

        print(f"  S{snum} {state}/{phase}: {len(peak_times)} peaks, "
              f"{len(trough_times)} troughs, LHA={len(lha_ids)} RSP={len(rsp_ids)}")

        for mname, trace in tracer.items():
            pwins = extract_windows(trace, centers, peak_times)
            twins = extract_windows(trace, centers, trough_times)
            pool_peaks[mname].extend(pwins)
            pool_troughs[mname].extend(twins)

    # Summaries
    stats_rows = []
    stacks = {}
    for m in metrics:
        ps = summarize_windows(pool_peaks[m])
        ts = summarize_windows(pool_troughs[m])
        stacks[m] = (ps, ts)
        res = stats_peak_vs_trough(ps, ts)
        stats_rows.append({'dataset': 'single_probe', 'metric': m, **res})

    df = pd.DataFrame(stats_rows)
    df.to_csv('data/entropy_inflection_tight_sp.csv', index=False)
    print(f"\nSaved data/entropy_inflection_tight_sp.csv")
    print(df.to_string(index=False))

    # Plot
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.2), constrained_layout=True)
    for ax, m in zip(axes, metrics):
        ps, ts = stacks[m]
        plot_pooled_panel(ax, ps, ts, m)
    fig.suptitle('Single-probe pooled peri-inflection — tight ±5 s window '
                 '(baseline −10 s to −5 s)', fontsize=15, fontweight='bold')
    fig.savefig('figures/entropy_inflection_tight_sp.png', dpi=140)
    plt.close(fig)
    print("Saved figures/entropy_inflection_tight_sp.png")
    return stacks, stats_rows


# =============================================================================
# RUN — dual-probe (pooled + by state)
# =============================================================================
def run_dual_probe():
    print("\n" + "=" * 88)
    print("DUAL-PROBE — TIGHT PERI-INFLECTION")
    print("=" * 88)
    dp_cfg = cfg["double_probe"]["coordinates_1"]["mouse01"]["sessions"]

    metrics = ['Velocity', 'ACA FR', 'LHA FR', 'ACA PC1', 'LHA PC1']
    pool_peaks = {m: [] for m in metrics}
    pool_troughs = {m: [] for m in metrics}
    state_peaks = {}   # state -> metric -> list windows
    state_troughs = {}

    for skey, sval in dp_cfg.items():
        snum = int(skey.split('_')[1])
        if snum in DP_SKIP:
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
            time_vals, vel, zones = load_behavior_dp(beh)
        except Exception as e:
            print(f"  S{snum}: behavior load err {e}, skip")
            continue

        ent_t, ent_v, _ = compute_entropy(
            zones, time_vals, vel, ENTROPY_WINDOW_SEC, ENTROPY_STEP_SEC)
        if len(ent_v) < 15:
            print(f"  S{snum}: few ent pts, skip")
            continue
        peaks_i, troughs_i = find_inflections(ent_v, SMOOTH_SIGMA, MIN_AMPLITUDE)
        peak_times = ent_t[peaks_i] if peaks_i else np.array([])
        trough_times = ent_t[troughs_i] if troughs_i else np.array([])

        aca_ids = dp_units_aca(p0)
        lha_ids = dp_units_lha(p1)
        try:
            sorting_p0 = se.read_kilosort(p0)
            aca_ids = np.array([u for u in aca_ids if u in set(sorting_p0.get_unit_ids())])
        except Exception as e:
            print(f"  S{snum}: P0 err {e}")
            aca_ids = np.array([])
            sorting_p0 = None
        try:
            sorting_p1 = se.read_kilosort(p1)
            lha_ids = np.array([u for u in lha_ids if u in set(sorting_p1.get_unit_ids())])
        except Exception as e:
            print(f"  S{snum}: P1 err {e}")
            lha_ids = np.array([])
            sorting_p1 = None

        if (sorting_p0 is None or len(aca_ids) < 2) and \
           (sorting_p1 is None or len(lha_ids) < 2):
            print(f"  S{snum}: insufficient units, skip")
            continue

        max_t = float(time_vals[-1])
        if sorting_p0 is not None and len(aca_ids) >= 2:
            centers_p0, aca_pop, aca_pc1 = compute_fine_neural(sorting_p0, aca_ids, max_t)
        else:
            centers_p0 = None
            aca_pop = aca_pc1 = None
        if sorting_p1 is not None and len(lha_ids) >= 2:
            centers_p1, lha_pop, lha_pc1 = compute_fine_neural(sorting_p1, lha_ids, max_t)
        else:
            centers_p1 = None
            lha_pop = lha_pc1 = None

        centers = centers_p0 if centers_p0 is not None else centers_p1
        vel_interp = np.interp(centers, time_vals, vel)

        tracer = {'Velocity': vel_interp}
        if aca_pop is not None:
            tracer['ACA FR'] = aca_pop
            tracer['ACA PC1'] = aca_pc1
        if lha_pop is not None:
            # align to same grid if different (usually same duration)
            if centers_p1 is not None and not np.array_equal(centers, centers_p1):
                lha_pop = np.interp(centers, centers_p1, lha_pop)
                lha_pc1 = np.interp(centers, centers_p1, lha_pc1)
            tracer['LHA FR'] = lha_pop
            tracer['LHA PC1'] = lha_pc1

        print(f"  S{snum} {state}/{phase}: {len(peak_times)} peaks, "
              f"{len(trough_times)} troughs, ACA={len(aca_ids)} LHA={len(lha_ids)}")

        state_peaks.setdefault(state, {m: [] for m in metrics})
        state_troughs.setdefault(state, {m: [] for m in metrics})

        for mname, trace in tracer.items():
            if trace is None:
                continue
            pwins = extract_windows(trace, centers, peak_times)
            twins = extract_windows(trace, centers, trough_times)
            pool_peaks[mname].extend(pwins)
            pool_troughs[mname].extend(twins)
            state_peaks[state][mname].extend(pwins)
            state_troughs[state][mname].extend(twins)

    # Pooled stats
    pooled_rows = []
    pooled_stacks = {}
    for m in metrics:
        ps = summarize_windows(pool_peaks[m])
        ts = summarize_windows(pool_troughs[m])
        pooled_stacks[m] = (ps, ts)
        res = stats_peak_vs_trough(ps, ts)
        pooled_rows.append({'dataset': 'dual_probe', 'metric': m, **res})
    df_pool = pd.DataFrame(pooled_rows)
    df_pool.to_csv('data/dp_entropy_inflection_tight.csv', index=False)
    print("\nPOOLED:")
    print(df_pool.to_string(index=False))

    # By-state stats
    state_rows = []
    state_stacks = {}
    for state in state_peaks:
        state_stacks[state] = {}
        for m in metrics:
            ps = summarize_windows(state_peaks[state][m])
            ts = summarize_windows(state_troughs[state][m])
            state_stacks[state][m] = (ps, ts)
            res = stats_peak_vs_trough(ps, ts)
            state_rows.append({'state': state, 'metric': m, **res})
    df_state = pd.DataFrame(state_rows)
    df_state.to_csv('data/dp_entropy_inflection_tight_by_state.csv', index=False)
    print("\nBY STATE:")
    print(df_state.to_string(index=False))

    # Plots — pooled
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.2), constrained_layout=True)
    for ax, m in zip(axes, metrics):
        ps, ts = pooled_stacks[m]
        plot_pooled_panel(ax, ps, ts, m)
    fig.suptitle('Dual-probe pooled peri-inflection — tight ±5 s window '
                 '(baseline −10 s to −5 s)', fontsize=15, fontweight='bold')
    fig.savefig('figures/dp_entropy_inflection_tight.png', dpi=140)
    plt.close(fig)

    # Plots — by state
    states = sorted(state_stacks.keys())
    fig, axes = plt.subplots(len(states), 5,
                             figsize=(22, 3.8 * len(states)), constrained_layout=True)
    if len(states) == 1:
        axes = axes[None, :]
    for row, state in enumerate(states):
        for col, m in enumerate(metrics):
            ps, ts = state_stacks[state][m]
            plot_pooled_panel(axes[row, col], ps, ts, f'{state} — {m}')
    fig.suptitle('Dual-probe peri-inflection by state — tight ±5 s window',
                 fontsize=15, fontweight='bold')
    fig.savefig('figures/dp_entropy_inflection_tight_by_state.png', dpi=140)
    plt.close(fig)
    print("Saved figures/dp_entropy_inflection_tight.png and _by_state.png")
    return pooled_stacks, state_stacks, pooled_rows, state_rows


if __name__ == '__main__':
    sp_stacks, sp_rows = run_single_probe()
    dp_pooled, dp_state, dp_pool_rows, dp_state_rows = run_dual_probe()
    print("\n\n=== DONE ===")
