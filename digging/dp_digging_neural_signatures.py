"""
Dual-Probe: Neural Signatures Around Digging Onset
====================================================
Aligns ACA and LHA neural data to manually-scored digging bouts.
Uses 1s bins (forgiving for manual scoring imprecision).

Metrics computed in peri-dig windows (-30s to +15s around onset):
  Population-level:  ACA FR, LHA FR, ACA PC1, LHA PC1
  Variability:       ACA Fano, LHA Fano, ACA FR var, LHA FR var
  Behavioral:        Velocity

Statistical comparisons:
  Pre-dig window:  [-3s to 0s]  (tight pre-onset)
  Post-dig window: [+1s to +4s] (avoiding ±1s onset uncertainty)
  Baseline:        [-30s to -15s] (for z-scoring)
  Rate of change:  linear slope in [-3s to 0s]
  Duration split:  short vs long bouts (median split)

Dig bouts: extracted from "Digging sand" manual label, min duration 2s.
Onset is treated with ±2s tolerance (manual scoring imprecision).
"""

import yaml
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.stats import mannwhitneyu, wilcoxon, spearmanr, linregress
from sklearn.decomposition import PCA
import spikeinterface.extractors as se
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
import time as timer

warnings.filterwarnings('ignore')

with open("paths.yaml") as f:
    cfg = yaml.safe_load(f)

# ---- Constants ----
FS = 30000
LHA_DEPTH_MIN = 0
LHA_DEPTH_MAX = 345
P0_MIN_FR = 0.2
P1_MIN_FR = 0.2
P1_MIN_AMP = 43

BIN_SEC = 1.0          # 1s bins — forgiving for manual scoring
SMOOTH_SIGMA = 3       # gaussian smooth (3s)
PRE_SEC = 30           # seconds before dig onset
POST_SEC = 15          # seconds after dig onset
MIN_DIG_DURATION = 2.0 # minimum dig bout length (seconds)
MIN_INTER_DIG = 10.0   # minimum gap between dig bouts to count as separate events

SKIP_SESSIONS = {23, 24}
FIG_XLIM = (-5, 10)  # visible range on all figures

sessions_cfg = cfg["double_probe"]["coordinates_1"]["mouse01"]["sessions"]

# Zone mapping
zone_priority = [
    'Home corner left', 'Home corner right', 'Central Arena Zone',
    'Foraging arena', 'Home', 'ladder to Arena', 'Transition Zone',
    'Pot-1 zone', 'Pot-2 Zone', 'Pot-3 zone', 'Pot-4 zone',
    'Pot-1', 'Pot-2', 'Pot-3', 'Pot-4',
]
zone_short = {
    'Home': 'H', 'ladder to Arena': 'L', 'Transition Zone': 'T',
    'Foraging arena': 'FA', 'Central Arena Zone': 'CA',
    'Pot-1': 'P1', 'Pot-2': 'P2', 'Pot-3': 'P3', 'Pot-4': 'P4',
    'Pot-1 zone': 'P1z', 'Pot-2 Zone': 'P2z', 'Pot-3 zone': 'P3z', 'Pot-4 zone': 'P4z',
    'Home corner left': 'HCL', 'Home corner right': 'HCR',
}


def load_behavior_xlsx(path):
    """Load dual-probe behavior xlsx."""
    df = pd.read_excel(path, header=None)
    col_names = df.iloc[34].tolist()
    data = df.iloc[36:].reset_index(drop=True)
    data.columns = col_names

    time_vals = pd.to_numeric(data['Recording time'], errors='coerce').values
    vel = pd.to_numeric(data['Velocity(Center-point)'], errors='coerce').values
    vel = np.nan_to_num(vel, nan=0.0)
    vel = np.clip(vel, 0, None)

    # Build zone array
    zones = np.full(len(time_vals), 'O', dtype=object)
    for zname in zone_priority:
        col_match = [c for c in col_names if isinstance(c, str) and
                     c.startswith('Zone(') and zname in c]
        if col_match:
            vals = pd.to_numeric(data[col_match[0]], errors='coerce').values
            mask = vals > 0.5
            short = zone_short.get(zname, zname[:3])
            zones[mask] = short

    # Get digging label
    dig_col = 'Digging sand'
    if dig_col in col_names:
        dig_vals = pd.to_numeric(data[dig_col], errors='coerce').values
        dig_vals = np.nan_to_num(dig_vals, nan=0.0)
    else:
        dig_vals = np.zeros(len(time_vals))

    return time_vals, vel, zones, dig_vals


def extract_dig_bouts(dig_vals, time_vals, min_duration, min_inter_dig):
    """Extract dig bout onsets from binary label.
    Merges bouts separated by < min_inter_dig seconds.
    Returns list of dicts with start_time, end_time, duration."""
    mask = dig_vals > 0.5
    diff = np.diff(mask.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1
    if mask[0]:
        starts = np.concatenate([[0], starts])
    if mask[-1]:
        ends = np.concatenate([ends, [len(mask)]])

    if len(starts) == 0:
        return []

    # Convert to times
    bout_times = [(time_vals[s], time_vals[min(e - 1, len(time_vals) - 1)])
                  for s, e in zip(starts, ends)]

    # Merge bouts separated by < min_inter_dig
    merged = [bout_times[0]]
    for s, e in bout_times[1:]:
        if s - merged[-1][1] < min_inter_dig:
            merged[-1] = (merged[-1][0], e)  # extend previous bout
        else:
            merged.append((s, e))

    # Filter by duration
    bouts = []
    for s, e in merged:
        dur = e - s
        if dur >= min_duration:
            bouts.append({'start_time': s, 'end_time': e, 'duration': dur})

    return bouts


def get_pot_at_dig(zones, time_vals, dig_start):
    """Determine which pot the mouse is at when digging starts."""
    idx = np.searchsorted(time_vals, dig_start)
    # Look in a ±2s window around onset for pot zones
    dt = np.median(np.diff(time_vals))
    window = int(2.0 / dt)
    start = max(0, idx - window)
    end = min(len(zones), idx + window)
    segment = zones[start:end]
    pot_zones = [z for z in segment if z.startswith('P') and not z.endswith('z')]
    if pot_zones:
        from collections import Counter
        return Counter(pot_zones).most_common(1)[0][0]
    return 'unknown'


def get_good_units_p0(sorted_path):
    """Probe 0 (ACA): KSLabel='good' + FR > 0.2, no AMP filter."""
    ci = Path(sorted_path) / "cluster_info.tsv"
    if not ci.exists():
        return np.array([])
    df = pd.read_csv(ci, sep='\t')
    label_col = 'group' if ('group' in df.columns and df['group'].eq('good').any()) else 'KSLabel'
    if label_col not in df.columns:
        return np.array([])
    good = df[(df[label_col] == 'good') & (df['fr'] > P0_MIN_FR)]
    return good['cluster_id'].values


def get_good_units_p1_lha(sorted_path):
    """Probe 1 (LHA): KSLabel='good' + FR > 0.2 + AMP > 43, depth 0-345um."""
    ci = Path(sorted_path) / "cluster_info.tsv"
    if not ci.exists():
        return np.array([])
    df = pd.read_csv(ci, sep='\t')
    label_col = 'group' if ('group' in df.columns and df['group'].eq('good').any()) else 'KSLabel'
    if label_col not in df.columns or 'depth' not in df.columns:
        return np.array([])
    good = df[(df[label_col] == 'good') &
              (df['fr'] > P1_MIN_FR) &
              (df['amp'] > P1_MIN_AMP) &
              (df['depth'] >= LHA_DEPTH_MIN) &
              (df['depth'] <= LHA_DEPTH_MAX)]
    return good['cluster_id'].values


def compute_fano_factor(spike_counts_per_unit, smooth_sigma=3):
    """Fano factor: var/mean across units at each time bin, smoothed."""
    mean_fr = np.mean(spike_counts_per_unit, axis=0)
    var_fr = np.var(spike_counts_per_unit, axis=0)
    # Avoid division by zero
    mean_fr_safe = np.where(mean_fr > 0.01, mean_fr, 0.01)
    fano = var_fr / mean_fr_safe
    return gaussian_filter1d(fano, smooth_sigma)


def compute_pop_variance(z_scored_fr, smooth_sigma=3):
    """Variance of z-scored FR across units at each time bin."""
    v = np.var(z_scored_fr, axis=0)
    return gaussian_filter1d(v, smooth_sigma)


# ========================================================================
# Discover sessions with sorted data + behavior + digging
# ========================================================================
session_meta = {}
for skey, sval in sessions_cfg.items():
    snum = int(skey.split('_')[1])
    if snum in SKIP_SESSIONS:
        continue
    p0_sorted = sval.get('probe_0_aca', {}).get('sorted')
    p1_sorted = sval.get('probe_1_lha_rsp', {}).get('sorted')
    behav = sval.get('behavior')
    if not p0_sorted or not p1_sorted or not behav:
        continue
    if not Path(p0_sorted).exists() or not Path(behav).exists():
        continue
    session_meta[snum] = {
        'state': sval['state'], 'phase': sval['phase'],
        'p0_sorted': p0_sorted, 'p1_sorted': p1_sorted,
        'behavior': behav,
    }

print(f"Found {len(session_meta)} sessions with sorted data + behavior")

# ========================================================================
# MAIN ANALYSIS
# ========================================================================
print("\n" + "=" * 100)
print("DUAL-PROBE: NEURAL SIGNATURES AROUND DIGGING ONSET")
print("ACA (probe 0) and LHA (probe 1, 0-345um)")
print(f"Window: -{PRE_SEC}s to +{POST_SEC}s, {BIN_SEC}s bins, min dig duration {MIN_DIG_DURATION}s")
print("=" * 100)

# Time axis for peri-dig windows (in seconds relative to onset)
n_pre_bins = int(PRE_SEC / BIN_SEC)
n_post_bins = int(POST_SEC / BIN_SEC)
n_total_bins = n_pre_bins + n_post_bins
time_axis = np.arange(-n_pre_bins, n_post_bins) * BIN_SEC + BIN_SEC / 2  # bin centers

metric_names = ['Velocity', 'ACA FR', 'LHA FR', 'ACA PC1', 'LHA PC1',
                'ACA Fano', 'LHA Fano', 'ACA FR var', 'LHA FR var']

# Pooled storage
pooled_windows = {m: [] for m in metric_names}
pooled_meta = []  # session, state, phase, pot, duration per event

all_event_data = []  # for CSV

for snum in sorted(session_meta.keys()):
    t0 = timer.time()
    meta = session_meta[snum]
    state, phase = meta['state'], meta['phase']

    # ---- Load behavior ----
    print(f"\n  S{snum} ({state}/{phase}): loading behavior...", end='', flush=True)
    time_vals, vel, zones, dig_vals = load_behavior_xlsx(meta['behavior'])

    # Check if digging is scored
    n_dig_bins = np.sum(dig_vals > 0.5)
    if n_dig_bins == 0:
        print(f" no digging scored, skipping")
        continue

    # Extract dig bouts
    bouts = extract_dig_bouts(dig_vals, time_vals, MIN_DIG_DURATION, MIN_INTER_DIG)
    if len(bouts) == 0:
        print(f" no dig bouts >= {MIN_DIG_DURATION}s, skipping")
        continue

    print(f" {len(bouts)} dig bouts", end='')

    # ---- Load neural data ----
    p0_path = Path(meta['p0_sorted'])
    p1_path = Path(meta['p1_sorted'])

    aca_ids = get_good_units_p0(p0_path)
    lha_ids = get_good_units_p1_lha(p1_path)

    try:
        sorting_p0 = se.read_kilosort(p0_path)
        avail_p0 = set(sorting_p0.get_unit_ids())
        aca_ids = np.array([u for u in aca_ids if u in avail_p0])
    except Exception:
        aca_ids = np.array([])

    try:
        sorting_p1 = se.read_kilosort(p1_path)
        avail_p1 = set(sorting_p1.get_unit_ids())
        lha_ids = np.array([u for u in lha_ids if u in avail_p1])
    except Exception:
        lha_ids = np.array([])

    print(f", ACA={len(aca_ids)}, LHA={len(lha_ids)}", end='')

    if len(aca_ids) < 2 and len(lha_ids) < 2:
        print(" — skipping (too few units)")
        continue

    # ---- Bin spikes at 1s resolution ----
    session_end = time_vals[-1] + 1
    bin_edges = np.arange(0, session_end + BIN_SEC, BIN_SEC)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Velocity at 1s bins
    vel_binned = np.interp(bin_centers, time_vals, vel)
    vel_smooth = gaussian_filter1d(vel_binned, SMOOTH_SIGMA)

    # ACA neural metrics
    has_aca = len(aca_ids) >= 2
    if has_aca:
        aca_counts = np.array([np.histogram(sorting_p0.get_unit_spike_train(u) / FS,
                                             bins=bin_edges)[0] for u in aca_ids])
        aca_z = np.array([(r - r.mean()) / max(r.std(), 1e-6) for r in aca_counts])
        aca_pop_fr = gaussian_filter1d(np.mean(aca_z, axis=0), SMOOTH_SIGMA)
        aca_pc1 = gaussian_filter1d(
            PCA(n_components=min(3, len(aca_ids))).fit_transform(aca_z.T)[:, 0], SMOOTH_SIGMA)
        aca_fano = compute_fano_factor(aca_counts, SMOOTH_SIGMA)
        aca_fr_var = compute_pop_variance(aca_z, SMOOTH_SIGMA)

    # LHA neural metrics
    has_lha = len(lha_ids) >= 2
    if has_lha:
        lha_counts = np.array([np.histogram(sorting_p1.get_unit_spike_train(u) / FS,
                                             bins=bin_edges)[0] for u in lha_ids])
        lha_z = np.array([(r - r.mean()) / max(r.std(), 1e-6) for r in lha_counts])
        lha_pop_fr = gaussian_filter1d(np.mean(lha_z, axis=0), SMOOTH_SIGMA)
        lha_pc1 = gaussian_filter1d(
            PCA(n_components=min(3, len(lha_ids))).fit_transform(lha_z.T)[:, 0], SMOOTH_SIGMA)
        lha_fano = compute_fano_factor(lha_counts, SMOOTH_SIGMA)
        lha_fr_var = compute_pop_variance(lha_z, SMOOTH_SIGMA)

    # ---- Extract peri-dig windows ----
    n_valid = 0
    for bout in bouts:
        onset_time = bout['start_time']
        onset_bin = int(onset_time / BIN_SEC)

        # Check window bounds
        start_bin = onset_bin - n_pre_bins
        end_bin = onset_bin + n_post_bins
        if start_bin < 0 or end_bin > len(bin_centers):
            continue

        # Which pot
        pot = get_pot_at_dig(zones, time_vals, onset_time)

        # Extract windows
        w_vel = vel_smooth[start_bin:end_bin]
        pooled_windows['Velocity'].append(w_vel)

        if has_aca:
            pooled_windows['ACA FR'].append(aca_pop_fr[start_bin:end_bin])
            pooled_windows['ACA PC1'].append(aca_pc1[start_bin:end_bin])
            pooled_windows['ACA Fano'].append(aca_fano[start_bin:end_bin])
            pooled_windows['ACA FR var'].append(aca_fr_var[start_bin:end_bin])

        if has_lha:
            pooled_windows['LHA FR'].append(lha_pop_fr[start_bin:end_bin])
            pooled_windows['LHA PC1'].append(lha_pc1[start_bin:end_bin])
            pooled_windows['LHA Fano'].append(lha_fano[start_bin:end_bin])
            pooled_windows['LHA FR var'].append(lha_fr_var[start_bin:end_bin])

        pooled_meta.append({
            'session': snum, 'state': state, 'phase': phase,
            'pot': pot, 'duration': bout['duration'],
            'onset_time': onset_time,
            'has_aca': has_aca, 'has_lha': has_lha,
        })
        n_valid += 1

    elapsed = timer.time() - t0
    print(f", {n_valid} valid windows [{elapsed:.1f}s]")

    # Per-session summary
    for bout in bouts:
        pot = get_pot_at_dig(zones, time_vals, bout['start_time'])
        all_event_data.append({
            'session': snum, 'state': state, 'phase': phase,
            'onset_time': bout['start_time'],
            'duration': bout['duration'],
            'pot': pot,
        })

# ========================================================================
# SUMMARY
# ========================================================================
print("\n" + "=" * 100)
print("SUMMARY")
print("=" * 100)

total_events = len(pooled_meta)
print(f"\nTotal dig events: {total_events}")

# By state
for state in ['fed', 'fasted', 'fed-HFD']:
    events = [m for m in pooled_meta if m['state'] == state]
    if events:
        pots = [e['pot'] for e in events]
        dur = [e['duration'] for e in events]
        from collections import Counter
        pot_counts = Counter(pots)
        print(f"  {state}: {len(events)} events, "
              f"median dur={np.median(dur):.1f}s, "
              f"pots: {dict(pot_counts)}")

# ========================================================================
# STATISTICAL TESTS: Pre vs Post (tight windows)
# ========================================================================
# Pre-dig: [-3s to 0s] → bin centers at -2.5, -1.5, -0.5
# Post-dig: [+1s to +4s] → bin centers at +1.5, +2.5, +3.5 (skip ±1s onset uncertainty)
# Baseline: [-30s to -15s] → for z-scoring
pre_slice = slice(n_pre_bins - 3, n_pre_bins)   # bins 27-29: -3s to 0s
post_slice = slice(n_pre_bins + 1, n_pre_bins + 4)  # bins 31-33: +1s to +4s
baseline_slice = slice(0, n_pre_bins - 15)       # bins 0-14: -30s to -15s

# Time indices for rate-of-change slope in [-3s, 0s]
roc_bin_indices = list(range(n_pre_bins - 3, n_pre_bins))  # bins 27,28,29
roc_times = np.array([time_axis[i] for i in roc_bin_indices])  # -2.5, -1.5, -0.5

# Duration split: short vs long by median
all_durations = np.array([m['duration'] for m in pooled_meta])
if len(all_durations) > 0:
    dur_median = np.median(all_durations)
    short_mask = np.array([m['duration'] <= dur_median for m in pooled_meta])
    long_mask = np.array([m['duration'] > dur_median for m in pooled_meta])
else:
    dur_median = 0
    short_mask = np.array([], dtype=bool)
    long_mask = np.array([], dtype=bool)

print("\n" + "=" * 100)
print("PRE vs POST DIG ONSET (tight windows)")
print(f"Pre: [-3s to 0s], Post: [+1s to +4s], Baseline: [-30s to -15s]")
print(f"Duration split: median = {dur_median:.1f}s "
      f"(short <= {dur_median:.1f}s: {np.sum(short_mask)}, "
      f"long > {dur_median:.1f}s: {np.sum(long_mask)})")
print("=" * 100)


def zscore_to_baseline(w, bl_slice):
    """Z-score each event to its own baseline."""
    w_z = w.copy()
    for i in range(len(w_z)):
        bl = w_z[i, bl_slice]
        bl_mean, bl_std = np.nanmean(bl), np.nanstd(bl)
        if bl_std > 1e-6:
            w_z[i] = (w_z[i] - bl_mean) / bl_std
        else:
            w_z[i] = w_z[i] - bl_mean
    return w_z


def compute_roc_slopes(w, roc_indices, roc_t):
    """Compute rate-of-change (linear slope) in [-3s, 0s] for each event."""
    slopes = np.full(len(w), np.nan)
    for i in range(len(w)):
        vals = w[i, roc_indices]
        if np.any(np.isnan(vals)):
            continue
        res = linregress(roc_t, vals)
        slopes[i] = res.slope
    return slopes


def run_pre_post_stats(w, pre_sl, post_sl, roc_idx, roc_t, label='', n_min=3):
    """Run pre-vs-post and rate-of-change stats. Returns dict of results."""
    if len(w) < n_min:
        return None
    pre_vals = np.nanmean(w[:, pre_sl], axis=1)
    post_vals = np.nanmean(w[:, post_sl], axis=1)
    slopes = compute_roc_slopes(w, roc_idx, roc_t)
    valid_slopes = slopes[~np.isnan(slopes)]

    try:
        _, mwu_p = mannwhitneyu(pre_vals, post_vals, alternative='two-sided')
    except Exception:
        mwu_p = 1.0
    try:
        _, wil_p = wilcoxon(pre_vals - post_vals)
    except Exception:
        wil_p = 1.0
    # Test if slope != 0
    try:
        _, slope_p = wilcoxon(valid_slopes)
    except Exception:
        slope_p = 1.0

    delta = np.mean(post_vals) - np.mean(pre_vals)
    mean_slope = np.nanmean(slopes)

    sig_wil = '*' if wil_p < 0.05 else ''
    sig_slope = '*' if slope_p < 0.05 else ''
    if label:
        print(f"    {label:15s}: n={len(w)}, pre={np.mean(pre_vals):+.3f}, "
              f"post={np.mean(post_vals):+.3f}, delta={delta:+.3f}, "
              f"Wilcox p={wil_p:.4f}{sig_wil}, "
              f"slope={mean_slope:+.3f}/s, slope_p={slope_p:.4f}{sig_slope}")

    return {
        'n_events': len(w),
        'pre_mean': np.mean(pre_vals), 'post_mean': np.mean(post_vals),
        'delta': delta, 'mwu_p': mwu_p, 'wilcoxon_p': wil_p,
        'mean_slope': mean_slope, 'slope_p': slope_p,
    }


results_rows = []

# ---- Pooled (all events) ----
print("\n  --- ALL EVENTS ---")
for mname in metric_names:
    windows = pooled_windows[mname]
    if len(windows) < 5:
        continue
    w = zscore_to_baseline(np.array(windows), baseline_slice)
    res = run_pre_post_stats(w, pre_slice, post_slice, roc_bin_indices, roc_times, mname)
    if res:
        res.update({'metric': mname, 'group': 'all'})
        results_rows.append(res)

# ---- Per state ----
for state_filter in ['fed', 'fasted', 'fed-HFD']:
    state_mask_arr = np.array([m['state'] == state_filter for m in pooled_meta])
    n_state = np.sum(state_mask_arr)
    if n_state < 3:
        continue

    print(f"\n  --- {state_filter.upper()} ({n_state} events) ---")
    for mname in metric_names:
        windows = pooled_windows[mname]
        if len(windows) < 5:
            continue

        if 'ACA' in mname:
            m_mask = np.array([m['state'] == state_filter and m['has_aca']
                               for m in pooled_meta])
        elif 'LHA' in mname:
            m_mask = np.array([m['state'] == state_filter and m['has_lha']
                               for m in pooled_meta])
        else:
            m_mask = state_mask_arr

        if len(windows) != len(m_mask):
            continue

        w = np.array(windows)
        w_f = w[m_mask[:len(w)]]
        if len(w_f) < 3:
            continue

        w_f = zscore_to_baseline(w_f, baseline_slice)
        res = run_pre_post_stats(w_f, pre_slice, post_slice, roc_bin_indices, roc_times, mname)
        if res:
            res.update({'metric': mname, 'group': state_filter})
            results_rows.append(res)

# ---- Duration split (all states pooled) ----
print(f"\n  --- SHORT BOUTS (<= {dur_median:.1f}s, n={np.sum(short_mask)}) ---")
for mname in metric_names:
    windows = pooled_windows[mname]
    if len(windows) < 5:
        continue
    w = np.array(windows)
    w_s = w[short_mask[:len(w)]]
    if len(w_s) < 3:
        continue
    w_s = zscore_to_baseline(w_s, baseline_slice)
    res = run_pre_post_stats(w_s, pre_slice, post_slice, roc_bin_indices, roc_times, mname)
    if res:
        res.update({'metric': mname, 'group': f'short_<={dur_median:.0f}s'})
        results_rows.append(res)

print(f"\n  --- LONG BOUTS (> {dur_median:.1f}s, n={np.sum(long_mask)}) ---")
for mname in metric_names:
    windows = pooled_windows[mname]
    if len(windows) < 5:
        continue
    w = np.array(windows)
    w_l = w[long_mask[:len(w)]]
    if len(w_l) < 3:
        continue
    w_l = zscore_to_baseline(w_l, baseline_slice)
    res = run_pre_post_stats(w_l, pre_slice, post_slice, roc_bin_indices, roc_times, mname)
    if res:
        res.update({'metric': mname, 'group': f'long_>{dur_median:.0f}s'})
        results_rows.append(res)

# ---- Short vs Long comparison (MWU on slopes and pre-dig values) ----
print(f"\n  --- SHORT vs LONG COMPARISON ---")
for mname in metric_names:
    windows = pooled_windows[mname]
    if len(windows) < 5:
        continue
    w = zscore_to_baseline(np.array(windows), baseline_slice)
    w_s = w[short_mask[:len(w)]]
    w_l = w[long_mask[:len(w)]]
    if len(w_s) < 3 or len(w_l) < 3:
        continue

    # Compare pre-dig values
    pre_s = np.nanmean(w_s[:, pre_slice], axis=1)
    pre_l = np.nanmean(w_l[:, pre_slice], axis=1)
    try:
        _, pre_p = mannwhitneyu(pre_s, pre_l, alternative='two-sided')
    except Exception:
        pre_p = 1.0

    # Compare slopes
    slopes_s = compute_roc_slopes(w_s, roc_bin_indices, roc_times)
    slopes_l = compute_roc_slopes(w_l, roc_bin_indices, roc_times)
    valid_s = slopes_s[~np.isnan(slopes_s)]
    valid_l = slopes_l[~np.isnan(slopes_l)]
    try:
        _, slope_p = mannwhitneyu(valid_s, valid_l, alternative='two-sided')
    except Exception:
        slope_p = 1.0

    sig_pre = '*' if pre_p < 0.05 else ''
    sig_slope = '*' if slope_p < 0.05 else ''
    print(f"    {mname:15s}: pre short={np.mean(pre_s):+.3f} vs long={np.mean(pre_l):+.3f} "
          f"p={pre_p:.4f}{sig_pre}, "
          f"slope short={np.nanmean(slopes_s):+.3f} vs long={np.nanmean(slopes_l):+.3f} "
          f"p={slope_p:.4f}{sig_slope}")

    results_rows.append({
        'metric': mname, 'group': 'short_vs_long',
        'n_events': len(w_s) + len(w_l),
        'pre_mean': np.mean(pre_s) - np.mean(pre_l),  # difference
        'mean_slope': np.nanmean(slopes_s) - np.nanmean(slopes_l),  # difference
        'mwu_p': pre_p, 'slope_p': slope_p,
    })

# ========================================================================
# Helper: shade analysis windows on axis
# ========================================================================
def shade_windows(ax):
    """Add onset line and analysis window shading."""
    ax.axvline(x=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Dig onset')
    ax.axvspan(-3, 0, color='#2196F3', alpha=0.08, label='Pre [-3s,0s]')
    ax.axvspan(1, 4, color='#4CAF50', alpha=0.08, label='Post [+1s,+4s]')
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)


# ========================================================================
# FIGURE 1: Pooled peri-dig traces (all events)
# ========================================================================
fig, axes = plt.subplots(3, 3, figsize=(24, 18))
axes = axes.flatten()

for ax_idx, mname in enumerate(metric_names):
    ax = axes[ax_idx]
    windows = pooled_windows[mname]
    if len(windows) < 3:
        ax.set_title(f'{mname} (no data)', fontsize=13)
        continue

    w = zscore_to_baseline(np.array(windows), baseline_slice)
    mean_trace = np.nanmean(w, axis=0)
    sem_trace = np.nanstd(w, axis=0) / np.sqrt(len(w))

    ax.plot(time_axis, mean_trace, color='black', linewidth=2)
    ax.fill_between(time_axis, mean_trace - sem_trace, mean_trace + sem_trace,
                    color='black', alpha=0.15)
    shade_windows(ax)

    ax.set_title(f'{mname} (n={len(w)})', fontsize=14, fontweight='bold')
    ax.set_xlabel('Time from dig onset (s)', fontsize=12)
    ax.set_ylabel('Z-score', fontsize=12)
    ax.tick_params(labelsize=11)
    ax.set_xlim(FIG_XLIM)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if ax_idx == 0:
        ax.legend(fontsize=10, loc='upper right')

fig.suptitle(f'Peri-Dig Neural Signatures — All Events (n={total_events})\n'
             f'Pre: [-3s,0s] vs Post: [+1s,+4s]',
             fontsize=18, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('figures/dp_digging_neural_pooled.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved figures/dp_digging_neural_pooled.png")

# ========================================================================
# FIGURE 2: Per-state comparison
# ========================================================================
state_colors = {'fed': '#4e79a7', 'fasted': '#e15759', 'fed-HFD': '#f28e2b'}
state_labels = {'fed': 'Fed', 'fasted': 'Fasted', 'fed-HFD': 'HFD'}

fig, axes = plt.subplots(3, 3, figsize=(24, 18))
axes = axes.flatten()

for ax_idx, mname in enumerate(metric_names):
    ax = axes[ax_idx]
    windows = pooled_windows[mname]
    if len(windows) < 3:
        ax.set_title(f'{mname} (no data)', fontsize=13)
        continue

    for state_filter, color in state_colors.items():
        if 'ACA' in mname:
            s_mask = np.array([m['state'] == state_filter and m['has_aca']
                               for m in pooled_meta])
        elif 'LHA' in mname:
            s_mask = np.array([m['state'] == state_filter and m['has_lha']
                               for m in pooled_meta])
        else:
            s_mask = np.array([m['state'] == state_filter for m in pooled_meta])

        w_list = [windows[i] for i in range(min(len(windows), len(s_mask))) if s_mask[i]]
        if len(w_list) < 2:
            continue
        w = zscore_to_baseline(np.array(w_list), baseline_slice)

        mean_trace = np.nanmean(w, axis=0)
        sem_trace = np.nanstd(w, axis=0) / np.sqrt(len(w))

        label = f'{state_labels[state_filter]} (n={len(w)})'
        ax.plot(time_axis, mean_trace, color=color, linewidth=2, label=label)
        ax.fill_between(time_axis, mean_trace - sem_trace, mean_trace + sem_trace,
                        color=color, alpha=0.12)

    shade_windows(ax)
    ax.set_title(mname, fontsize=14, fontweight='bold')
    ax.set_xlabel('Time from dig onset (s)', fontsize=12)
    ax.set_ylabel('Z-score', fontsize=12)
    ax.tick_params(labelsize=11)
    ax.set_xlim(FIG_XLIM)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=10, loc='upper right')

fig.suptitle('Peri-Dig Neural Signatures by State',
             fontsize=18, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('figures/dp_digging_neural_by_state.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved figures/dp_digging_neural_by_state.png")

# ========================================================================
# FIGURE 3: ACA vs LHA lead-lag (FR traces overlaid)
# ========================================================================
fig, axes = plt.subplots(1, 3, figsize=(24, 7))

for ax_idx, state_filter in enumerate(['fed', 'fasted', 'fed-HFD']):
    ax = axes[ax_idx]

    for mname, color, label_prefix in [
        ('ACA FR', '#4e79a7', 'ACA'),
        ('LHA FR', '#e15759', 'LHA'),
    ]:
        windows = pooled_windows[mname]
        if 'ACA' in mname:
            s_mask = np.array([m['state'] == state_filter and m['has_aca']
                               for m in pooled_meta])
        else:
            s_mask = np.array([m['state'] == state_filter and m['has_lha']
                               for m in pooled_meta])

        w_list = [windows[i] for i in range(min(len(windows), len(s_mask))) if s_mask[i]]
        if len(w_list) < 2:
            continue
        w = zscore_to_baseline(np.array(w_list), baseline_slice)

        mean_trace = np.nanmean(w, axis=0)
        sem_trace = np.nanstd(w, axis=0) / np.sqrt(len(w))

        ax.plot(time_axis, mean_trace, color=color, linewidth=2.5,
                label=f'{label_prefix} (n={len(w)})')
        ax.fill_between(time_axis, mean_trace - sem_trace, mean_trace + sem_trace,
                        color=color, alpha=0.12)

    shade_windows(ax)
    ax.set_title(state_labels.get(state_filter, state_filter),
                 fontsize=16, fontweight='bold')
    ax.set_xlabel('Time from dig onset (s)', fontsize=14)
    ax.set_ylabel('Z-score', fontsize=14)
    ax.tick_params(labelsize=12)
    ax.legend(fontsize=13, loc='upper right')
    ax.set_xlim(FIG_XLIM)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

fig.suptitle('ACA vs LHA Firing Rate Around Dig Onset',
             fontsize=18, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('figures/dp_digging_aca_vs_lha.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved figures/dp_digging_aca_vs_lha.png")

# ========================================================================
# FIGURE 4: Duration split — short vs long dig bouts
# ========================================================================
fig, axes = plt.subplots(3, 3, figsize=(24, 18))
axes = axes.flatten()
dur_colors = {'short': '#9467bd', 'long': '#2ca02c'}

for ax_idx, mname in enumerate(metric_names):
    ax = axes[ax_idx]
    windows = pooled_windows[mname]
    if len(windows) < 5:
        ax.set_title(f'{mname} (no data)', fontsize=13)
        continue

    w_all = zscore_to_baseline(np.array(windows), baseline_slice)

    for dur_group, mask, color in [
        (f'Short (<={dur_median:.0f}s)', short_mask, dur_colors['short']),
        (f'Long (>{dur_median:.0f}s)', long_mask, dur_colors['long']),
    ]:
        w = w_all[mask[:len(w_all)]]
        if len(w) < 2:
            continue
        mean_trace = np.nanmean(w, axis=0)
        sem_trace = np.nanstd(w, axis=0) / np.sqrt(len(w))

        ax.plot(time_axis, mean_trace, color=color, linewidth=2,
                label=f'{dur_group} (n={len(w)})')
        ax.fill_between(time_axis, mean_trace - sem_trace, mean_trace + sem_trace,
                        color=color, alpha=0.12)

    shade_windows(ax)
    ax.set_title(mname, fontsize=14, fontweight='bold')
    ax.set_xlabel('Time from dig onset (s)', fontsize=12)
    ax.set_ylabel('Z-score', fontsize=12)
    ax.tick_params(labelsize=11)
    ax.set_xlim(FIG_XLIM)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=10, loc='upper right')

fig.suptitle(f'Peri-Dig Neural Signatures: Short vs Long Bouts '
             f'(median={dur_median:.1f}s)',
             fontsize=18, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('figures/dp_digging_neural_duration_split.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved figures/dp_digging_neural_duration_split.png")

# ========================================================================
# FIGURE 5: Rate of change bar chart (slopes in [-3s, 0s])
# ========================================================================
fig, axes = plt.subplots(1, 3, figsize=(24, 7))

# Panel 1: All events — slope per metric
ax = axes[0]
slope_data = {}
for mname in metric_names:
    windows = pooled_windows[mname]
    if len(windows) < 5:
        continue
    w = zscore_to_baseline(np.array(windows), baseline_slice)
    slopes = compute_roc_slopes(w, roc_bin_indices, roc_times)
    valid = slopes[~np.isnan(slopes)]
    if len(valid) < 3:
        continue
    slope_data[mname] = valid

if slope_data:
    names = list(slope_data.keys())
    means = [np.mean(slope_data[n]) for n in names]
    sems = [np.std(slope_data[n]) / np.sqrt(len(slope_data[n])) for n in names]
    colors_bar = ['#e15759' if m < 0 else '#4e79a7' for m in means]
    bars = ax.bar(range(len(names)), means, yerr=sems, capsize=4,
                  color=colors_bar, edgecolor='black', linewidth=0.5)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=11)
    ax.set_ylabel('Slope (Z/s) in [-3s, 0s]', fontsize=13, fontweight='bold')
    ax.set_title('All Events', fontsize=14, fontweight='bold')
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.set_xlim(FIG_XLIM)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Add significance stars
    for i, n in enumerate(names):
        try:
            _, p = wilcoxon(slope_data[n])
        except Exception:
            p = 1.0
        if p < 0.05:
            y_pos = means[i] + sems[i] + 0.02 if means[i] > 0 else means[i] - sems[i] - 0.04
            ax.text(i, y_pos, '*', fontsize=16, fontweight='bold', ha='center', color='black')

# Panel 2: Slopes by state
ax = axes[1]
bar_width = 0.25
x_pos = np.arange(len(metric_names))
for si, (state_filter, color) in enumerate(state_colors.items()):
    slopes_state = []
    for mname in metric_names:
        windows = pooled_windows[mname]
        if len(windows) < 5:
            slopes_state.append((np.nan, np.nan))
            continue
        if 'ACA' in mname:
            s_mask = np.array([m['state'] == state_filter and m['has_aca']
                               for m in pooled_meta])
        elif 'LHA' in mname:
            s_mask = np.array([m['state'] == state_filter and m['has_lha']
                               for m in pooled_meta])
        else:
            s_mask = np.array([m['state'] == state_filter for m in pooled_meta])

        w = zscore_to_baseline(np.array(windows), baseline_slice)
        w_f = w[s_mask[:len(w)]]
        if len(w_f) < 3:
            slopes_state.append((np.nan, np.nan))
            continue
        sl = compute_roc_slopes(w_f, roc_bin_indices, roc_times)
        valid = sl[~np.isnan(sl)]
        if len(valid) < 2:
            slopes_state.append((np.nan, np.nan))
        else:
            slopes_state.append((np.mean(valid), np.std(valid) / np.sqrt(len(valid))))

    ms = [s[0] for s in slopes_state]
    es = [s[1] for s in slopes_state]
    ax.bar(x_pos + si * bar_width, ms, bar_width, yerr=es, capsize=3,
           color=color, edgecolor='black', linewidth=0.5,
           label=state_labels[state_filter])

ax.set_xticks(x_pos + bar_width)
ax.set_xticklabels(metric_names, rotation=45, ha='right', fontsize=10)
ax.set_ylabel('Slope (Z/s) in [-3s, 0s]', fontsize=13, fontweight='bold')
ax.set_title('By State', fontsize=14, fontweight='bold')
ax.axhline(0, color='gray', linewidth=0.5)
ax.legend(fontsize=11)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Panel 3: Slopes by duration
ax = axes[2]
bar_width = 0.35
for di, (dur_label, mask, color) in enumerate([
    (f'Short', short_mask, dur_colors['short']),
    (f'Long', long_mask, dur_colors['long']),
]):
    slopes_dur = []
    for mname in metric_names:
        windows = pooled_windows[mname]
        if len(windows) < 5:
            slopes_dur.append((np.nan, np.nan))
            continue
        w = zscore_to_baseline(np.array(windows), baseline_slice)
        w_d = w[mask[:len(w)]]
        if len(w_d) < 3:
            slopes_dur.append((np.nan, np.nan))
            continue
        sl = compute_roc_slopes(w_d, roc_bin_indices, roc_times)
        valid = sl[~np.isnan(sl)]
        if len(valid) < 2:
            slopes_dur.append((np.nan, np.nan))
        else:
            slopes_dur.append((np.mean(valid), np.std(valid) / np.sqrt(len(valid))))

    ms = [s[0] for s in slopes_dur]
    es = [s[1] for s in slopes_dur]
    ax.bar(x_pos + di * bar_width, ms, bar_width, yerr=es, capsize=3,
           color=color, edgecolor='black', linewidth=0.5, label=dur_label)

ax.set_xticks(x_pos + bar_width / 2)
ax.set_xticklabels(metric_names, rotation=45, ha='right', fontsize=10)
ax.set_ylabel('Slope (Z/s) in [-3s, 0s]', fontsize=13, fontweight='bold')
ax.set_title('By Duration', fontsize=14, fontweight='bold')
ax.axhline(0, color='gray', linewidth=0.5)
ax.legend(fontsize=11)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig.suptitle('Pre-Dig Rate of Change [-3s to 0s]',
             fontsize=18, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('figures/dp_digging_neural_roc.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved figures/dp_digging_neural_roc.png")

# ========================================================================
# FIGURE 6: Per-session individual bout traces
# ========================================================================
per_session_dir = Path('figures/dp_digging_per_session')
per_session_dir.mkdir(exist_ok=True)

# Colormap for individual bouts
bout_cmap = plt.cm.tab10

# Group events by session
sessions_with_events = sorted(set(m['session'] for m in pooled_meta))

for snum in sessions_with_events:
    s_indices = [i for i, m in enumerate(pooled_meta) if m['session'] == snum]
    if len(s_indices) == 0:
        continue

    meta0 = pooled_meta[s_indices[0]]
    state, phase = meta0['state'], meta0['phase']
    n_bouts = len(s_indices)

    fig, axes = plt.subplots(3, 3, figsize=(24, 18))
    axes = axes.flatten()

    for ax_idx, mname in enumerate(metric_names):
        ax = axes[ax_idx]
        windows_all = pooled_windows[mname]

        # Filter indices for this session that have data for this metric
        if 'ACA' in mname:
            valid_idx = [i for i in s_indices if pooled_meta[i]['has_aca'] and i < len(windows_all)]
        elif 'LHA' in mname:
            valid_idx = [i for i in s_indices if pooled_meta[i]['has_lha'] and i < len(windows_all)]
        else:
            valid_idx = [i for i in s_indices if i < len(windows_all)]

        if len(valid_idx) == 0:
            ax.set_title(f'{mname} (no data)', fontsize=13)
            continue

        # Z-score each bout to its own baseline
        for bi, idx in enumerate(valid_idx):
            w = windows_all[idx].copy()
            bl = w[baseline_slice]
            bl_mean, bl_std = np.nanmean(bl), np.nanstd(bl)
            if bl_std > 1e-6:
                w = (w - bl_mean) / bl_std
            else:
                w = w - bl_mean

            pot = pooled_meta[idx].get('pot', '?')
            dur = pooled_meta[idx]['duration']
            color = bout_cmap(bi % 10)
            ax.plot(time_axis, w, color=color, linewidth=1.5, alpha=0.7,
                    label=f'B{bi+1} {pot} {dur:.0f}s')

        shade_windows(ax)
        ax.set_xlim(FIG_XLIM)
        ax.set_title(f'{mname} ({len(valid_idx)} bouts)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Time from dig onset (s)', fontsize=12)
        ax.set_ylabel('Z-score', fontsize=12)
        ax.tick_params(labelsize=11)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if len(valid_idx) <= 15:
            ax.legend(fontsize=8, loc='upper right', ncol=2)

    fig.suptitle(f'S{snum} — {state.capitalize()} / {phase.capitalize()} — '
                 f'{n_bouts} Dig Bouts (Individual Traces)',
                 fontsize=18, fontweight='bold', y=1.01)
    plt.tight_layout()
    fname = per_session_dir / f'S{snum}_{state}_{phase}_dig_bouts.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {fname}")

# Save CSVs
df_events = pd.DataFrame(all_event_data)
df_events.to_csv('data/dp_digging_events.csv', index=False)

df_results = pd.DataFrame(results_rows)
df_results.to_csv('data/dp_digging_neural_stats.csv', index=False)

print(f"\nSaved data/dp_digging_events.csv ({len(df_events)} rows)")
print(f"Saved data/dp_digging_neural_stats.csv ({len(df_results)} rows)")
print("\nDone.")
