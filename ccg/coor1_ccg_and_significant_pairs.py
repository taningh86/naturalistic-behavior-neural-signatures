"""
Coordinates-1 Mouse01: Full CCG (-500 to +500ms) + significant pair detection.
Step 1: Compute mean CCG at 1ms resolution for LHA-LHA, RSP-RSP, LHA-RSP across all 8 sessions.
Step 2: Identify the peak co-occurrence time window from CCG results.
Step 3: Find all pairs with significant co-occurrence using GPU + circular-shift testing.

Unit selection: KSLabel=='good' AND fr > 0.3 Hz AND amp > 48 uV
"""

import yaml
from pathlib import Path
import numpy as np
import pandas as pd
import cupy as cp
import spikeinterface.extractors as se
import warnings
import time
import sys

warnings.filterwarnings('ignore')

# Log output when running unattended
log_path = Path("data/coor1_ccg_run.log")
log_file = open(log_path, 'w')

class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()

sys.stdout = Tee(sys.__stdout__, log_file)
sys.stderr = Tee(sys.__stderr__, log_file)

with open("paths.yaml") as f:
    paths_config = yaml.safe_load(f)

LHA_DEPTH_MAX = 1300
RSP_DEPTH_MIN = 1300
FS = 30000
BIN_SIZE_MS = 1
BIN_SAMPLES = int(BIN_SIZE_MS * FS / 1000)

MAX_LAG_MS = 500
LAGS = np.arange(-MAX_LAG_MS, MAX_LAG_MS + 1)

N_SHUFFLES = 500
MIN_SHIFT_BINS = 1000  # 1 second


MIN_FR = 0.3       # Hz
MIN_AMP = 48       # uV

def get_good_units_by_region(sorted_path_obj):
    ci = sorted_path_obj / "cluster_info.tsv"
    if not ci.exists():
        return np.array([]), np.array([])
    df = pd.read_csv(ci, sep='\t')
    label_col = None
    if 'group' in df.columns and df['group'].eq('good').any():
        label_col = 'group'
    elif 'KSLabel' in df.columns:
        label_col = 'KSLabel'
    if label_col is None:
        return np.array([]), np.array([])

    n_total = len(df)
    ks_good = df[df[label_col] == 'good']
    n_ks = len(ks_good)
    fr_pass = ks_good[ks_good['fr'] > MIN_FR]
    n_fr = len(fr_pass)
    good = fr_pass[fr_pass['amp'] > MIN_AMP]
    n_final = len(good)
    print(f"    Unit selection: {n_total} total -> {n_ks} KS good -> {n_fr} fr>{MIN_FR} -> {n_final} amp>{MIN_AMP}")

    lha_ids = good[good['depth'] < LHA_DEPTH_MAX]['cluster_id'].values
    rsp_ids = good[good['depth'] >= RSP_DEPTH_MIN]['cluster_id'].values
    return lha_ids, rsp_ids


def prebin_spike_trains(sorting, unit_ids):
    spike_trains = {}
    for uid in unit_ids:
        spike_trains[uid] = sorting.get_unit_spike_train(uid)
    all_min, all_max = np.inf, 0
    for uid in unit_ids:
        st = spike_trains[uid]
        if len(st) > 0:
            all_min = min(all_min, np.min(st))
            all_max = max(all_max, np.max(st))
    n_bins = int((all_max - all_min) / BIN_SAMPLES) + 1
    binned = {}
    for uid in unit_ids:
        st = spike_trains[uid]
        t = np.zeros(n_bins)
        if len(st) > 0:
            b = ((st - all_min) // BIN_SAMPLES).astype(int)
            b = b[(b >= 0) & (b < n_bins)]
            np.add.at(t, b, 1)
        std_val = np.std(t)
        if std_val > 1e-8:
            t = (t - np.mean(t)) / std_val
        else:
            t = t - np.mean(t)
        binned[uid] = t
    return binned, n_bins


def fast_ccg_gpu(t1_gpu, t2_gpu, n_bins, max_lag):
    ccg = cp.empty(2 * max_lag + 1, dtype=cp.float64)
    for i, lag in enumerate(range(-max_lag, max_lag + 1)):
        if lag == 0:
            ccg[i] = cp.dot(t1_gpu, t2_gpu) / n_bins
        elif lag > 0:
            ccg[i] = cp.dot(t1_gpu[lag:], t2_gpu[:-lag]) / (n_bins - lag)
        else:
            alag = -lag
            ccg[i] = cp.dot(t1_gpu[:-alag], t2_gpu[alag:]) / (n_bins - alag)
    return ccg


def compute_mean_ccg(binned_gpu, ids_a, ids_b, n_bins, same_region=False):
    ccg_sum = cp.zeros(2 * MAX_LAG_MS + 1, dtype=cp.float64)
    n_pairs = 0
    if same_region:
        total = len(ids_a) * (len(ids_a) - 1) // 2
        for i in range(len(ids_a)):
            for j in range(i + 1, len(ids_a)):
                ccg_sum += fast_ccg_gpu(binned_gpu[ids_a[i]], binned_gpu[ids_a[j]], n_bins, MAX_LAG_MS)
                n_pairs += 1
                if n_pairs % 500 == 0:
                    print(f"      {n_pairs}/{total} pairs...", flush=True)
    else:
        total = len(ids_a) * len(ids_b)
        for a_uid in ids_a:
            t1 = binned_gpu[a_uid]
            for b_uid in ids_b:
                ccg_sum += fast_ccg_gpu(t1, binned_gpu[b_uid], n_bins, MAX_LAG_MS)
                n_pairs += 1
                if n_pairs % 500 == 0:
                    print(f"      {n_pairs}/{total} pairs...", flush=True)
    if n_pairs > 0:
        ccg_sum /= n_pairs
    return cp.asnumpy(ccg_sum), n_pairs


def compute_peak_matrix(X_gpu, n_bins, peak_window):
    n_units = X_gpu.shape[0]
    n_lags = 2 * peak_window + 1
    all_corr = cp.zeros((n_lags, n_units, n_units), dtype=cp.float32)
    for i, lag in enumerate(range(-peak_window, peak_window + 1)):
        if lag == 0:
            all_corr[i] = cp.dot(X_gpu, X_gpu.T) / n_bins
        elif lag > 0:
            all_corr[i] = cp.dot(X_gpu[:, lag:], X_gpu[:, :-lag].T) / (n_bins - lag)
        else:
            alag = -lag
            all_corr[i] = cp.dot(X_gpu[:, :-alag], X_gpu[:, alag:].T) / (n_bins - alag)
    abs_corr = cp.abs(all_corr)
    peak_lag_idx = cp.argmax(abs_corr, axis=0)
    rows = cp.arange(n_units)[:, None]
    cols = cp.arange(n_units)[None, :]
    peak_vals = all_corr[peak_lag_idx, rows, cols]
    peak_lags = peak_lag_idx - peak_window
    return peak_vals, peak_lags


def fdr_bh(p_values, alpha=0.05):
    n = len(p_values)
    if n == 0:
        return np.array([], dtype=bool)
    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]
    thresholds = np.arange(1, n + 1) / n * alpha
    below = sorted_p <= thresholds
    if not np.any(below):
        return np.zeros(n, dtype=bool)
    max_k = np.max(np.where(below))
    result = np.zeros(n, dtype=bool)
    result[sorted_idx[:max_k + 1]] = True
    return result


# =============================================================================
# STEP 1: FULL CCG (-500 to +500ms)
# =============================================================================
sessions = paths_config["single_probe"]["coordinates_1"]["mouse01"]["sessions"]
session_meta = {
    1: ('fed', 'exploration'), 2: ('fed', 'foraging'),
    3: ('fed', 'exploration'), 4: ('fed', 'foraging'),
    5: ('fasted', 'exploration'), 6: ('fasted', 'foraging'),
    7: ('fasted', 'exploration'), 8: ('fasted', 'foraging'),
}

session_unit_info = {}

# Delete stale cached results (criteria changed)
for old_file in ["data/coor1_ccg_summary.csv", "data/coor1_ccg_full.npz",
                 "data/coor1_all_pairs_significance.csv", "data/coor1_significant_pairs.csv"]:
    p = Path(old_file)
    if p.exists():
        p.unlink()
        print(f"  Deleted stale: {old_file}")

print("=" * 90)
print("STEP 1: CROSS-CORRELOGRAMS (-500 to +500ms)")
print(f"  Unit criteria: KSLabel=='good' AND fr>{MIN_FR} AND amp>{MIN_AMP}")
print("=" * 90)

all_ccg_results = []
all_ccgs = {}

for snum in range(1, 9):
    sname = f'session_{snum}'
    sc = sessions[sname]
    sp = Path(sc['sorted'])
    state, phase = session_meta[snum]

    lha_ids, rsp_ids = get_good_units_by_region(sp)
    print(f"\nSession {snum} ({state}/{phase}): LHA={len(lha_ids)}, RSP={len(rsp_ids)}")

    sorting = se.read_kilosort(sp)
    avail = set(sorting.get_unit_ids())
    lha_ids = np.array([u for u in lha_ids if u in avail])
    rsp_ids = np.array([u for u in rsp_ids if u in avail])

    all_ids = np.concatenate([lha_ids, rsp_ids])
    if len(all_ids) < 2:
        print("  SKIP - too few units")
        continue

    t0 = time.time()
    binned, n_bins = prebin_spike_trains(sorting, all_ids)
    binned_gpu = {uid: cp.asarray(binned[uid]) for uid in all_ids}
    print(f"  Binned {len(all_ids)} units into {n_bins} bins, transferred to GPU in {time.time()-t0:.1f}s")

    session_unit_info[snum] = {
        'lha_ids': lha_ids, 'rsp_ids': rsp_ids, 'all_ids': all_ids,
        'n_bins': n_bins, 'state': state, 'phase': phase,
    }

    for nt, ids_a, ids_b, same in [
        ('LHA-LHA', lha_ids, lha_ids, True),
        ('RSP-RSP', rsp_ids, rsp_ids, True),
        ('LHA-RSP', lha_ids, rsp_ids, False),
    ]:
        if (same and len(ids_a) < 2) or (not same and (len(ids_a) < 1 or len(ids_b) < 1)):
            print(f"  {nt}: SKIP")
            continue

        t1 = time.time()
        ccg, n_pairs = compute_mean_ccg(binned_gpu, ids_a, ids_b, n_bins, same_region=same)
        elapsed = time.time() - t1
        all_ccgs[(snum, nt)] = ccg

        if same:
            search_ccg = ccg.copy()
            search_ccg[LAGS == 0] = 0
        else:
            search_ccg = ccg
        peak_idx = np.argmax(np.abs(search_ccg))
        peak_lag = LAGS[peak_idx]
        peak_val = search_ccg[peak_idx]

        zero_val = ccg[LAGS == 0][0]
        val_1 = np.mean([ccg[LAGS == 1][0], ccg[LAGS == -1][0]])
        val_5 = np.mean([ccg[LAGS == 5][0], ccg[LAGS == -5][0]])
        val_10 = np.mean([ccg[LAGS == 10][0], ccg[LAGS == -10][0]])
        val_50 = np.mean([ccg[LAGS == 50][0], ccg[LAGS == -50][0]])
        val_100 = np.mean([ccg[LAGS == 100][0], ccg[LAGS == -100][0]])
        val_200 = np.mean([ccg[LAGS == 200][0], ccg[LAGS == -200][0]])
        val_300 = np.mean([ccg[LAGS == 300][0], ccg[LAGS == -300][0]])
        val_500 = np.mean([ccg[LAGS == 500][0], ccg[LAGS == -500][0]])

        abs_ccg = np.abs(ccg)
        half_max = np.max(abs_ccg) / 2
        above_half = np.where(abs_ccg >= half_max)[0]
        hwhm = (LAGS[above_half[-1]] - LAGS[above_half[0]]) / 2 if len(above_half) > 1 else 0

        print(f"  {nt}: {n_pairs} pairs in {elapsed:.1f}s")
        print(f"    Peak lag: {peak_lag}ms (r={peak_val:.7f})")
        print(f"    lag0={zero_val:.7f}  +/-1ms={val_1:.7f}  +/-5ms={val_5:.7f}  "
              f"+/-10ms={val_10:.7f}  +/-50ms={val_50:.7f}")
        print(f"    +/-100ms={val_100:.7f}  +/-200ms={val_200:.7f}  "
              f"+/-300ms={val_300:.7f}  +/-500ms={val_500:.7f}  HWHM={hwhm:.0f}ms")

        all_ccg_results.append({
            'session': snum, 'state': state, 'phase': phase, 'network': nt,
            'n_pairs': n_pairs, 'peak_lag_ms': peak_lag, 'peak_value': peak_val,
            'lag0': zero_val, 'lag1': val_1, 'lag5': val_5, 'lag10': val_10,
            'lag50': val_50, 'lag100': val_100, 'lag200': val_200,
            'lag300': val_300, 'lag500': val_500, 'hwhm_ms': hwhm,
        })

    del binned_gpu
    cp.get_default_memory_pool().free_all_blocks()

ccg_df = pd.DataFrame(all_ccg_results)
ccg_df.to_csv("data/coor1_ccg_summary.csv", index=False)

ccg_data = {'lags': LAGS}
for (snum, nt), ccg in all_ccgs.items():
    ccg_data[f's{snum}_{nt}'] = ccg
np.savez("data/coor1_ccg_full.npz", **ccg_data)

print("\n" + "=" * 90)
print("CCG SUMMARY BY NETWORK")
print("=" * 90)
for nt in ['LHA-LHA', 'RSP-RSP', 'LHA-RSP']:
    sub = ccg_df[ccg_df['network'] == nt]
    print(f"\n  {nt}:")
    for _, r in sub.iterrows():
        print(f"    S{int(r['session'])} ({r['state']}/{r['phase']}): "
              f"peak={int(r['peak_lag_ms'])}ms r={r['peak_value']:.7f} HWHM={r['hwhm_ms']:.0f}ms")
    for state in ['fed', 'fasted']:
        ps = sub[sub['state'] == state]
        if len(ps) > 0:
            print(f"    AVG {state}: lag0={ps['lag0'].mean():.7f} lag1={ps['lag1'].mean():.7f} "
                  f"lag5={ps['lag5'].mean():.7f} lag10={ps['lag10'].mean():.7f} "
                  f"lag50={ps['lag50'].mean():.7f} lag100={ps['lag100'].mean():.7f}")


# =============================================================================
# STEP 2: DETERMINE PEAK WINDOWS
# =============================================================================
print("\n\n" + "=" * 90)
print("STEP 2: PEAK CO-OCCURRENCE WINDOWS")
print("=" * 90)

# For each network, find where most CCG peaks fall
peak_windows = {}
for nt in ['LHA-LHA', 'RSP-RSP', 'LHA-RSP']:
    sub = ccg_df[ccg_df['network'] == nt]
    peaks = sub['peak_lag_ms'].values
    abs_peaks = np.abs(peaks)

    # Count peaks in windows
    windows = {'0-1ms': 0, '2-5ms': 0, '6-10ms': 0, '11-50ms': 0, '51-100ms': 0,
               '101-200ms': 0, '201-500ms': 0}
    for p in abs_peaks:
        if p <= 1: windows['0-1ms'] += 1
        elif p <= 5: windows['2-5ms'] += 1
        elif p <= 10: windows['6-10ms'] += 1
        elif p <= 50: windows['11-50ms'] += 1
        elif p <= 100: windows['51-100ms'] += 1
        elif p <= 200: windows['101-200ms'] += 1
        else: windows['201-500ms'] += 1

    # Determine the primary and secondary windows for significance testing
    primary_window = max(windows, key=windows.get)

    # Map window name to lag range for significance testing
    window_to_lags = {
        '0-1ms': 1, '2-5ms': 5, '6-10ms': 10, '11-50ms': 50,
        '51-100ms': 100, '101-200ms': 200, '201-500ms': 500,
    }

    # Use a generous window that captures the primary peak
    # If most peaks are at 0-1ms, test +/-5ms (standard for fast interactions)
    # But also test broader windows if HWHM suggests broader structure
    mean_hwhm = sub['hwhm_ms'].mean()

    # Primary test window: based on where peaks concentrate
    if primary_window == '0-1ms':
        test_window = 5  # +/-5ms captures the sharp peak
    else:
        test_window = window_to_lags[primary_window]

    # Note: broad window disabled — inflated HWHM in LHA-RSP is an artifact
    # of flat CCGs near zero, not real slow structure. All peaks are at 0-1ms.
    broad_window = None

    peak_windows[nt] = {
        'distribution': windows,
        'primary_window': primary_window,
        'test_window_ms': test_window,
        'mean_hwhm': mean_hwhm,
        'broad_window_ms': broad_window,
    }

    print(f"\n  {nt}:")
    print(f"    Peak distribution: {windows}")
    print(f"    Primary window: {primary_window}")
    print(f"    Mean HWHM: {mean_hwhm:.0f}ms")
    print(f"    Test window for significance: +/-{test_window}ms")
    if broad_window:
        print(f"    Also testing broad window: +/-{broad_window}ms")


# =============================================================================
# STEP 3: SIGNIFICANT PAIRS (GPU shuffle test)
# =============================================================================
print("\n\n" + "=" * 90)
print("STEP 3: SIGNIFICANT PAIR DETECTION")
print("=" * 90)

all_pairs_list = []

for snum in range(1, 9):
    sname = f'session_{snum}'
    if snum not in session_unit_info:
        continue

    info = session_unit_info[snum]
    lha_ids = info['lha_ids']
    rsp_ids = info['rsp_ids']
    all_ids = info['all_ids']
    n_bins = info['n_bins']
    state = info['state']
    phase = info['phase']
    n_lha = len(lha_ids)
    n_units = len(all_ids)

    sc = sessions[sname]
    sp = Path(sc['sorted'])
    print(f"\nSession {snum} ({state}/{phase}): {n_lha} LHA + {len(rsp_ids)} RSP = {n_units} units")

    # Reload and bin (we freed GPU memory after step 1)
    sorting = se.read_kilosort(sp)
    binned, n_bins = prebin_spike_trains(sorting, all_ids)

    X = np.zeros((n_units, n_bins), dtype=np.float32)
    for i, uid in enumerate(all_ids):
        X[i] = binned[uid].astype(np.float32)
    X_gpu = cp.asarray(X)
    del X

    # Determine which test windows to use for each network
    # We'll run ALL relevant windows and combine results
    test_configs = []
    for nt in ['LHA-LHA', 'RSP-RSP', 'LHA-RSP']:
        pw = peak_windows[nt]
        test_configs.append((nt, pw['test_window_ms']))
        if pw['broad_window_ms'] is not None:
            test_configs.append((nt, pw['broad_window_ms']))

    # Get unique windows needed
    unique_windows = sorted(set(w for _, w in test_configs))
    print(f"  Testing windows: {unique_windows}ms")

    # For each window size, compute observed peaks and run shuffle test
    window_results = {}
    for win in unique_windows:
        t1 = time.time()
        obs_peak_vals, obs_peak_lags = compute_peak_matrix(X_gpu, n_bins, win)
        obs_abs = cp.abs(obs_peak_vals)

        exceed_count = cp.zeros((n_units, n_units), dtype=cp.int32)
        X_shuffled = cp.empty_like(X_gpu)

        for s in range(N_SHUFFLES):
            shifts = np.random.randint(MIN_SHIFT_BINS, n_bins - MIN_SHIFT_BINS, size=n_units)
            for i in range(n_units):
                X_shuffled[i] = cp.roll(X_gpu[i], int(shifts[i]))
            shuf_peak_vals, _ = compute_peak_matrix(X_shuffled, n_bins, win)
            shuf_abs = cp.abs(shuf_peak_vals)
            exceed_count += (shuf_abs >= obs_abs).astype(cp.int32)
            if (s + 1) % 100 == 0:
                print(f"    Window +/-{win}ms: Shuffle {s+1}/{N_SHUFFLES}...", flush=True)

        p_values = cp.asnumpy((exceed_count.astype(cp.float32) + 1) / (N_SHUFFLES + 1))
        obs_peak_vals_cpu = cp.asnumpy(obs_peak_vals)
        obs_peak_lags_cpu = cp.asnumpy(obs_peak_lags)

        window_results[win] = {
            'obs_peak_vals': obs_peak_vals_cpu,
            'obs_peak_lags': obs_peak_lags_cpu,
            'p_values': p_values,
        }
        print(f"    Window +/-{win}ms done in {time.time()-t1:.1f}s")

    del X_shuffled

    # Collect pairs for each network with appropriate window
    for nt in ['LHA-LHA', 'RSP-RSP', 'LHA-RSP']:
        pw = peak_windows[nt]
        primary_win = pw['test_window_ms']
        broad_win = pw['broad_window_ms']

        # Start with primary window results
        res = window_results[primary_win]
        pairs = []
        for i in range(n_units):
            for j in range(i + 1, n_units):
                i_is_lha = i < n_lha
                j_is_lha = j < n_lha
                if i_is_lha and j_is_lha:
                    pair_nt = 'LHA-LHA'
                elif not i_is_lha and not j_is_lha:
                    pair_nt = 'RSP-RSP'
                else:
                    pair_nt = 'LHA-RSP'

                if pair_nt != nt:
                    continue

                p_primary = res['p_values'][i, j]
                corr_primary = res['obs_peak_vals'][i, j]
                lag_primary = res['obs_peak_lags'][i, j]

                # Check broad window too if applicable
                p_broad = np.nan
                corr_broad = np.nan
                lag_broad = np.nan
                if broad_win and broad_win in window_results:
                    bres = window_results[broad_win]
                    p_broad = bres['p_values'][i, j]
                    corr_broad = bres['obs_peak_vals'][i, j]
                    lag_broad = bres['obs_peak_lags'][i, j]

                # Use whichever window gives the more significant result
                if not np.isnan(p_broad) and p_broad < p_primary:
                    best_p = p_broad
                    best_corr = corr_broad
                    best_lag = lag_broad
                    best_win = broad_win
                else:
                    best_p = p_primary
                    best_corr = corr_primary
                    best_lag = lag_primary
                    best_win = primary_win

                pairs.append({
                    'session': snum, 'state': state, 'phase': phase,
                    'network': nt,
                    'unit_a': int(all_ids[i]), 'unit_b': int(all_ids[j]),
                    'peak_lag_ms': int(best_lag), 'peak_corr': float(best_corr),
                    'p_value': float(best_p), 'test_window_ms': int(best_win),
                })

        if len(pairs) == 0:
            continue

        pairs_df = pd.DataFrame(pairs)
        # FDR correction
        sig = fdr_bh(pairs_df['p_value'].values, alpha=0.05)
        pairs_df['significant'] = sig
        all_pairs_list.append(pairs_df)

        n_sig = sig.sum()
        print(f"  {nt}: {n_sig}/{len(pairs_df)} significant ({100*n_sig/len(pairs_df):.1f}%)")

    del X_gpu
    cp.get_default_memory_pool().free_all_blocks()

# Save all results
result = pd.concat(all_pairs_list, ignore_index=True)
result.to_csv("data/coor1_all_pairs_significance.csv", index=False)

sig_result = result[result['significant'] == True].copy()
sig_result.to_csv("data/coor1_significant_pairs.csv", index=False)

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n\n" + "=" * 90)
print("SIGNIFICANT CO-OCCURRENCE PAIRS BY SESSION (FDR < 0.05)")
print("=" * 90)

for snum in range(1, 9):
    if snum not in session_unit_info:
        continue
    info = session_unit_info[snum]
    s_df = sig_result[sig_result['session'] == snum]
    print(f"\n--- Session {snum} ({info['state']}/{info['phase']}): {len(s_df)} significant pairs ---")

    for nt in ['LHA-LHA', 'RSP-RSP', 'LHA-RSP']:
        nt_df = s_df[s_df['network'] == nt].sort_values('peak_corr', ascending=False)
        total_nt = len(result[(result['session'] == snum) & (result['network'] == nt)])
        if len(nt_df) > 0:
            print(f"\n  {nt}: {len(nt_df)}/{total_nt} pairs significant "
                  f"({100*len(nt_df)/total_nt:.1f}%)")
            show = nt_df.head(10)
            for _, r in show.iterrows():
                print(f"    Units {int(r['unit_a']):>4} - {int(r['unit_b']):>4}: "
                      f"r={r['peak_corr']:+.6f} at lag={int(r['peak_lag_ms']):+d}ms "
                      f"(win=+/-{int(r['test_window_ms'])}ms), p={r['p_value']:.4f}")
            if len(nt_df) > 10:
                print(f"    ... ({len(nt_df) - 10} more pairs)")
        else:
            print(f"\n  {nt}: 0/{total_nt} pairs significant")

# Summary table
print("\n" + "=" * 90)
print("SUMMARY TABLE")
print("=" * 90)
print(f"  {'Sess':>4} {'State':>7} {'Phase':>11} {'Network':>8} {'Total':>6} {'Sig':>5} "
      f"{'%':>6} {'Mean r(sig)':>11} {'Max r':>9}")

for snum in range(1, 9):
    if snum not in session_unit_info:
        continue
    info = session_unit_info[snum]
    for nt in ['LHA-LHA', 'RSP-RSP', 'LHA-RSP']:
        total = len(result[(result['session'] == snum) & (result['network'] == nt)])
        sig = sig_result[(sig_result['session'] == snum) & (sig_result['network'] == nt)]
        n_sig = len(sig)
        pct = 100 * n_sig / total if total > 0 else 0
        mean_r = sig['peak_corr'].mean() if n_sig > 0 else 0
        max_r = sig['peak_corr'].max() if n_sig > 0 else 0
        print(f"  {snum:>4} {info['state']:>7} {info['phase']:>11} {nt:>8} "
              f"{total:>6} {n_sig:>5} {pct:>5.1f}% "
              f"{mean_r:>+10.6f} {max_r:>+8.6f}")

print("\n[DONE]")
