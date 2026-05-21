"""
Coordinates-1 Mouse01: RESUME significant pair detection (Step 3 only).
Loads cached CCG results from Steps 1+2, then runs shuffle-based significance
testing with per-session checkpoint saves so progress survives crashes.

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

# Append to log instead of overwriting
log_path = Path("data/coor1_ccg_run.log")
log_file = open(log_path, 'a')

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

print("\n\n" + "=" * 90)
print("RESUMING STEP 3: SIGNIFICANT PAIR DETECTION")
print(f"  Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 90)

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

MIN_FR = 0.3
MIN_AMP = 48

CHECKPOINT_DIR = Path("data/coor1_ccg_checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)


# ── Load cached CCG summary to reconstruct peak_windows ──
ccg_df = pd.read_csv("data/coor1_ccg_summary.csv")
print(f"  Loaded cached CCG summary: {len(ccg_df)} rows")

peak_windows = {}
for nt in ['LHA-LHA', 'RSP-RSP', 'LHA-RSP']:
    sub = ccg_df[ccg_df['network'] == nt]
    peaks = sub['peak_lag_ms'].values
    abs_peaks = np.abs(peaks)

    windows = {'0-1ms': 0, '2-5ms': 0, '6-10ms': 0, '11-50ms': 0,
               '51-100ms': 0, '101-200ms': 0, '201-500ms': 0}
    for p in abs_peaks:
        if p <= 1: windows['0-1ms'] += 1
        elif p <= 5: windows['2-5ms'] += 1
        elif p <= 10: windows['6-10ms'] += 1
        elif p <= 50: windows['11-50ms'] += 1
        elif p <= 100: windows['51-100ms'] += 1
        elif p <= 200: windows['101-200ms'] += 1
        else: windows['201-500ms'] += 1

    primary_window = max(windows, key=windows.get)
    window_to_lags = {
        '0-1ms': 1, '2-5ms': 5, '6-10ms': 10, '11-50ms': 50,
        '51-100ms': 100, '101-200ms': 200, '201-500ms': 500,
    }

    if primary_window == '0-1ms':
        test_window = 5
    else:
        test_window = window_to_lags[primary_window]

    peak_windows[nt] = {
        'test_window_ms': test_window,
        'broad_window_ms': None,
    }
    print(f"  {nt}: test window = +/-{test_window}ms")


# ── Helper functions ──

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


# ── Step 3: Significant pairs with per-session checkpoints ──

sessions = paths_config["single_probe"]["coordinates_1"]["mouse01"]["sessions"]
session_meta = {
    1: ('fed', 'exploration'), 2: ('fed', 'foraging'),
    3: ('fed', 'exploration'), 4: ('fed', 'foraging'),
    5: ('fasted', 'exploration'), 6: ('fasted', 'foraging'),
    7: ('fasted', 'exploration'), 8: ('fasted', 'foraging'),
}

for snum in range(1, 9):
    checkpoint_file = CHECKPOINT_DIR / f"session_{snum}_pairs.csv"
    if checkpoint_file.exists():
        print(f"\nSession {snum}: CHECKPOINT EXISTS — skipping")
        continue

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
    n_lha = len(lha_ids)
    n_units = len(all_ids)

    if n_units < 2:
        print("  SKIP — too few units")
        continue

    binned, n_bins = prebin_spike_trains(sorting, all_ids)

    X = np.zeros((n_units, n_bins), dtype=np.float32)
    for i, uid in enumerate(all_ids):
        X[i] = binned[uid].astype(np.float32)
    X_gpu = cp.asarray(X)
    del X, binned

    # Determine test windows
    test_configs = []
    for nt in ['LHA-LHA', 'RSP-RSP', 'LHA-RSP']:
        pw = peak_windows[nt]
        test_configs.append((nt, pw['test_window_ms']))
        if pw['broad_window_ms'] is not None:
            test_configs.append((nt, pw['broad_window_ms']))

    unique_windows = sorted(set(w for _, w in test_configs))
    print(f"  Testing windows: {unique_windows}ms")

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

    # Collect pairs
    session_pairs = []
    for nt in ['LHA-LHA', 'RSP-RSP', 'LHA-RSP']:
        pw = peak_windows[nt]
        primary_win = pw['test_window_ms']
        broad_win = pw['broad_window_ms']

        res = window_results[primary_win]
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

                p_broad = np.nan
                corr_broad = np.nan
                lag_broad = np.nan
                if broad_win and broad_win in window_results:
                    bres = window_results[broad_win]
                    p_broad = bres['p_values'][i, j]
                    corr_broad = bres['obs_peak_vals'][i, j]
                    lag_broad = bres['obs_peak_lags'][i, j]

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

                session_pairs.append({
                    'session': snum, 'state': state, 'phase': phase,
                    'network': nt,
                    'unit_a': int(all_ids[i]), 'unit_b': int(all_ids[j]),
                    'peak_lag_ms': int(best_lag), 'peak_corr': float(best_corr),
                    'p_value': float(best_p), 'test_window_ms': int(best_win),
                })

    if len(session_pairs) > 0:
        pairs_df = pd.DataFrame(session_pairs)
        sig = fdr_bh(pairs_df['p_value'].values, alpha=0.05)
        pairs_df['significant'] = sig

        # Save checkpoint
        pairs_df.to_csv(checkpoint_file, index=False)
        print(f"  CHECKPOINT SAVED: {checkpoint_file}")

        for nt in ['LHA-LHA', 'RSP-RSP', 'LHA-RSP']:
            nt_df = pairs_df[pairs_df['network'] == nt]
            n_sig = nt_df['significant'].sum()
            print(f"  {nt}: {n_sig}/{len(nt_df)} significant ({100*n_sig/len(nt_df):.1f}%)")

    del X_gpu
    cp.get_default_memory_pool().free_all_blocks()


# ── Combine all checkpoints into final output ──
print("\n\n" + "=" * 90)
print("COMBINING CHECKPOINTS INTO FINAL OUTPUT")
print("=" * 90)

all_parts = []
for snum in range(1, 9):
    checkpoint_file = CHECKPOINT_DIR / f"session_{snum}_pairs.csv"
    if checkpoint_file.exists():
        df = pd.read_csv(checkpoint_file)
        all_parts.append(df)
        print(f"  Loaded session {snum}: {len(df)} pairs")
    else:
        print(f"  WARNING: session {snum} checkpoint missing!")

if len(all_parts) > 0:
    result = pd.concat(all_parts, ignore_index=True)
    result.to_csv("data/coor1_all_pairs_significance.csv", index=False)

    sig_result = result[result['significant'] == True].copy()
    sig_result.to_csv("data/coor1_significant_pairs.csv", index=False)

    print(f"\n  Total pairs: {len(result)}")
    print(f"  Significant pairs: {len(sig_result)}")

    # Summary table
    print("\n" + "=" * 90)
    print("SUMMARY TABLE")
    print("=" * 90)
    print(f"  {'Sess':>4} {'State':>7} {'Phase':>11} {'Network':>8} {'Total':>6} {'Sig':>5} "
          f"{'%':>6} {'Mean r(sig)':>11} {'Max r':>9}")

    for snum in range(1, 9):
        state, phase = session_meta[snum]
        for nt in ['LHA-LHA', 'RSP-RSP', 'LHA-RSP']:
            total = len(result[(result['session'] == snum) & (result['network'] == nt)])
            sig = sig_result[(sig_result['session'] == snum) & (sig_result['network'] == nt)]
            n_sig = len(sig)
            pct = 100 * n_sig / total if total > 0 else 0
            mean_r = sig['peak_corr'].mean() if n_sig > 0 else 0
            max_r = sig['peak_corr'].max() if n_sig > 0 else 0
            print(f"  {snum:>4} {state:>7} {phase:>11} {nt:>8} "
                  f"{total:>6} {n_sig:>5} {pct:>5.1f}% "
                  f"{mean_r:>+10.6f} {max_r:>+8.6f}")

print(f"\n[DONE] {time.strftime('%Y-%m-%d %H:%M:%S')}")
