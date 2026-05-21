"""
Find significant pairwise co-occurrence within and between LHA/RSP
for Coordinates-2 Mouse01, all 6 sessions.
Uses GPU matrix multiplication for all-pairs correlations and
circular-shift significance testing (500 shuffles, FDR correction).
"""

import yaml
from pathlib import Path
import numpy as np
import pandas as pd
import cupy as cp
import spikeinterface.extractors as se
import warnings
import time

warnings.filterwarnings('ignore')

with open("paths.yaml") as f:
    paths_config = yaml.safe_load(f)

LHA_DEPTH_MAX = 1410
RSP_DEPTH_MIN = 4725
FS = 30000
BIN_SIZE_MS = 1
BIN_SAMPLES = int(BIN_SIZE_MS * FS / 1000)
N_SHUFFLES = 500
PEAK_WINDOW = 5  # +/- 5ms for peak detection
MIN_SHIFT_BINS = 1000  # 1 second at 1ms bins


def get_good_units_by_region(sorted_path_obj):
    ci = sorted_path_obj / "cluster_info.tsv"
    if not ci.exists():
        return np.array([]), np.array([])
    df = pd.read_csv(ci, sep='\t')
    label_col = 'KSLabel' if 'KSLabel' in df.columns else None
    if label_col is None:
        return np.array([]), np.array([])
    good = df[df[label_col] == 'good']
    lha_ids = good[good['depth'] <= LHA_DEPTH_MAX]['cluster_id'].values
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
    """Compute all-pairs peak correlation in +/- peak_window ms using matrix multiply.
    Returns peak values (with sign) and peak lags."""
    n_units = X_gpu.shape[0]
    n_lags = 2 * peak_window + 1

    # Correlation at each lag via matrix multiply
    all_corr = cp.zeros((n_lags, n_units, n_units), dtype=cp.float32)
    for i, lag in enumerate(range(-peak_window, peak_window + 1)):
        if lag == 0:
            all_corr[i] = cp.dot(X_gpu, X_gpu.T) / n_bins
        elif lag > 0:
            all_corr[i] = cp.dot(X_gpu[:, lag:], X_gpu[:, :-lag].T) / (n_bins - lag)
        else:
            alag = -lag
            all_corr[i] = cp.dot(X_gpu[:, :-alag], X_gpu[:, alag:].T) / (n_bins - alag)

    # Find lag with max |correlation| for each pair
    abs_corr = cp.abs(all_corr)
    peak_lag_idx = cp.argmax(abs_corr, axis=0)
    rows = cp.arange(n_units)[:, None]
    cols = cp.arange(n_units)[None, :]
    peak_vals = all_corr[peak_lag_idx, rows, cols]
    peak_lags = peak_lag_idx - peak_window

    return peak_vals, peak_lags


def fdr_bh(p_values, alpha=0.05):
    """Benjamini-Hochberg FDR correction. Returns boolean mask of significant entries."""
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
sessions = paths_config["single_probe"]["coordinates_2"]["mouse01"]["sessions"]
session_meta = {1: 'exploration', 2: 'foraging', 3: 'exploration',
                4: 'foraging', 5: 'exploration', 6: 'foraging'}

all_pairs_list = []

for sname, snum in [('session_1', 1), ('session_2', 2), ('session_3', 3),
                     ('session_4', 4), ('session_5', 5), ('session_6', 6)]:
    sc = sessions[sname]
    sp = Path(sc['sorted'])
    phase = session_meta[snum]

    lha_ids, rsp_ids = get_good_units_by_region(sp)
    print(f"\nSession {snum} ({phase}): LHA={len(lha_ids)}, RSP={len(rsp_ids)}")

    sorting = se.read_kilosort(sp)
    avail = set(sorting.get_unit_ids())
    lha_ids = np.array([u for u in lha_ids if u in avail])
    rsp_ids = np.array([u for u in rsp_ids if u in avail])

    all_ids = np.concatenate([lha_ids, rsp_ids])
    n_lha = len(lha_ids)
    n_rsp = len(rsp_ids)
    n_units = len(all_ids)

    t0 = time.time()
    binned, n_bins = prebin_spike_trains(sorting, all_ids)

    # Build matrix on GPU (float32 for speed)
    X = np.zeros((n_units, n_bins), dtype=np.float32)
    for i, uid in enumerate(all_ids):
        X[i] = binned[uid].astype(np.float32)
    X_gpu = cp.asarray(X)
    del X
    print(f"  Binned & transferred to GPU in {time.time()-t0:.1f}s")

    # Compute observed peak correlations
    t1 = time.time()
    obs_peak_vals, obs_peak_lags = compute_peak_matrix(X_gpu, n_bins, PEAK_WINDOW)
    obs_abs = cp.abs(obs_peak_vals)
    print(f"  Observed peaks computed in {time.time()-t1:.1f}s")

    # Shuffle significance test
    t2 = time.time()
    exceed_count = cp.zeros((n_units, n_units), dtype=cp.int32)
    X_shuffled = cp.empty_like(X_gpu)

    for s in range(N_SHUFFLES):
        shifts = np.random.randint(MIN_SHIFT_BINS, n_bins - MIN_SHIFT_BINS, size=n_units)
        for i in range(n_units):
            X_shuffled[i] = cp.roll(X_gpu[i], int(shifts[i]))

        shuf_peak_vals, _ = compute_peak_matrix(X_shuffled, n_bins, PEAK_WINDOW)
        shuf_abs = cp.abs(shuf_peak_vals)
        exceed_count += (shuf_abs >= obs_abs).astype(cp.int32)

        if (s + 1) % 100 == 0:
            print(f"    Shuffle {s+1}/{N_SHUFFLES}...", flush=True)

    p_values_gpu = (exceed_count.astype(cp.float32) + 1) / (N_SHUFFLES + 1)
    p_values = cp.asnumpy(p_values_gpu)
    obs_peak_vals_cpu = cp.asnumpy(obs_peak_vals)
    obs_peak_lags_cpu = cp.asnumpy(obs_peak_lags)
    print(f"  Shuffle test done in {time.time()-t2:.1f}s")

    # Collect all unique pairs
    pairs = []
    for i in range(n_units):
        for j in range(i + 1, n_units):
            i_is_lha = i < n_lha
            j_is_lha = j < n_lha
            if i_is_lha and j_is_lha:
                nt = 'LHA-LHA'
            elif not i_is_lha and not j_is_lha:
                nt = 'RSP-RSP'
            else:
                nt = 'LHA-RSP'

            pairs.append({
                'session': snum,
                'phase': phase,
                'network': nt,
                'unit_a': int(all_ids[i]),
                'unit_b': int(all_ids[j]),
                'peak_lag_ms': int(obs_peak_lags_cpu[i, j]),
                'peak_corr': float(obs_peak_vals_cpu[i, j]),
                'p_value': float(p_values[i, j]),
            })

    pairs_df = pd.DataFrame(pairs)

    # FDR correction per network type
    for nt in ['LHA-LHA', 'RSP-RSP', 'LHA-RSP']:
        mask = pairs_df['network'] == nt
        sub_p = pairs_df.loc[mask, 'p_value'].values
        sig = fdr_bh(sub_p, alpha=0.05)
        pairs_df.loc[mask, 'significant'] = sig

    sig_df = pairs_df[pairs_df['significant'] == True]
    n_lha_lha = len(sig_df[sig_df['network'] == 'LHA-LHA'])
    n_rsp_rsp = len(sig_df[sig_df['network'] == 'RSP-RSP'])
    n_lha_rsp = len(sig_df[sig_df['network'] == 'LHA-RSP'])
    total_pairs = len(pairs_df)
    print(f"  Significant (FDR<0.05): LHA-LHA={n_lha_lha}, RSP-RSP={n_rsp_rsp}, "
          f"LHA-RSP={n_lha_rsp} out of {total_pairs} total pairs")

    all_pairs_list.append(pairs_df)

    # Free GPU memory
    del X_gpu, X_shuffled
    cp.get_default_memory_pool().free_all_blocks()

# Combine and save all pairs (with significance flag)
result = pd.concat(all_pairs_list, ignore_index=True)
result.to_csv("data/coor2_all_pairs_significance.csv", index=False)

sig_result = result[result['significant'] == True].copy()
sig_result.to_csv("data/coor2_significant_pairs.csv", index=False)

# Print summary
print("\n" + "=" * 90)
print("SIGNIFICANT CO-OCCURRENCE PAIRS BY SESSION (FDR < 0.05)")
print("=" * 90)

for snum in range(1, 7):
    phase = session_meta[snum]
    s_df = sig_result[sig_result['session'] == snum]
    print(f"\n--- Session {snum} ({phase}): {len(s_df)} significant pairs ---")

    for nt in ['LHA-LHA', 'RSP-RSP', 'LHA-RSP']:
        nt_df = s_df[s_df['network'] == nt].sort_values('peak_corr', ascending=False)
        total_nt = len(result[(result['session'] == snum) & (result['network'] == nt)])
        if len(nt_df) > 0:
            print(f"\n  {nt}: {len(nt_df)}/{total_nt} pairs significant "
                  f"({100*len(nt_df)/total_nt:.1f}%)")
            # Show top 10 and bottom 5
            show = nt_df.head(10)
            for _, r in show.iterrows():
                print(f"    Units {int(r['unit_a']):>4} - {int(r['unit_b']):>4}: "
                      f"r={r['peak_corr']:+.6f} at lag={int(r['peak_lag_ms']):+d}ms, "
                      f"p={r['p_value']:.4f}")
            if len(nt_df) > 10:
                print(f"    ... ({len(nt_df) - 10} more pairs)")
        else:
            print(f"\n  {nt}: 0/{total_nt} pairs significant")

# Summary table
print("\n" + "=" * 90)
print("SUMMARY TABLE")
print("=" * 90)
print(f"  {'Session':>7} {'Phase':>11} {'Network':>8} {'Total':>6} {'Sig':>5} {'%':>6} "
      f"{'Mean r(sig)':>11} {'Max r':>9}")

for snum in range(1, 7):
    phase = session_meta[snum]
    for nt in ['LHA-LHA', 'RSP-RSP', 'LHA-RSP']:
        total = len(result[(result['session'] == snum) & (result['network'] == nt)])
        sig = sig_result[(sig_result['session'] == snum) & (sig_result['network'] == nt)]
        n_sig = len(sig)
        pct = 100 * n_sig / total if total > 0 else 0
        mean_r = sig['peak_corr'].mean() if n_sig > 0 else 0
        max_r = sig['peak_corr'].max() if n_sig > 0 else 0
        print(f"  {snum:>7} {phase:>11} {nt:>8} {total:>6} {n_sig:>5} {pct:>5.1f}% "
              f"{mean_r:>+10.6f} {max_r:>+8.6f}")

print("\n[DONE]")
