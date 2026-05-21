"""
Full cross-correlogram analysis for Coordinates-2 Mouse01.
Compute mean CCG at 1ms resolution from -500ms to +500ms
for LHA-LHA, RSP-RSP, and LHA-RSP across all 6 sessions.
Uses CuPy for GPU-accelerated dot products.
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

MAX_LAG_MS = 500
LAGS = np.arange(-MAX_LAG_MS, MAX_LAG_MS + 1)  # -500 to +500


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


def fast_ccg_gpu(t1_gpu, t2_gpu, n_bins, max_lag):
    """Compute CCG from -max_lag to +max_lag using CuPy dot products on GPU."""
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
    """Compute population-average CCG on GPU. For same_region, upper triangle only."""
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


# =============================================================================
sessions = paths_config["single_probe"]["coordinates_2"]["mouse01"]["sessions"]
session_meta = {1: 'exploration', 2: 'foraging', 3: 'exploration',
                4: 'foraging', 5: 'exploration', 6: 'foraging'}

all_results = []
all_ccgs = {}  # store full CCGs for later

for sname, snum in [('session_1',1),('session_2',2),('session_3',3),
                     ('session_4',4),('session_5',5),('session_6',6)]:
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
    t0 = time.time()
    binned, n_bins = prebin_spike_trains(sorting, all_ids)
    print(f"  Binned {len(all_ids)} units into {n_bins} bins in {time.time()-t0:.1f}s")

    # Transfer to GPU
    binned_gpu = {uid: cp.asarray(binned[uid]) for uid in all_ids}
    print(f"  Transferred to GPU")

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

        # Find peak excluding lag=0 for same-region
        if same:
            search_ccg = ccg.copy()
            search_ccg[LAGS == 0] = 0
        else:
            search_ccg = ccg

        peak_idx = np.argmax(np.abs(search_ccg))
        peak_lag = LAGS[peak_idx]
        peak_val = search_ccg[peak_idx]

        # Values at key lags (mean of +/- for symmetric view)
        zero_val = ccg[LAGS == 0][0]
        val_1 = np.mean([ccg[LAGS == 1][0], ccg[LAGS == -1][0]])
        val_2 = np.mean([ccg[LAGS == 2][0], ccg[LAGS == -2][0]])
        val_5 = np.mean([ccg[LAGS == 5][0], ccg[LAGS == -5][0]])
        val_10 = np.mean([ccg[LAGS == 10][0], ccg[LAGS == -10][0]])
        val_25 = np.mean([ccg[LAGS == 25][0], ccg[LAGS == -25][0]])
        val_50 = np.mean([ccg[LAGS == 50][0], ccg[LAGS == -50][0]])
        val_100 = np.mean([ccg[LAGS == 100][0], ccg[LAGS == -100][0]])
        val_200 = np.mean([ccg[LAGS == 200][0], ccg[LAGS == -200][0]])
        val_300 = np.mean([ccg[LAGS == 300][0], ccg[LAGS == -300][0]])
        val_500 = np.mean([ccg[LAGS == 500][0], ccg[LAGS == -500][0]])

        # HWHM
        abs_ccg = np.abs(ccg)
        half_max = np.max(abs_ccg) / 2
        above_half = np.where(abs_ccg >= half_max)[0]
        hwhm = (LAGS[above_half[-1]] - LAGS[above_half[0]]) / 2 if len(above_half) > 1 else 0

        print(f"  {nt}: {n_pairs} pairs in {elapsed:.1f}s")
        print(f"    Peak lag: {peak_lag}ms (r={peak_val:.7f})")
        print(f"    lag0={zero_val:.7f}  +/-1ms={val_1:.7f}  +/-2ms={val_2:.7f}  "
              f"+/-5ms={val_5:.7f}  +/-10ms={val_10:.7f}")
        print(f"    +/-25ms={val_25:.7f}  +/-50ms={val_50:.7f}  +/-100ms={val_100:.7f}")
        print(f"    +/-200ms={val_200:.7f}  +/-300ms={val_300:.7f}  +/-500ms={val_500:.7f}  "
              f"HWHM={hwhm:.0f}ms")

        all_results.append({
            'session': snum, 'phase': phase, 'network': nt, 'n_pairs': n_pairs,
            'peak_lag_ms': peak_lag, 'peak_value': peak_val,
            'lag0': zero_val, 'lag1': val_1, 'lag2': val_2, 'lag5': val_5,
            'lag10': val_10, 'lag25': val_25, 'lag50': val_50, 'lag100': val_100,
            'lag200': val_200, 'lag300': val_300, 'lag500': val_500, 'hwhm_ms': hwhm,
        })

    # Free GPU memory between sessions
    del binned_gpu
    cp.get_default_memory_pool().free_all_blocks()

df = pd.DataFrame(all_results)
df.to_csv("data/coor2_ccg_summary.csv", index=False)

# Save full CCGs
ccg_data = {'lags': LAGS}
for (snum, nt), ccg in all_ccgs.items():
    ccg_data[f's{snum}_{nt}'] = ccg
np.savez("data/coor2_ccg_full.npz", **ccg_data)

print("\n" + "=" * 90)
print("SUMMARY BY NETWORK AND PHASE")
print("=" * 90)

for nt in ['LHA-LHA', 'RSP-RSP', 'LHA-RSP']:
    sub = df[df['network'] == nt]
    print(f"\n  {nt}:")
    print(f"    {'Sess':>4} {'Phase':>11} {'Peak':>6} {'peak_r':>11} {'HWHM':>5} "
          f"{'lag0':>10} {'lag1':>10} {'lag5':>10} {'lag10':>10} "
          f"{'lag50':>10} {'lag100':>10} {'lag200':>10} {'lag300':>10} {'lag500':>10}")
    for _, r in sub.iterrows():
        print(f"    {int(r['session']):>4} {r['phase']:>11} {int(r['peak_lag_ms']):>4}ms "
              f"{r['peak_value']:>10.7f} {r['hwhm_ms']:>4.0f}ms "
              f"{r['lag0']:>10.7f} {r['lag1']:>10.7f} "
              f"{r['lag5']:>10.7f} {r['lag10']:>10.7f} "
              f"{r['lag50']:>10.7f} {r['lag100']:>10.7f} "
              f"{r['lag200']:>10.7f} {r['lag300']:>10.7f} {r['lag500']:>10.7f}")

    for phase in ['exploration', 'foraging']:
        ps = sub[sub['phase'] == phase]
        if len(ps) > 0:
            print(f"    {'AVG':>4} {phase:>11} {'':>6} {'':>11} {'':>5} "
                  f"{ps['lag0'].mean():>10.7f} {ps['lag1'].mean():>10.7f} "
                  f"{ps['lag5'].mean():>10.7f} {ps['lag10'].mean():>10.7f} "
                  f"{ps['lag50'].mean():>10.7f} {ps['lag100'].mean():>10.7f} "
                  f"{ps['lag200'].mean():>10.7f} {ps['lag300'].mean():>10.7f} "
                  f"{ps['lag500'].mean():>10.7f}")

print("\n[DONE]")
