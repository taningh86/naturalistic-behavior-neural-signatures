"""
Diagnose the lag-0 peak in CCGs.
1) Compute CCG at 0.1ms resolution (sub-millisecond) to see true peak lag
2) Compute raw coincidence histogram (not z-scored)
3) Compare a strong RSP-RSP pair and an LHA-RSP pair
"""

import yaml
from pathlib import Path
import numpy as np
import cupy as cp
import spikeinterface.extractors as se
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

with open("paths.yaml") as f:
    paths_config = yaml.safe_load(f)

FS = 30000  # 30kHz = 0.0333ms per sample
sessions_cfg = paths_config["single_probe"]["coordinates_1"]["mouse01"]["sessions"]

# LHA-RSP pairs only
PAIRS = [
    (6, 3, 270, 'LHA-RSP'),     # S6 fasted/for, r=0.063
    (8, 103, 450, 'LHA-RSP'),   # S8 fasted/for, r=0.070
]

fig, axes = plt.subplots(len(PAIRS), 3, figsize=(18, 5 * len(PAIRS)))

for row, (snum, uid_a, uid_b, nt) in enumerate(PAIRS):
    sc = sessions_cfg[f"session_{snum}"]
    sp = Path(sc['sorted'])
    state, phase = sc['state'], sc['phase']
    sorting = se.read_kilosort(sp)

    # Get raw spike times in SAMPLES (30kHz)
    st_a = sorting.get_unit_spike_train(uid_a)
    st_b = sorting.get_unit_spike_train(uid_b)

    print(f"\n{'='*70}")
    print(f"{nt}: units {uid_a} & {uid_b}, S{snum} ({state}/{phase})")
    print(f"  Unit {uid_a}: {len(st_a)} spikes, FR={len(st_a)/(st_a[-1]-st_a[0])*FS:.1f} Hz")
    print(f"  Unit {uid_b}: {len(st_b)} spikes, FR={len(st_b)/(st_b[-1]-st_b[0])*FS:.1f} Hz")

    # =====================================================================
    # Panel 1: Raw spike-time difference histogram (no binning at all)
    # Compute all pairwise time differences within +/- 50ms (1500 samples)
    # =====================================================================
    max_lag_samples = 1500  # 50ms at 30kHz
    diffs = []
    # Use searchsorted for efficiency
    for spike_time in st_a:
        lo = np.searchsorted(st_b, spike_time - max_lag_samples)
        hi = np.searchsorted(st_b, spike_time + max_lag_samples)
        if hi > lo:
            diffs.extend((st_b[lo:hi] - spike_time).tolist())

    diffs = np.array(diffs, dtype=np.float64) / FS * 1000  # convert to ms
    print(f"  Pairwise diffs computed: {len(diffs)} within +/-50ms")

    ax1 = axes[row, 0]
    ax1.hist(diffs, bins=np.arange(-50, 50.1, 0.1), color='steelblue',
             edgecolor='none', alpha=0.8)
    ax1.axvline(0, color='red', linestyle='--', alpha=0.6)
    ax1.set_xlabel('Time difference (ms)')
    ax1.set_ylabel('Count')
    ax1.set_title(f'{nt} {uid_a}×{uid_b} S{snum}\nRaw spike diffs (0.1ms bins, ±50ms)')
    ax1.set_xlim(-50, 50)

    # =====================================================================
    # Panel 2: Zoom into +/- 5ms at 0.033ms resolution (1 sample)
    # =====================================================================
    close_diffs = diffs[(diffs >= -5) & (diffs <= 5)]
    ax2 = axes[row, 1]
    ax2.hist(close_diffs, bins=np.arange(-5, 5.033, 1/30),  # 1-sample resolution
             color='coral', edgecolor='none', alpha=0.8)
    ax2.axvline(0, color='red', linestyle='--', alpha=0.6)
    ax2.set_xlabel('Time difference (ms)')
    ax2.set_ylabel('Count')
    ax2.set_title(f'Zoom ±5ms (sample-level resolution)')
    ax2.set_xlim(-5, 5)

    # Stats on the +/- 5ms region
    for window in [0.033, 0.1, 0.5, 1.0, 2.0]:
        n_in = np.sum(np.abs(close_diffs) <= window)
        print(f"  Pairs within +/-{window:.3f}ms: {n_in}")

    # =====================================================================
    # Panel 3: CCG with 0.1ms bins (z-scored) to find true peak lag
    # =====================================================================
    BIN_01_SAMPLES = 3  # 0.1ms = 3 samples at 30kHz
    max_lag_01 = 50  # +/- 50ms = 500 bins of 0.1ms each

    all_min = min(np.min(st_a), np.min(st_b))
    all_max = max(np.max(st_a), np.max(st_b))
    n_bins_01 = int((all_max - all_min) / BIN_01_SAMPLES) + 1

    def bin_zscore(st, n_b):
        t = np.zeros(n_b)
        b = ((st - all_min) // BIN_01_SAMPLES).astype(int)
        b = b[(b >= 0) & (b < n_b)]
        np.add.at(t, b, 1)
        s = np.std(t)
        if s > 1e-8:
            t = (t - np.mean(t)) / s
        else:
            t = t - np.mean(t)
        return t

    t1 = cp.asarray(bin_zscore(st_a, n_bins_01).astype(np.float32))
    t2 = cp.asarray(bin_zscore(st_b, n_bins_01).astype(np.float32))

    lags_01 = np.arange(-max_lag_01 * 10, max_lag_01 * 10 + 1)  # in 0.1ms units
    lags_ms = lags_01 * 0.1  # in ms
    ccg = np.empty(len(lags_01), dtype=np.float64)

    for i, lag in enumerate(lags_01):
        if lag == 0:
            ccg[i] = float(cp.dot(t1, t2) / n_bins_01)
        elif lag > 0:
            ccg[i] = float(cp.dot(t1[lag:], t2[:-lag]) / (n_bins_01 - lag))
        else:
            alag = -lag
            ccg[i] = float(cp.dot(t1[:-alag], t2[alag:]) / (n_bins_01 - alag))

    peak_idx = np.argmax(np.abs(ccg))
    peak_lag_ms = lags_ms[peak_idx]
    peak_val = ccg[peak_idx]
    print(f"  0.1ms CCG peak: r={peak_val:.6f} at lag={peak_lag_ms:.1f}ms")

    ax3 = axes[row, 2]
    ax3.plot(lags_ms, ccg, color='#2c3e50', linewidth=0.5)
    ax3.axvline(0, color='red', linestyle='--', alpha=0.5)
    ax3.axvline(peak_lag_ms, color='orange', linestyle='-', alpha=0.7,
                label=f'peak={peak_lag_ms:.1f}ms')
    ax3.set_xlabel('Lag (ms)')
    ax3.set_ylabel('Cross-correlation (z-scored)')
    ax3.set_title(f'CCG at 0.1ms resolution (±50ms)\npeak r={peak_val:.4f} at {peak_lag_ms:.1f}ms')
    ax3.set_xlim(-50, 50)
    ax3.legend(fontsize=9)

    cp.get_default_memory_pool().free_all_blocks()

fig.suptitle('Diagnosing Lag-0 CCG Peak', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('figures/coor1_ccg_lag0_diagnosis.png', dpi=200, bbox_inches='tight')
print("\nSaved to figures/coor1_ccg_lag0_diagnosis.png")
plt.close()
