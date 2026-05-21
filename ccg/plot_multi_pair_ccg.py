"""
Plot cross-correlograms for multiple significant pairs across network types,
sessions, and metabolic states. 8 panels in a 4x2 grid.
"""

import yaml
from pathlib import Path
import numpy as np
import cupy as cp
import spikeinterface.extractors as se
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import time

warnings.filterwarnings('ignore')

with open("paths.yaml") as f:
    paths_config = yaml.safe_load(f)

FS = 30000
BIN_SIZE_MS = 1
BIN_SAMPLES = int(BIN_SIZE_MS * FS / 1000)
MAX_LAG_MS = 500
LHA_DEPTH_MAX = 1300

sessions_cfg = paths_config["single_probe"]["coordinates_1"]["mouse01"]["sessions"]

# Pairs to plot: (session, unit_a, unit_b, network_type, label)
PAIRS = [
    # RSP-RSP fed — strongest
    (1, 167, 183, 'RSP-RSP', 'S1 fed/exp — r=0.502'),
    # RSP-RSP fasted — strongest
    (6, 183, 363, 'RSP-RSP', 'S6 fasted/for — r=0.553'),
    # LHA-LHA fed — strongest
    (2, 45, 67, 'LHA-LHA', 'S2 fed/for'),
    # LHA-LHA fasted — strongest
    (8, 76, 104, 'LHA-LHA', 'S8 fasted/for — r=0.118'),
    # LHA-RSP fasted S6
    (6, 3, 270, 'LHA-RSP', 'S6 fasted/for — r=0.063'),
    # LHA-RSP fasted S7
    (7, 45, 270, 'LHA-RSP', 'S7 fasted/exp'),
    # LHA-RSP fasted S8 — strongest
    (8, 103, 450, 'LHA-RSP', 'S8 fasted/for — r=0.070'),
    # LHA-RSP fasted S8 — 2nd strongest
    (8, 76, 450, 'LHA-RSP', 'S8 fasted/for — r=0.054'),
]


def compute_pair_ccg(session_num: int, uid_a: int, uid_b: int) -> tuple:
    """Load a session, bin two units, compute CCG on GPU."""
    sc = sessions_cfg[f"session_{session_num}"]
    sp = Path(sc['sorted'])
    sorting = se.read_kilosort(sp)

    st_a = sorting.get_unit_spike_train(uid_a)
    st_b = sorting.get_unit_spike_train(uid_b)

    all_min = min(np.min(st_a), np.min(st_b))
    all_max = max(np.max(st_a), np.max(st_b))
    n_bins = int((all_max - all_min) / BIN_SAMPLES) + 1

    def _bin(st):
        t = np.zeros(n_bins)
        b = ((st - all_min) // BIN_SAMPLES).astype(int)
        b = b[(b >= 0) & (b < n_bins)]
        np.add.at(t, b, 1)
        s = np.std(t)
        if s > 1e-8:
            t = (t - np.mean(t)) / s
        else:
            t = t - np.mean(t)
        return t

    t1 = cp.asarray(_bin(st_a).astype(np.float32))
    t2 = cp.asarray(_bin(st_b).astype(np.float32))

    lags = np.arange(-MAX_LAG_MS, MAX_LAG_MS + 1)
    ccg = np.empty(len(lags), dtype=np.float64)

    for i, lag in enumerate(lags):
        if lag == 0:
            ccg[i] = float(cp.dot(t1, t2) / n_bins)
        elif lag > 0:
            ccg[i] = float(cp.dot(t1[lag:], t2[:-lag]) / (n_bins - lag))
        else:
            alag = -lag
            ccg[i] = float(cp.dot(t1[:-alag], t2[alag:]) / (n_bins - alag))

    return lags, ccg


# Check LHA-LHA fed pair exists
print("Checking LHA-LHA fed pairs...")
sig = pd.read_csv("data/coor1_significant_pairs.csv")
lha_fed = sig[(sig['network'] == 'LHA-LHA') & (sig['session'].isin([1,2,3,4]))].sort_values('peak_corr', ascending=False)
print(f"Top LHA-LHA fed pairs:")
print(lha_fed.head(5)[['session','unit_a','unit_b','peak_corr']].to_string())

# Check if S7 LHA-RSP pair 45-270 exists
lha_rsp_s7 = sig[(sig['network'] == 'LHA-RSP') & (sig['session'] == 7)].sort_values('peak_corr', ascending=False)
print(f"\nTop LHA-RSP S7 pairs:")
print(lha_rsp_s7.head(5)[['session','unit_a','unit_b','peak_corr']].to_string())

# Update pair 2 (LHA-LHA fed) with actual top pair
top_lha_fed = lha_fed.iloc[0]
PAIRS[2] = (int(top_lha_fed['session']), int(top_lha_fed['unit_a']),
            int(top_lha_fed['unit_b']), 'LHA-LHA',
            f"S{int(top_lha_fed['session'])} fed — r={top_lha_fed['peak_corr']:.3f}")

# Update LHA-RSP S7 with actual top pair
top_s7 = lha_rsp_s7.iloc[0]
PAIRS[5] = (int(top_s7['session']), int(top_s7['unit_a']),
            int(top_s7['unit_b']), 'LHA-RSP',
            f"S7 fasted/exp — r={top_s7['peak_corr']:.3f}")

# --- Compute all CCGs ---
print("\nComputing CCGs...")
ccg_data = []
for session_num, uid_a, uid_b, nt, label in PAIRS:
    t0 = time.time()
    lags, ccg = compute_pair_ccg(session_num, uid_a, uid_b)
    peak_idx = np.argmax(np.abs(ccg))
    peak_lag = lags[peak_idx]
    peak_val = ccg[peak_idx]
    print(f"  {nt} units {uid_a}-{uid_b} ({label}): peak r={peak_val:.4f} "
          f"at lag={peak_lag}ms [{time.time()-t0:.1f}s]")
    ccg_data.append((lags, ccg, nt, uid_a, uid_b, label, peak_val, peak_lag))
    cp.get_default_memory_pool().free_all_blocks()

# --- Plot 4x2 grid ---
COLORS = {
    'RSP-RSP': '#27ae60',
    'LHA-LHA': '#e74c3c',
    'LHA-RSP': '#2980b9',
}

fig, axes = plt.subplots(4, 2, figsize=(16, 18))

for idx, (lags, ccg, nt, uid_a, uid_b, label, peak_val, peak_lag) in enumerate(ccg_data):
    row = idx // 2
    col = idx % 2
    ax = axes[row, col]
    color = COLORS[nt]

    # Full CCG as line
    ax.plot(lags, ccg, color=color, linewidth=0.6, alpha=0.9)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.4, linewidth=0.5)
    ax.axhline(0, color='gray', linestyle='-', alpha=0.3, linewidth=0.5)

    # Zoom inset for +/- 25ms
    inset = ax.inset_axes([0.62, 0.55, 0.35, 0.40])
    zoom = (lags >= -25) & (lags <= 25)
    inset.bar(lags[zoom], ccg[zoom], width=1, color=color, edgecolor='none', alpha=0.8)
    inset.axvline(0, color='gray', linestyle='--', alpha=0.4, linewidth=0.5)
    inset.axhline(0, color='gray', linestyle='-', alpha=0.3, linewidth=0.5)
    inset.set_xlim(-25, 25)
    inset.set_title('±25ms', fontsize=8)
    inset.tick_params(labelsize=7)

    ax.set_title(f'{nt}:  units {uid_a} × {uid_b}\n{label}  |  peak r={peak_val:.4f} at {peak_lag}ms',
                 fontsize=11, fontweight='bold')
    ax.set_xlim(-500, 500)
    ax.set_xlabel('Lag (ms)', fontsize=10)
    ax.set_ylabel('Cross-correlation', fontsize=10)

fig.suptitle('Coor1 Mouse01 — Individual Pair Cross-Correlograms (±500ms)',
             fontsize=15, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.savefig('figures/coor1_multi_pair_ccg.png', dpi=200, bbox_inches='tight')
print("\nSaved to figures/coor1_multi_pair_ccg.png")
plt.close()
