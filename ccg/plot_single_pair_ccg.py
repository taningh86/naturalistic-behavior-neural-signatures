"""
Plot the full cross-correlogram (-500 to +500ms) for a single LHA-RSP pair.
Uses GPU dot products for speed.
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

# --- Config ---
SESSION_NUM = 8
UNIT_A = 103  # LHA
UNIT_B = 450  # RSP
FS = 30000
BIN_SIZE_MS = 1
BIN_SAMPLES = int(BIN_SIZE_MS * FS / 1000)
MAX_LAG_MS = 500

# Load session
sessions = paths_config["single_probe"]["coordinates_1"]["mouse01"]["sessions"]
sc = sessions[f"session_{SESSION_NUM}"]
sp = Path(sc['sorted'])
state = sc['state']
phase = sc['phase']

sorting = se.read_kilosort(sp)
print(f"Session {SESSION_NUM} ({state}/{phase}), units {UNIT_A} (LHA) & {UNIT_B} (RSP)")

# Get spike trains and bin
st_a = sorting.get_unit_spike_train(UNIT_A)
st_b = sorting.get_unit_spike_train(UNIT_B)

all_min = min(np.min(st_a), np.min(st_b))
all_max = max(np.max(st_a), np.max(st_b))
n_bins = int((all_max - all_min) / BIN_SAMPLES) + 1

def bin_and_zscore(st):
    t = np.zeros(n_bins)
    b = ((st - all_min) // BIN_SAMPLES).astype(int)
    b = b[(b >= 0) & (b < n_bins)]
    np.add.at(t, b, 1)
    std_val = np.std(t)
    if std_val > 1e-8:
        t = (t - np.mean(t)) / std_val
    else:
        t = t - np.mean(t)
    return t

t1 = cp.asarray(bin_and_zscore(st_a).astype(np.float32))
t2 = cp.asarray(bin_and_zscore(st_b).astype(np.float32))

# Compute CCG at each lag
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

peak_idx = np.argmax(np.abs(ccg))
peak_lag = lags[peak_idx]
peak_val = ccg[peak_idx]

print(f"Peak: r={peak_val:.6f} at lag={peak_lag}ms")
print(f"lag0={ccg[lags==0][0]:.6f}")

# --- Plot ---
fig, axes = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})

# Full CCG
ax = axes[0]
ax.plot(lags, ccg, color='#2c3e50', linewidth=0.8)
ax.axvline(0, color='red', linestyle='--', alpha=0.5, linewidth=0.7)
ax.axhline(0, color='gray', linestyle='-', alpha=0.3, linewidth=0.5)
ax.set_xlabel('Lag (ms)', fontsize=12)
ax.set_ylabel('Cross-correlation (z-scored)', fontsize=12)
ax.set_title(
    f'Cross-Correlogram: LHA unit {UNIT_A} × RSP unit {UNIT_B}\n'
    f'Session {SESSION_NUM} ({state}/{phase})  |  Peak r={peak_val:.4f} at lag={peak_lag}ms',
    fontsize=13, fontweight='bold'
)
ax.set_xlim(-500, 500)

# Zoomed inset: +/- 50ms
ax2 = axes[1]
zoom_mask = (lags >= -50) & (lags <= 50)
ax2.bar(lags[zoom_mask], ccg[zoom_mask], width=1, color='#2980b9', edgecolor='none', alpha=0.85)
ax2.axvline(0, color='red', linestyle='--', alpha=0.5, linewidth=0.7)
ax2.axhline(0, color='gray', linestyle='-', alpha=0.3, linewidth=0.5)
ax2.set_xlabel('Lag (ms)', fontsize=12)
ax2.set_ylabel('Cross-correlation', fontsize=12)
ax2.set_title('Zoom: ±50 ms', fontsize=11)
ax2.set_xlim(-50, 50)

plt.tight_layout()
plt.savefig('figures/coor1_single_pair_ccg_LHA103_RSP450_S8.png', dpi=200, bbox_inches='tight')
print("Saved to figures/coor1_single_pair_ccg_LHA103_RSP450_S8.png")
plt.close()
