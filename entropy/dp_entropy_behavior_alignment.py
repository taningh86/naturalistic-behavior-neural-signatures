"""
Dual-Probe: Entropy Trace Aligned with Manually-Scored Behaviors
================================================================
Plots entropy timeseries with behavior annotations for fed sessions S3 & S4.
Behaviors shown as colored horizontal bars whenever active (value=1).
Entropy peaks/troughs marked with vertical lines.
"""

import yaml
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import entropy as sp_entropy
from scipy.ndimage import gaussian_filter1d
from scipy.signal import argrelextrema
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

with open("paths.yaml") as f:
    cfg = yaml.safe_load(f)

# ---- Constants ----
ENTROPY_WINDOW_SEC = 60
ENTROPY_STEP_SEC = 10
SMOOTH_SIGMA = 3
MIN_AMPLITUDE = 0.3

# Sessions to plot — all 18 (S23/S24 excluded, different paradigm)
SKIP_SESSIONS = {23, 24}
TARGET_SESSIONS = []  # populated below from paths.yaml

# Behaviors to include (column names in xlsx)
BEHAVIOR_COLS = [
    'Feeding',
    'Digging sand',
    'Transition wall exploration',
    'Hiding in corners',
    'Quick one loop at home',
    'Incomplete home returns',
    'Contemplation at T-zone looking at Home',
    'Rearing',
    'Hiding food at home',
]

# Short labels for display
BEHAVIOR_LABELS = [
    'Feeding',
    'Digging sand',
    'Transition wall expl.',
    'Hiding in corners',
    'Quick loop at home',
    'Incomplete home ret.',
    'Contemplation at T-zone',
    'Rearing',
    'Hiding food at home',
]

# Distinct colors for each behavior
BEHAVIOR_COLORS = [
    '#d62728',   # red - Feeding
    '#8c564b',   # brown - Digging sand
    '#2ca02c',   # green - Transition wall exploration
    '#9467bd',   # purple - Hiding in corners
    '#ff7f0e',   # orange - Quick one loop at home
    '#1f77b4',   # blue - Incomplete home returns
    '#e377c2',   # pink - Contemplation at T-zone
    '#17becf',   # cyan - Rearing
    '#bcbd22',   # yellow-green - Hiding food at home
]

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

sessions_cfg = cfg["double_probe"]["coordinates_1"]["mouse01"]["sessions"]


def load_behavior_xlsx(path):
    """Load dual-probe behavior xlsx (header at row 34, data from row 36)."""
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

    # Load behavior columns
    behaviors = {}
    for bcol in BEHAVIOR_COLS:
        if bcol in col_names:
            bvals = pd.to_numeric(data[bcol], errors='coerce').values
            bvals = np.nan_to_num(bvals, nan=0.0)
            behaviors[bcol] = bvals
        else:
            print(f"  WARNING: behavior column '{bcol}' not found")
            behaviors[bcol] = np.zeros(len(time_vals))

    return time_vals, vel, zones, behaviors


def compute_entropy_causal(zones, time_vals, vel, window_sec, step_sec):
    """Compute behavioral entropy (causal, end-assigned)."""
    dt = np.median(np.diff(time_vals))
    window_bins = int(window_sec / dt)
    step_bins = int(step_sec / dt)

    ent_times, ent_vals, vel_means = [], [], []
    for start_idx in range(0, len(zones) - window_bins, step_bins):
        wz = zones[start_idx:start_idx + window_bins]
        transitions = []
        for j in range(1, len(wz)):
            if wz[j] != wz[j - 1]:
                transitions.append(f"{wz[j-1]}->{wz[j]}")
        if len(transitions) < 3:
            continue
        counts = Counter(transitions)
        probs = np.array(list(counts.values()), dtype=float)
        probs /= probs.sum()
        h = sp_entropy(probs, base=2)
        ent_times.append(time_vals[start_idx + window_bins - 1])
        ent_vals.append(h)
        vel_means.append(np.nanmean(vel[start_idx:start_idx + window_bins]))

    return np.array(ent_times), np.array(ent_vals), np.array(vel_means)


def find_inflections(values, smooth_sigma, min_amp, order=3):
    smoothed = gaussian_filter1d(values, smooth_sigma)
    peaks = list(argrelextrema(smoothed, np.greater, order=order)[0])
    troughs = list(argrelextrema(smoothed, np.less, order=order)[0])
    all_ext = sorted([(p, 'peak', smoothed[p]) for p in peaks] +
                     [(t, 'trough', smoothed[t]) for t in troughs],
                     key=lambda x: x[0])
    if len(all_ext) < 2:
        return [], [], smoothed
    filtered = [all_ext[0]]
    for i in range(1, len(all_ext)):
        if all_ext[i][1] == filtered[-1][1]:
            if all_ext[i][1] == 'peak':
                if all_ext[i][2] > filtered[-1][2]:
                    filtered[-1] = all_ext[i]
            else:
                if all_ext[i][2] < filtered[-1][2]:
                    filtered[-1] = all_ext[i]
        else:
            amp = abs(all_ext[i][2] - filtered[-1][2])
            if amp >= min_amp:
                filtered.append(all_ext[i])
    final_peaks = [f[0] for f in filtered if f[1] == 'peak']
    final_troughs = [f[0] for f in filtered if f[1] == 'trough']
    return final_peaks, final_troughs, smoothed


def downsample_behavior(bvals, time_vals, ent_times, window_sec=10):
    """Downsample behavior to entropy time resolution.
    For each entropy timepoint, compute fraction of time behavior was active
    in the preceding window_sec seconds."""
    dt = np.median(np.diff(time_vals))
    win_bins = int(window_sec / dt)
    fracs = np.zeros(len(ent_times))
    for i, et in enumerate(ent_times):
        idx = np.searchsorted(time_vals, et)
        start = max(0, idx - win_bins)
        end = min(len(bvals), idx + 1)
        if end > start:
            fracs[i] = np.mean(bvals[start:end])
    return fracs


# ============================================================
# Discover all valid sessions
# ============================================================
for skey, sval in sessions_cfg.items():
    snum = int(skey.split('_')[1])
    if snum in SKIP_SESSIONS:
        continue
    behav = sval.get('behavior')
    if not behav:
        continue
    if not Path(behav).exists():
        continue
    TARGET_SESSIONS.append(snum)
TARGET_SESSIONS.sort()
print(f"Will plot {len(TARGET_SESSIONS)} sessions: {TARGET_SESSIONS}")

# ============================================================
# PLOT — one figure per session
# ============================================================
out_dir = Path('figures/entropy_behavior_alignment')

for snum in TARGET_SESSIONS:
    skey = f"session_{snum}"
    sval = sessions_cfg[skey]
    behav_path = sval['behavior']
    state = sval['state']
    phase = sval['phase']

    print(f"S{snum} ({state}/{phase}): loading behavior...", end=' ', flush=True)
    time_vals, vel, zones, behaviors = load_behavior_xlsx(behav_path)

    # Compute entropy
    ent_times, ent_vals, vel_means = compute_entropy_causal(
        zones, time_vals, vel, ENTROPY_WINDOW_SEC, ENTROPY_STEP_SEC)
    print(f"entropy={len(ent_times)} pts", flush=True)

    # Find inflections
    peaks_idx, troughs_idx, smoothed = find_inflections(
        ent_vals, SMOOTH_SIGMA, MIN_AMPLITUDE)

    # Convert entropy times to minutes for display
    ent_mins = ent_times / 60.0

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(32, 12))
    fig.subplots_adjust(left=0.06, right=0.75)

    # Plot velocity on secondary y-axis (orange)
    ax2 = ax.twinx()
    ax2.plot(ent_mins, vel_means, color='#ff8c00', linewidth=1.8, alpha=0.5,
             label='Velocity (cm/s)')
    vel_smoothed = gaussian_filter1d(vel_means, SMOOTH_SIGMA)
    ax2.plot(ent_mins, vel_smoothed, color='#ff8c00', linewidth=2.5, alpha=0.85,
             label='Velocity (smoothed)')
    ax2.set_ylabel('Velocity (cm/s)', fontsize=14, fontweight='bold', color='#ff8c00')
    ax2.tick_params(axis='y', labelsize=12, colors='#ff8c00')
    ax2.spines['right'].set_color('#ff8c00')
    ax2.spines['top'].set_visible(False)

    # Plot entropy trace
    ax.plot(ent_mins, ent_vals, color='black', linewidth=1.5, alpha=0.5, label='Entropy (raw)')
    ax.plot(ent_mins, smoothed, color='black', linewidth=2.5, label='Entropy (smoothed)')

    # Mark peaks and troughs
    if peaks_idx:
        ax.scatter(ent_mins[peaks_idx], smoothed[peaks_idx],
                   color='red', s=120, zorder=5, marker='^', edgecolors='black',
                   linewidths=1.0, label=f'Peaks (n={len(peaks_idx)})')
    if troughs_idx:
        ax.scatter(ent_mins[troughs_idx], smoothed[troughs_idx],
                   color='blue', s=120, zorder=5, marker='v', edgecolors='black',
                   linewidths=1.0, label=f'Troughs (n={len(troughs_idx)})')

    # Get entropy y-range for behavior bar placement
    ent_min = np.nanmin(ent_vals)
    ent_max = np.nanmax(ent_vals)
    ent_range = ent_max - ent_min

    # Place behavior bars below the entropy trace
    bar_height = ent_range * 0.065
    gap = ent_range * 0.025
    n_behaviors = len(BEHAVIOR_COLS)
    bottom_start = ent_min - gap - n_behaviors * (bar_height + gap)

    for b_idx, bcol in enumerate(BEHAVIOR_COLS):
        bvals = behaviors[bcol]
        bfrac = downsample_behavior(bvals, time_vals, ent_times)

        y_base = bottom_start + (n_behaviors - 1 - b_idx) * (bar_height + gap)
        color = BEHAVIOR_COLORS[b_idx]
        label = BEHAVIOR_LABELS[b_idx]

        # Draw filled bars where behavior is active (fraction > 0.05)
        active = bfrac > 0.05
        if np.any(active):
            diff = np.diff(active.astype(int))
            starts = np.where(diff == 1)[0] + 1
            ends = np.where(diff == -1)[0] + 1
            if active[0]:
                starts = np.concatenate([[0], starts])
            if active[-1]:
                ends = np.concatenate([ends, [len(active)]])
            for s, e in zip(starts, ends):
                ax.fill_between(ent_mins[s:e], y_base, y_base + bar_height,
                                color=color, alpha=0.85, linewidth=0)
        ax.axhline(y=y_base, color=color, linewidth=0.3, alpha=0.3)
        ax.text(ent_mins[-1] + 0.3, y_base + bar_height / 2, label,
                fontsize=17, fontweight='bold', color=color,
                ha='left', va='center')

    # Separator line between entropy and behavior lanes
    sep_y = ent_min - gap * 0.5
    ax.axhline(y=sep_y, color='gray', linewidth=1.0, linestyle='-', alpha=0.5)

    # Vertical dashed lines at peaks/troughs
    for pidx in peaks_idx:
        ax.axvline(x=ent_mins[pidx], color='red', linestyle='--',
                   alpha=0.3, linewidth=1.0)
    for tidx in troughs_idx:
        ax.axvline(x=ent_mins[tidx], color='blue', linestyle='--',
                   alpha=0.3, linewidth=1.0)

    # Formatting
    ax.set_ylim(bottom_start - gap, ent_max + ent_range * 0.15)
    ax.set_xlabel('Time (min)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Behavioral Entropy (bits)', fontsize=14, fontweight='bold')
    ax.set_title(f'S{snum} — {state.capitalize()} / {phase.capitalize()}',
                 fontsize=18, fontweight='bold')
    ax.tick_params(labelsize=12)

    # Combined legend outside plot (entropy + velocity)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1[:4] + h2, l1[:4] + l2, loc='upper left',
              bbox_to_anchor=(1.01, 1.0), fontsize=14,
              framealpha=0.9, edgecolor='black', borderpad=0.8)

    ax.spines['top'].set_visible(False)

    fname = out_dir / f'S{snum}_{state}_{phase}_entropy_behavior.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> Saved {fname}")

print(f"\nDone. {len(TARGET_SESSIONS)} figures saved to {out_dir}/")
