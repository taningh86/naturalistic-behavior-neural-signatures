"""
Dual-Probe: Zone Visitation During Entropy Decline (Peak → Trough)
==================================================================
For each session, identifies peak→trough intervals in the smoothed entropy
trace, then reports which zones the mouse visits during each decline and
for how long (fraction of time).
"""

import yaml
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import entropy as sp_entropy
from scipy.ndimage import gaussian_filter1d
from scipy.signal import argrelextrema
from collections import Counter, OrderedDict
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
SKIP_SESSIONS = {23, 24}

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

# Display order for zones (logical grouping)
ZONE_ORDER = ['H', 'HCL', 'HCR', 'L', 'T', 'FA', 'CA',
              'P1', 'P2', 'P3', 'P4', 'P1z', 'P2z', 'P3z', 'P4z', 'O']
ZONE_COLORS = {
    'H': '#4e79a7', 'HCL': '#7bafd4', 'HCR': '#a6cee3',
    'L': '#f28e2b', 'T': '#ffbe7d',
    'FA': '#59a14f', 'CA': '#8cd17d',
    'P1': '#e15759', 'P2': '#ff9d9a', 'P3': '#b07aa1', 'P4': '#d4a6c8',
    'P1z': '#e15759', 'P2z': '#ff9d9a', 'P3z': '#b07aa1', 'P4z': '#d4a6c8',
    'O': '#bab0ac',
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

    return time_vals, vel, zones


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


def get_zone_fractions(zones, time_vals, t_start, t_end):
    """Get fraction of time spent in each zone between t_start and t_end."""
    mask = (time_vals >= t_start) & (time_vals <= t_end)
    segment = zones[mask]
    if len(segment) == 0:
        return {}
    counts = Counter(segment)
    total = len(segment)
    fracs = {z: counts.get(z, 0) / total for z in ZONE_ORDER if counts.get(z, 0) > 0}
    return fracs


def get_zone_sequence(zones, time_vals, t_start, t_end):
    """Get sequence of zone visits (transitions) between t_start and t_end."""
    mask = (time_vals >= t_start) & (time_vals <= t_end)
    segment = zones[mask]
    if len(segment) == 0:
        return []
    # Compress to transitions
    visits = [segment[0]]
    for z in segment[1:]:
        if z != visits[-1]:
            visits.append(z)
    return visits


# ============================================================
# Discover sessions
# ============================================================
target_sessions = []
for skey, sval in sessions_cfg.items():
    snum = int(skey.split('_')[1])
    if snum in SKIP_SESSIONS:
        continue
    behav = sval.get('behavior')
    if not behav or not Path(behav).exists():
        continue
    target_sessions.append(snum)
target_sessions.sort()
print(f"Processing {len(target_sessions)} sessions: {target_sessions}")

# ============================================================
# Analyze peak→trough zone visitation
# ============================================================
out_dir = Path('figures/entropy_behavior_alignment')
all_decline_data = []  # for CSV output

for snum in target_sessions:
    skey = f"session_{snum}"
    sval = sessions_cfg[skey]
    behav_path = sval['behavior']
    state = sval['state']
    phase = sval['phase']

    print(f"\nS{snum} ({state}/{phase}): loading...", end=' ', flush=True)
    time_vals, vel, zones = load_behavior_xlsx(behav_path)[:3]

    ent_times, ent_vals, vel_means = compute_entropy_causal(
        zones, time_vals, vel, ENTROPY_WINDOW_SEC, ENTROPY_STEP_SEC)
    peaks_idx, troughs_idx, smoothed = find_inflections(
        ent_vals, SMOOTH_SIGMA, MIN_AMPLITUDE)

    # Pair peaks with following troughs
    decline_intervals = []
    for pidx in peaks_idx:
        # Find next trough after this peak
        next_troughs = [t for t in troughs_idx if t > pidx]
        if next_troughs:
            tidx = next_troughs[0]
            decline_intervals.append((pidx, tidx))

    print(f"{len(peaks_idx)} peaks, {len(troughs_idx)} troughs, "
          f"{len(decline_intervals)} decline intervals")

    if len(decline_intervals) == 0:
        continue

    # Print zone details for each decline
    for di, (pidx, tidx) in enumerate(decline_intervals):
        t_start = ent_times[pidx]
        t_end = ent_times[tidx]
        duration = t_end - t_start
        ent_drop = smoothed[pidx] - smoothed[tidx]

        fracs = get_zone_fractions(zones, time_vals, t_start, t_end)
        seq = get_zone_sequence(zones, time_vals, t_start, t_end)

        print(f"  Decline {di+1}: {t_start/60:.1f}-{t_end/60:.1f} min "
              f"({duration:.0f}s), entropy {smoothed[pidx]:.2f} -> {smoothed[tidx]:.2f} "
              f"(drop {ent_drop:.2f} bits)")
        # Top zones by time
        sorted_fracs = sorted(fracs.items(), key=lambda x: -x[1])
        zone_str = ', '.join([f"{z}={f*100:.0f}%" for z, f in sorted_fracs[:6]])
        print(f"    Zones: {zone_str}")
        # Sequence (first 20 transitions)
        seq_str = '->'.join(seq[:25])
        if len(seq) > 25:
            seq_str += f"... ({len(seq)} total)"
        print(f"    Sequence: {seq_str}")

        # Store for CSV
        for z, f in fracs.items():
            all_decline_data.append({
                'session': snum, 'state': state, 'phase': phase,
                'decline_idx': di + 1,
                'start_min': t_start / 60, 'end_min': t_end / 60,
                'duration_s': duration,
                'ent_peak': smoothed[pidx], 'ent_trough': smoothed[tidx],
                'ent_drop': ent_drop,
                'zone': z, 'fraction': f,
                'n_transitions': len(seq) - 1,
            })

    # ---- Figure: stacked bar chart of zone fractions per decline ----
    n_declines = len(decline_intervals)
    fig, axes = plt.subplots(2, 1, figsize=(max(14, n_declines * 3), 12),
                             gridspec_kw={'height_ratios': [2, 1]})

    # Top panel: entropy trace with decline intervals shaded
    ax_ent = axes[0]
    ent_mins = ent_times / 60.0
    ax_ent.plot(ent_mins, ent_vals, color='black', linewidth=1.2, alpha=0.4)
    ax_ent.plot(ent_mins, smoothed, color='black', linewidth=2.5)
    ax_ent.scatter(ent_mins[peaks_idx], smoothed[peaks_idx],
                   color='red', s=100, zorder=5, marker='^', edgecolors='black')
    ax_ent.scatter(ent_mins[troughs_idx], smoothed[troughs_idx],
                   color='blue', s=100, zorder=5, marker='v', edgecolors='black')
    for di, (pidx, tidx) in enumerate(decline_intervals):
        ax_ent.axvspan(ent_mins[pidx], ent_mins[tidx],
                       alpha=0.15, color='gray')
        mid = (ent_mins[pidx] + ent_mins[tidx]) / 2
        ax_ent.text(mid, smoothed[pidx] + 0.15, f'D{di+1}',
                    ha='center', fontsize=12, fontweight='bold', color='gray')
    ax_ent.set_ylabel('Entropy (bits)', fontsize=14, fontweight='bold')
    ax_ent.set_title(f'S{snum} — {state.capitalize()} / {phase.capitalize()} — '
                     f'Peak→Trough Zone Visitation',
                     fontsize=16, fontweight='bold')
    ax_ent.tick_params(labelsize=12)
    ax_ent.spines['top'].set_visible(False)
    ax_ent.spines['right'].set_visible(False)

    # Bottom panel: stacked bar chart
    ax_bar = axes[1]
    x_positions = np.arange(n_declines)
    bar_width = 0.6

    # Collect zone fractions for all declines
    all_fracs = []
    bar_labels = []
    for di, (pidx, tidx) in enumerate(decline_intervals):
        t_start = ent_times[pidx]
        t_end = ent_times[tidx]
        fracs = get_zone_fractions(zones, time_vals, t_start, t_end)
        all_fracs.append(fracs)
        dur = t_end - t_start
        bar_labels.append(f'D{di+1}\n{dur:.0f}s')

    # Get all zones present across declines
    all_zones_present = set()
    for fracs in all_fracs:
        all_zones_present.update(fracs.keys())
    # Order them
    zones_ordered = [z for z in ZONE_ORDER if z in all_zones_present]

    # Draw stacked bars
    bottoms = np.zeros(n_declines)
    legend_handles = []
    for z in zones_ordered:
        vals = [fracs.get(z, 0) for fracs in all_fracs]
        color = ZONE_COLORS.get(z, '#bab0ac')
        bars = ax_bar.bar(x_positions, vals, bar_width, bottom=bottoms,
                          color=color, edgecolor='white', linewidth=0.5)
        # Add percentage labels for zones > 10%
        for i, v in enumerate(vals):
            if v > 0.10:
                ax_bar.text(x_positions[i], bottoms[i] + v / 2, f'{z}\n{v*100:.0f}%',
                            ha='center', va='center', fontsize=10, fontweight='bold')
        bottoms += vals
        legend_handles.append(plt.Rectangle((0, 0), 1, 1, fc=color, label=z))

    ax_bar.set_xticks(x_positions)
    ax_bar.set_xticklabels(bar_labels, fontsize=12)
    ax_bar.set_ylabel('Zone Fraction', fontsize=14, fontweight='bold')
    ax_bar.set_ylim(0, 1.05)
    ax_bar.tick_params(labelsize=12)
    ax_bar.spines['top'].set_visible(False)
    ax_bar.spines['right'].set_visible(False)

    # Legend
    ax_bar.legend(handles=legend_handles, loc='upper left',
                  bbox_to_anchor=(1.01, 1.0), fontsize=12,
                  framealpha=0.9, edgecolor='black', title='Zone', title_fontsize=13)

    plt.tight_layout()
    fname = out_dir / f'S{snum}_{state}_{phase}_peak_to_trough_zones.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> Saved {fname}")

# Save CSV
df_out = pd.DataFrame(all_decline_data)
csv_path = 'data/dp_entropy_peak_to_trough_zones.csv'
df_out.to_csv(csv_path, index=False)
print(f"\nSaved {csv_path} ({len(df_out)} rows)")
print("Done.")
