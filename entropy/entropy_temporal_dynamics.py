"""
Entropy Temporal Dynamics — Foraging Sessions (S2, S4, S6, S8)

Questions:
1. Does entropy decline gradually or drop suddenly?
2. Do entropy drops coincide with food discovery (feeding/digging at Pot-4)?
3. Or are drops driven by repetitive zone-switching independent of food events?

Analyses:
A. Entropy rate-of-change (derivative) — gradual drift vs abrupt transitions
B. Change-point detection on entropy traces (PELT-like segmentation)
C. Overlay feeding/digging events at Pot-4 on entropy traces
D. Cross-correlation between Pot-4 events and entropy changes
E. Entropy conditioned on whether Pot-4 has been discovered yet
"""

import yaml
from pathlib import Path
import numpy as np
import pandas as pd
from collections import Counter
from scipy.stats import entropy as sp_entropy, spearmanr, mannwhitneyu
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import warnings

warnings.filterwarnings('ignore')

with open("paths.yaml") as f:
    cfg = yaml.safe_load(f)

sessions_cfg = cfg["single_probe"]["coordinates_1"]["mouse01"]["sessions"]
session_meta = {
    1: ('fed', 'exploration'), 2: ('fed', 'foraging'),
    3: ('fed', 'exploration'), 4: ('fed', 'foraging'),
    5: ('fasted', 'exploration'), 6: ('fasted', 'foraging'),
    7: ('fasted', 'exploration'), 8: ('fasted', 'foraging'),
}

priority_order = [
    'Right corner', 'Left corner', 'Arna center', 'Foraging arena',
    'Home', 'Ladder', 'Transition zone',
    'Pot-1 zone', 'Pot-2 zone', 'Pot-3 zone', 'Pot-4 zone',
    'Pot-1', 'Pot-2', 'Pot-3', 'Pot-4',
]

zone_short = {
    'Home': 'H', 'Ladder': 'L', 'Transition zone': 'T',
    'Foraging arena': 'FA', 'Arna center': 'AC',
    'Pot-1': 'P1', 'Pot-2': 'P2', 'Pot-3': 'P3', 'Pot-4': 'P4',
    'Pot-1 zone': 'P1z', 'Pot-2 zone': 'P2z', 'Pot-3 zone': 'P3z', 'Pot-4 zone': 'P4z',
    'Right corner': 'RC', 'Left corner': 'LC', 'other': 'O',
}

ENTROPY_WINDOW_SEC = 60
ENTROPY_STEP_SEC = 10
FORAGING_SESSIONS = [2, 4, 6, 8]


# =============================================================================
# HELPERS
# =============================================================================
def load_behavior(behav_path):
    df_raw = pd.read_csv(behav_path, header=None)
    var_names = df_raw.iloc[:, 0].values
    time_vals = df_raw.iloc[1, 1:].astype(float).values
    data = df_raw.iloc[:, 1:].values
    behav = {'time': time_vals}
    for i, name in enumerate(var_names):
        if isinstance(name, str):
            behav[name.strip()] = data[i].astype(float)
    return behav


def get_zones(behav):
    n = len(behav['time'])
    zones = np.full(n, 'O', dtype=object)
    for var_name in priority_order:
        if var_name in behav:
            mask = behav[var_name] > 0.5
            zones[mask] = zone_short.get(var_name, var_name)
    return zones


def compute_entropy_trace(zones, time_vals):
    """Sliding-window transition entropy (matches entropy_neural_signatures.py)."""
    dt = np.median(np.diff(time_vals))
    window_bins = int(ENTROPY_WINDOW_SEC / dt)
    step_bins = int(ENTROPY_STEP_SEC / dt)

    ent_times, ent_vals = [], []
    for start_idx in range(0, len(zones) - window_bins, step_bins):
        end_idx = start_idx + window_bins
        wz = zones[start_idx:end_idx]
        transitions = []
        for j in range(1, len(wz)):
            if wz[j] != wz[j-1]:
                transitions.append(f"{wz[j-1]}->{wz[j]}")
        if len(transitions) < 3:
            continue
        counts = Counter(transitions)
        probs = np.array(list(counts.values()), dtype=float)
        probs /= probs.sum()
        h = sp_entropy(probs, base=2)
        ent_times.append(time_vals[start_idx + window_bins // 2])
        ent_vals.append(h)

    return np.array(ent_times), np.array(ent_vals)


def get_event_onsets(behav, event_name):
    """Get onset times for a binary behavioral event."""
    if event_name not in behav:
        return np.array([])
    signal = behav[event_name]
    onsets = np.where(np.diff(signal > 0.5) & (signal[1:] > 0.5))[0] + 1
    return behav['time'][onsets]


def get_event_bouts(behav, event_name):
    """Get (onset, offset, duration) for each bout of a binary event."""
    if event_name not in behav:
        return []
    signal = (behav[event_name] > 0.5).astype(int)
    diff = np.diff(signal)
    onsets = np.where(diff == 1)[0] + 1
    offsets = np.where(diff == -1)[0] + 1
    if signal[0] > 0.5:
        onsets = np.concatenate([[0], onsets])
    if signal[-1] > 0.5:
        offsets = np.concatenate([offsets, [len(signal) - 1]])
    bouts = []
    for on, off in zip(onsets, offsets):
        bouts.append((behav['time'][on], behav['time'][off],
                       behav['time'][off] - behav['time'][on]))
    return bouts


def is_at_pot4(behav, time_point, tolerance=2.0):
    """Check if animal is at Pot-4 or Pot-4 zone within tolerance seconds."""
    t = behav['time']
    mask = np.abs(t - time_point) <= tolerance
    for var in ['Pot-4', 'Pot-4 zone']:
        if var in behav and np.any(behav[var][mask] > 0.5):
            return True
    return False


def detect_entropy_drops(ent_times, ent_vals, threshold_pct=25):
    """Detect significant drops: points where entropy falls below threshold percentile.
    Returns onset indices of drop episodes."""
    thresh = np.percentile(ent_vals, threshold_pct)
    below = ent_vals < thresh
    # Find onset of each drop episode
    diff = np.diff(below.astype(int))
    onsets = np.where(diff == 1)[0] + 1
    # If starts below threshold
    if below[0]:
        onsets = np.concatenate([[0], onsets])
    return onsets, thresh


def compute_entropy_derivative(ent_times, ent_vals, smooth_sigma=2):
    """Compute smoothed derivative of entropy trace."""
    dt = np.median(np.diff(ent_times))
    smoothed = gaussian_filter1d(ent_vals, sigma=smooth_sigma)
    deriv = np.gradient(smoothed, dt)
    return deriv, smoothed


# =============================================================================
# LOAD DATA
# =============================================================================
print("=" * 70)
print("ENTROPY TEMPORAL DYNAMICS — Foraging Sessions")
print("=" * 70)

all_data = {}

for snum in FORAGING_SESSIONS:
    state, phase = session_meta[snum]
    sc = sessions_cfg[f"session_{snum}"]
    behav_path = sc.get('behavior')
    if not behav_path or not Path(behav_path).exists():
        print(f"S{snum}: no behavior data — SKIP")
        continue

    behav = load_behavior(behav_path)
    zones = get_zones(behav)
    ent_times, ent_vals = compute_entropy_trace(zones, behav['time'])

    # Get behavioral events
    feeding_onsets = get_event_onsets(behav, 'Feeding')
    digging_onsets = get_event_onsets(behav, 'Digging')
    feeding_bouts = get_event_bouts(behav, 'Feeding')
    digging_bouts = get_event_bouts(behav, 'Digging')

    # Which feeding/digging events are at Pot-4?
    feed_at_p4 = [t for t in feeding_onsets if is_at_pot4(behav, t)]
    dig_at_p4 = [t for t in digging_onsets if is_at_pot4(behav, t)]
    feed_not_p4 = [t for t in feeding_onsets if not is_at_pot4(behav, t)]
    dig_not_p4 = [t for t in digging_onsets if not is_at_pot4(behav, t)]

    # Pot-4 visit times (any time at Pot-4 or Pot-4 zone)
    p4_mask = np.zeros(len(behav['time']), dtype=bool)
    for var in ['Pot-4', 'Pot-4 zone']:
        if var in behav:
            p4_mask |= (behav[var] > 0.5)
    p4_onset_idx = np.where(np.diff(p4_mask.astype(int)) == 1)[0] + 1
    p4_visit_times = behav['time'][p4_onset_idx] if len(p4_onset_idx) > 0 else np.array([])

    # First Pot-4 feeding event
    first_p4_feed = feed_at_p4[0] if len(feed_at_p4) > 0 else None
    first_p4_dig = dig_at_p4[0] if len(dig_at_p4) > 0 else None

    # Entropy derivative
    deriv, smoothed = compute_entropy_derivative(ent_times, ent_vals)

    # Entropy drop detection
    drop_onsets, drop_thresh = detect_entropy_drops(ent_times, ent_vals)

    all_data[snum] = {
        'state': state, 'phase': phase, 'behav': behav, 'zones': zones,
        'ent_times': ent_times, 'ent_vals': ent_vals,
        'deriv': deriv, 'smoothed': smoothed,
        'feeding_onsets': feeding_onsets, 'digging_onsets': digging_onsets,
        'feed_at_p4': np.array(feed_at_p4), 'dig_at_p4': np.array(dig_at_p4),
        'feed_not_p4': np.array(feed_not_p4), 'dig_not_p4': np.array(dig_not_p4),
        'p4_visit_times': p4_visit_times,
        'first_p4_feed': first_p4_feed, 'first_p4_dig': first_p4_dig,
        'drop_onsets': drop_onsets, 'drop_thresh': drop_thresh,
    }

    print(f"\nS{snum} ({state}/{phase}):")
    print(f"  Entropy: mean={np.mean(ent_vals):.2f}, std={np.std(ent_vals):.2f}, "
          f"min={np.min(ent_vals):.2f}, max={np.max(ent_vals):.2f}")
    print(f"  Feeding events: {len(feeding_onsets)} total, {len(feed_at_p4)} at Pot-4, "
          f"{len(feed_not_p4)} elsewhere")
    print(f"  Digging events: {len(digging_onsets)} total, {len(dig_at_p4)} at Pot-4, "
          f"{len(dig_not_p4)} elsewhere")
    print(f"  Pot-4 visits: {len(p4_visit_times)}")
    print(f"  First P4 feed: {first_p4_feed:.1f}s" if first_p4_feed else "  First P4 feed: NONE")
    print(f"  First P4 dig: {first_p4_dig:.1f}s" if first_p4_dig else "  First P4 dig: NONE")
    print(f"  Entropy drops (below {drop_thresh:.2f}): {len(drop_onsets)} episodes")


# =============================================================================
# ANALYSIS A: Gradual vs Sudden — Derivative Distribution
# =============================================================================
print("\n\n" + "=" * 70)
print("ANALYSIS A: ENTROPY RATE OF CHANGE")
print("=" * 70)

for snum in FORAGING_SESSIONS:
    if snum not in all_data:
        continue
    d = all_data[snum]
    deriv = d['deriv']

    # Classify derivative values
    big_neg = np.sum(deriv < -0.01)  # sharp drops
    big_pos = np.sum(deriv > 0.01)   # sharp rises
    small = np.sum(np.abs(deriv) <= 0.01)  # gradual/stable

    # Largest drops
    sorted_deriv = np.sort(deriv)
    top5_drops = sorted_deriv[:5]

    # Autocorrelation of entropy — high = gradual drift, low = jumpy
    ent_centered = d['ent_vals'] - np.mean(d['ent_vals'])
    ac1 = np.corrcoef(ent_centered[:-1], ent_centered[1:])[0, 1]
    ac3 = np.corrcoef(ent_centered[:-3], ent_centered[3:])[0, 1]

    print(f"\nS{snum} ({d['state']}/{d['phase']}):")
    print(f"  Derivative: mean={np.mean(deriv):.4f}, std={np.std(deriv):.4f}")
    print(f"  Sharp drops (d<-0.01): {big_neg}/{len(deriv)} ({100*big_neg/len(deriv):.0f}%)")
    print(f"  Sharp rises (d>+0.01): {big_pos}/{len(deriv)} ({100*big_pos/len(deriv):.0f}%)")
    print(f"  Stable (|d|<0.01):     {small}/{len(deriv)} ({100*small/len(deriv):.0f}%)")
    print(f"  Top 5 negative derivatives: {[f'{x:.4f}' for x in top5_drops]}")
    print(f"  Autocorrelation: lag-1={ac1:.3f}, lag-3={ac3:.3f}")

    # Check for step-like changes: run length of consecutive same-sign derivative
    signs = np.sign(deriv)
    run_lengths = []
    current_run = 1
    for i in range(1, len(signs)):
        if signs[i] == signs[i-1] and signs[i] != 0:
            current_run += 1
        else:
            run_lengths.append(current_run)
            current_run = 1
    run_lengths.append(current_run)
    print(f"  Derivative run lengths: mean={np.mean(run_lengths):.1f}, "
          f"max={np.max(run_lengths)}, median={np.median(run_lengths):.0f}")


# =============================================================================
# ANALYSIS B: Entropy Drops vs Pot-4 Events
# =============================================================================
print("\n\n" + "=" * 70)
print("ANALYSIS B: DO ENTROPY DROPS COINCIDE WITH POT-4 FOOD EVENTS?")
print("=" * 70)

for snum in FORAGING_SESSIONS:
    if snum not in all_data:
        continue
    d = all_data[snum]
    ent_times = d['ent_times']
    ent_vals = d['ent_vals']

    print(f"\nS{snum} ({d['state']}/{d['phase']}):")

    # For each entropy drop onset, find nearest Pot-4 event
    if len(d['drop_onsets']) > 0 and len(d['feed_at_p4']) > 0:
        for didx in d['drop_onsets']:
            drop_time = ent_times[didx]
            nearest_p4 = d['feed_at_p4'][np.argmin(np.abs(d['feed_at_p4'] - drop_time))]
            dt = drop_time - nearest_p4
            print(f"  Drop at {drop_time:.0f}s — nearest P4 feed: {nearest_p4:.0f}s (lag={dt:+.0f}s)")
    elif len(d['drop_onsets']) > 0:
        print(f"  {len(d['drop_onsets'])} drops but NO Pot-4 feeding events")
    else:
        print(f"  No entropy drops detected")

    # Entropy at time of Pot-4 events vs random times
    if len(d['feed_at_p4']) > 0:
        ent_at_p4 = np.interp(d['feed_at_p4'], ent_times, ent_vals)
        ent_overall = np.mean(ent_vals)
        print(f"  Entropy at P4 feeds: mean={np.mean(ent_at_p4):.3f} vs overall={ent_overall:.3f} "
              f"(diff={np.mean(ent_at_p4)-ent_overall:+.3f})")

    if len(d['dig_at_p4']) > 0:
        ent_at_p4_dig = np.interp(d['dig_at_p4'], ent_times, ent_vals)
        print(f"  Entropy at P4 digs: mean={np.mean(ent_at_p4_dig):.3f} vs overall={np.mean(ent_vals):.3f}")

    # Compare: entropy BEFORE vs AFTER first Pot-4 discovery
    if d['first_p4_feed'] is not None:
        fp4 = d['first_p4_feed']
        before = ent_vals[ent_times < fp4]
        after = ent_vals[ent_times >= fp4]
        if len(before) > 3 and len(after) > 3:
            stat, p = mannwhitneyu(before, after, alternative='two-sided')
            print(f"  Pre-P4-discovery entropy: {np.mean(before):.3f} ({len(before)} pts)")
            print(f"  Post-P4-discovery entropy: {np.mean(after):.3f} ({len(after)} pts)")
            print(f"  Mann-Whitney U: p={p:.4f} {'*' if p < 0.05 else 'ns'}")
        else:
            print(f"  First P4 feed at {fp4:.0f}s — too few points for pre/post comparison")

    # Key test: What % of entropy drops occur WITHOUT a nearby P4 event?
    if len(d['drop_onsets']) > 0:
        p4_all_events = np.sort(np.concatenate([d['feed_at_p4'], d['dig_at_p4']]))
        drops_near_p4 = 0
        drops_no_p4 = 0
        proximity_sec = 30  # within 30s window
        for didx in d['drop_onsets']:
            drop_time = ent_times[didx]
            if len(p4_all_events) > 0:
                min_dist = np.min(np.abs(p4_all_events - drop_time))
                if min_dist <= proximity_sec:
                    drops_near_p4 += 1
                else:
                    drops_no_p4 += 1
            else:
                drops_no_p4 += 1
        total_drops = drops_near_p4 + drops_no_p4
        print(f"  Drops near P4 events (±{proximity_sec}s): {drops_near_p4}/{total_drops} "
              f"({100*drops_near_p4/total_drops:.0f}%)")
        print(f"  Drops WITHOUT P4 events: {drops_no_p4}/{total_drops} "
              f"({100*drops_no_p4/total_drops:.0f}%)")


# =============================================================================
# ANALYSIS C: What zone transitions dominate during low entropy?
# =============================================================================
print("\n\n" + "=" * 70)
print("ANALYSIS C: DOMINANT TRANSITIONS DURING LOW vs HIGH ENTROPY")
print("=" * 70)

for snum in FORAGING_SESSIONS:
    if snum not in all_data:
        continue
    d = all_data[snum]
    ent_times = d['ent_times']
    ent_vals = d['ent_vals']
    zones = d['zones']
    time_vals = d['behav']['time']
    dt = np.median(np.diff(time_vals))
    window_bins = int(ENTROPY_WINDOW_SEC / dt)

    q25 = np.percentile(ent_vals, 25)
    q75 = np.percentile(ent_vals, 75)

    # Get transitions during low and high entropy windows
    low_transitions = Counter()
    high_transitions = Counter()

    step_bins = int(ENTROPY_STEP_SEC / dt)
    ent_idx = 0
    for start_idx in range(0, len(zones) - window_bins, step_bins):
        if ent_idx >= len(ent_vals):
            break
        wz = zones[start_idx:start_idx + window_bins]
        trans = []
        for j in range(1, len(wz)):
            if wz[j] != wz[j-1]:
                trans.append(f"{wz[j-1]}->{wz[j]}")
        if len(trans) < 3:
            ent_idx += 1
            continue
        if ent_vals[ent_idx] <= q25:
            low_transitions.update(trans)
        elif ent_vals[ent_idx] >= q75:
            high_transitions.update(trans)
        ent_idx += 1

    print(f"\nS{snum} ({d['state']}/{d['phase']}):")
    print(f"  LOW entropy (<=Q25={q25:.2f}) — top transitions:")
    for trans, count in low_transitions.most_common(8):
        pct = 100 * count / sum(low_transitions.values())
        involves_p4 = 'P4' in trans
        print(f"    {trans:<20} {count:>5} ({pct:5.1f}%) {'<-- P4' if involves_p4 else ''}")

    print(f"  HIGH entropy (>=Q75={q75:.2f}) — top transitions:")
    for trans, count in high_transitions.most_common(8):
        pct = 100 * count / sum(high_transitions.values())
        involves_p4 = 'P4' in trans
        print(f"    {trans:<20} {count:>5} ({pct:5.1f}%) {'<-- P4' if involves_p4 else ''}")

    # Fraction of transitions involving P4 in low vs high entropy
    low_p4 = sum(v for k, v in low_transitions.items() if 'P4' in k)
    low_total = sum(low_transitions.values())
    high_p4 = sum(v for k, v in high_transitions.items() if 'P4' in k)
    high_total = sum(high_transitions.values())

    print(f"\n  P4-involving transitions: LOW={100*low_p4/low_total:.1f}% "
          f"({low_p4}/{low_total}), HIGH={100*high_p4/high_total:.1f}% "
          f"({high_p4}/{high_total})")

    # Fraction involving Pot-2 (visible food in foraging? no — exploration has Pot-2)
    low_p2 = sum(v for k, v in low_transitions.items() if 'P2' in k)
    high_p2 = sum(v for k, v in high_transitions.items() if 'P2' in k)
    print(f"  P2-involving transitions: LOW={100*low_p2/low_total:.1f}%, "
          f"HIGH={100*high_p2/high_total:.1f}%")

    # How many UNIQUE transition types in low vs high?
    print(f"  Unique transition types: LOW={len(low_transitions)}, HIGH={len(high_transitions)}")


# =============================================================================
# ANALYSIS D: Pot-4 visit rate vs entropy over time
# =============================================================================
print("\n\n" + "=" * 70)
print("ANALYSIS D: POT-4 VISIT RATE vs ENTROPY")
print("=" * 70)

for snum in FORAGING_SESSIONS:
    if snum not in all_data:
        continue
    d = all_data[snum]
    ent_times = d['ent_times']
    ent_vals = d['ent_vals']

    # Compute Pot-4 visit rate in same sliding windows
    p4_rate = np.zeros(len(ent_times))
    for i, et in enumerate(ent_times):
        t0 = et - ENTROPY_WINDOW_SEC / 2
        t1 = et + ENTROPY_WINDOW_SEC / 2
        n_visits = np.sum((d['p4_visit_times'] >= t0) & (d['p4_visit_times'] < t1))
        p4_rate[i] = n_visits / (ENTROPY_WINDOW_SEC / 60)  # visits per minute

    if np.std(p4_rate) > 0:
        rho, p = spearmanr(ent_vals, p4_rate)
        print(f"\nS{snum} ({d['state']}/{d['phase']}): entropy vs P4 visit rate: "
              f"rho={rho:.3f}, p={p:.4f} {'*' if p < 0.05 else 'ns'}")
        print(f"  P4 visit rate: mean={np.mean(p4_rate):.2f}/min, "
              f"max={np.max(p4_rate):.1f}/min")
    else:
        print(f"\nS{snum}: No Pot-4 visits or constant rate")

    all_data[snum]['p4_rate'] = p4_rate


# =============================================================================
# FIGURE 1: Entropy traces with behavioral events
# =============================================================================
fig, axes = plt.subplots(4, 1, figsize=(16, 16))
fig.suptitle("Entropy Temporal Dynamics — Foraging Sessions\n"
             "(60s window transition entropy, 10s steps)",
             fontsize=14, fontweight='bold')

for idx, snum in enumerate(FORAGING_SESSIONS):
    if snum not in all_data:
        continue
    d = all_data[snum]
    ax = axes[idx]

    color = '#e74c3c' if d['state'] == 'fasted' else '#3498db'

    # Entropy trace
    ax.plot(d['ent_times'], d['ent_vals'], color=color, linewidth=1.5, alpha=0.8,
            label='Entropy')
    ax.plot(d['ent_times'], d['smoothed'], color='black', linewidth=2, alpha=0.6,
            label='Smoothed')

    # Drop threshold
    ax.axhline(d['drop_thresh'], color='gray', linestyle=':', alpha=0.5,
               label=f'Q25={d["drop_thresh"]:.1f}')

    # Feeding events
    for ft in d['feed_at_p4']:
        ax.axvline(ft, color='red', linewidth=2, alpha=0.7, zorder=3)
    for ft in d['feed_not_p4']:
        ax.axvline(ft, color='orange', linewidth=1.5, alpha=0.5)

    # Digging at Pot-4
    for dt_val in d['dig_at_p4']:
        ax.axvline(dt_val, color='darkred', linewidth=1.5, alpha=0.5, linestyle='--')

    # First P4 discovery marker
    if d['first_p4_feed'] is not None:
        ax.axvline(d['first_p4_feed'], color='green', linewidth=3, alpha=0.8,
                   linestyle='-', label=f'1st P4 feed ({d["first_p4_feed"]:.0f}s)')

    # Pot-4 visit rate on twin axis
    ax2 = ax.twinx()
    ax2.fill_between(d['ent_times'], d['p4_rate'], alpha=0.15, color='purple')
    ax2.set_ylabel('P4 visits/min', color='purple', fontsize=9)
    ax2.tick_params(axis='y', labelcolor='purple')
    ax2.set_ylim(0, max(np.max(d['p4_rate']) * 1.5, 1))

    ax.set_title(f"S{snum} ({d['state']}/{d['phase']}) — "
                 f"P4 feeds: {len(d['feed_at_p4'])}, "
                 f"non-P4 feeds: {len(d['feed_not_p4'])}, "
                 f"P4 visits: {len(d['p4_visit_times'])}",
                 fontsize=11)
    ax.set_ylabel("Entropy (bits)")
    ax.set_ylim(0, 5.5)
    ax.legend(loc='upper right', fontsize=8)

axes[-1].set_xlabel("Time (s)")

# Custom legend
from matplotlib.lines import Line2D
custom_lines = [
    Line2D([0], [0], color='red', linewidth=2, label='Feed at P4'),
    Line2D([0], [0], color='orange', linewidth=1.5, label='Feed elsewhere'),
    Line2D([0], [0], color='darkred', linewidth=1.5, linestyle='--', label='Dig at P4'),
    Line2D([0], [0], color='green', linewidth=3, label='1st P4 feed'),
]
fig.legend(handles=custom_lines, loc='lower center', ncol=4, fontsize=10,
           bbox_to_anchor=(0.5, -0.02))

plt.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.savefig("figures/entropy_temporal_dynamics.png", dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved figures/entropy_temporal_dynamics.png")


# =============================================================================
# FIGURE 2: Derivative analysis — gradual vs sudden
# =============================================================================
fig, axes = plt.subplots(4, 2, figsize=(16, 14))
fig.suptitle("Entropy Rate of Change — Gradual Drift vs Abrupt Drops",
             fontsize=14, fontweight='bold')

for idx, snum in enumerate(FORAGING_SESSIONS):
    if snum not in all_data:
        continue
    d = all_data[snum]
    color = '#e74c3c' if d['state'] == 'fasted' else '#3498db'

    # Left: derivative over time
    ax = axes[idx, 0]
    ax.plot(d['ent_times'], d['deriv'], color=color, linewidth=1, alpha=0.7)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axhline(-0.01, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(0.01, color='gray', linestyle=':', alpha=0.5)
    ax.fill_between(d['ent_times'], d['deriv'], 0,
                    where=d['deriv'] < -0.01, alpha=0.3, color='red',
                    label='Sharp drops')
    ax.set_title(f"S{snum} ({d['state']}/{d['phase']}) — dH/dt", fontsize=10)
    ax.set_ylabel("dEntropy/dt")
    ax.legend(fontsize=8)

    # Right: histogram of derivatives
    ax = axes[idx, 1]
    ax.hist(d['deriv'], bins=40, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=1)

    # Mark percentiles
    p5 = np.percentile(d['deriv'], 5)
    p95 = np.percentile(d['deriv'], 95)
    ax.axvline(p5, color='red', linestyle='--', label=f'P5={p5:.4f}')
    ax.axvline(p95, color='blue', linestyle='--', label=f'P95={p95:.4f}')
    ax.set_title(f"Derivative distribution (skew={pd.Series(d['deriv']).skew():.2f})",
                 fontsize=10)
    ax.set_xlabel("dEntropy/dt")
    ax.legend(fontsize=8)

axes[-1, 0].set_xlabel("Time (s)")
plt.tight_layout()
plt.savefig("figures/entropy_derivative_analysis.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved figures/entropy_derivative_analysis.png")


# =============================================================================
# FIGURE 3: Low vs High entropy transition composition
# =============================================================================
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle("Zone Transition Composition During Low vs High Entropy",
             fontsize=14, fontweight='bold')

for idx, snum in enumerate(FORAGING_SESSIONS):
    if snum not in all_data:
        continue
    d = all_data[snum]
    ent_times = d['ent_times']
    ent_vals = d['ent_vals']
    zones = d['zones']
    time_vals = d['behav']['time']
    dt_val = np.median(np.diff(time_vals))
    window_bins = int(ENTROPY_WINDOW_SEC / dt_val)
    step_bins = int(ENTROPY_STEP_SEC / dt_val)

    q25 = np.percentile(ent_vals, 25)
    q75 = np.percentile(ent_vals, 75)

    for row, (label, threshold_func, q_val) in enumerate([
        ("LOW", lambda x: x <= q25, q25),
        ("HIGH", lambda x: x >= q75, q75)
    ]):
        ax = axes[row, idx]
        transitions = Counter()
        ent_idx = 0
        for start_idx in range(0, len(zones) - window_bins, step_bins):
            if ent_idx >= len(ent_vals):
                break
            if threshold_func(ent_vals[ent_idx]):
                wz = zones[start_idx:start_idx + window_bins]
                for j in range(1, len(wz)):
                    if wz[j] != wz[j-1]:
                        transitions[f"{wz[j-1]}->{wz[j]}"] += 1
            ent_idx += 1

        # Top 10 transitions as horizontal bar
        top = transitions.most_common(10)
        if len(top) > 0:
            labels_t = [t[0] for t in top][::-1]
            counts = [t[1] for t in top][::-1]
            total = sum(transitions.values())
            pcts = [100 * c / total for c in counts]

            colors = ['#cc0000' if 'P4' in l else '#3366cc' if 'P2' in l else '#999999'
                      for l in labels_t]
            ax.barh(range(len(labels_t)), pcts, color=colors, edgecolor='black', linewidth=0.5)
            ax.set_yticks(range(len(labels_t)))
            ax.set_yticklabels(labels_t, fontsize=8)
            ax.set_xlabel("% of transitions")

        title_str = f"S{snum} {label} ent" if row == 0 else f"S{snum} {label} ent"
        ax.set_title(f"S{snum} ({d['state'][:3]}/{d['phase'][:3]}) — {label} entropy",
                     fontsize=9)

# Add legend
from matplotlib.patches import Patch as MPatch
legend_elements = [
    MPatch(facecolor='#cc0000', edgecolor='black', label='P4-involved'),
    MPatch(facecolor='#3366cc', edgecolor='black', label='P2-involved'),
    MPatch(facecolor='#999999', edgecolor='black', label='Other'),
]
fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=11,
           bbox_to_anchor=(0.5, -0.02))

plt.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.savefig("figures/entropy_transition_composition.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved figures/entropy_transition_composition.png")


# =============================================================================
# SUMMARY TABLE
# =============================================================================
print("\n\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

summary_rows = []
for snum in FORAGING_SESSIONS:
    if snum not in all_data:
        continue
    d = all_data[snum]
    ent = d['ent_vals']
    deriv = d['deriv']

    # Autocorrelation
    ec = ent - np.mean(ent)
    ac1 = np.corrcoef(ec[:-1], ec[1:])[0, 1]

    # Derivative skewness
    skew = pd.Series(deriv).skew()

    # % time in low entropy
    q25 = np.percentile(ent, 25)
    pct_low = 100 * np.sum(ent <= q25) / len(ent)

    # P4 correlation
    if np.std(d['p4_rate']) > 0:
        rho_p4, p_p4 = spearmanr(ent, d['p4_rate'])
    else:
        rho_p4, p_p4 = np.nan, np.nan

    row = {
        'Session': f'S{snum}', 'State': d['state'], 'Phase': d['phase'],
        'Entropy_mean': np.mean(ent), 'Entropy_std': np.std(ent),
        'AC_lag1': ac1, 'Deriv_skew': skew,
        'N_drops': len(d['drop_onsets']),
        'N_P4_feeds': len(d['feed_at_p4']),
        'P4_rate_rho': rho_p4, 'P4_rate_p': p_p4,
        'First_P4_feed_sec': d['first_p4_feed'] if d['first_p4_feed'] else np.nan,
    }
    summary_rows.append(row)

    print(f"\nS{snum} ({d['state']}/{d['phase']}):")
    print(f"  Autocorrelation lag-1: {ac1:.3f} {'(gradual)' if ac1 > 0.7 else '(mixed)' if ac1 > 0.4 else '(jumpy)'}")
    print(f"  Derivative skewness: {skew:.3f} {'(negative skew = sharp drops)' if skew < -0.3 else '(symmetric)' if abs(skew) < 0.3 else '(positive skew)'}")
    print(f"  Entropy vs P4 rate: rho={rho_p4:.3f}, p={p_p4:.4f}" if not np.isnan(rho_p4) else "  No P4 visits")
    print(f"  Interpretation: {'GRADUAL decline' if ac1 > 0.7 else 'MIX of gradual + sudden' if ac1 > 0.4 else 'SUDDEN drops'}")

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv("data/entropy_temporal_dynamics.csv", index=False)
print(f"\nSaved data/entropy_temporal_dynamics.csv")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
