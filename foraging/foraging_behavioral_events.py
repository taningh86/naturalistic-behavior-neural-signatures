"""
Extract foraging behavioral events for sessions 2, 4, 6, 8.

Events to detect:
1. Pot-2 visits: mouse dwells at Pot-2 or Pot-2 zone for >= 1s (10 bins)
2. Pot-2 -> other pot transitions: mouse leaves Pot-2 and dwells at another pot >= 1s
3. Digging events: time, duration, which pot
4. Feeding events: time, duration, which pot (if determinable)

Goal: map the behavioral sequence from expectation (Pot-2) -> surprise ->
strategy shift -> commitment (digging) -> reward (feeding).
"""

import yaml
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys

with open("paths.yaml") as f:
    paths_config = yaml.safe_load(f)

BIN_MS = 100  # 100ms bins
MIN_DWELL_BINS = 10  # 1 second = 10 bins


def load_behavior_data(session_num):
    sp = paths_config['single_probe']['coordinates_1']['mouse01']['sessions']
    sc = sp[f'session_{session_num}']
    bp = sc.get('behavior')
    if not bp or not Path(bp).exists():
        return None, 0
    df = pd.read_csv(bp, header=None)
    n_bins = df.shape[1] - 1
    result = {}
    for i in range(df.shape[0]):
        name = str(df.iloc[i, 0]).strip()
        if name and name != 'nan':
            vals = pd.to_numeric(df.iloc[i, 1:], errors='coerce').values
            result[name] = vals
    return result, n_bins


def find_dwell_events(signal, min_bins=MIN_DWELL_BINS):
    """Find contiguous runs of 1s in a binary signal, >= min_bins long.
    Returns list of (start_bin, end_bin, duration_bins)."""
    events = []
    in_event = False
    start = 0
    for i in range(len(signal)):
        val = signal[i]
        if not np.isnan(val) and val > 0:
            if not in_event:
                in_event = True
                start = i
        else:
            if in_event:
                dur = i - start
                if dur >= min_bins:
                    events.append((start, i - 1, dur))
                in_event = False
    if in_event:
        dur = len(signal) - start
        if dur >= min_bins:
            events.append((start, len(signal) - 1, dur))
    return events


def find_behavior_events(signal, min_bins=1):
    """Find contiguous runs of behavior (1s), min 1 bin."""
    events = []
    in_event = False
    start = 0
    for i in range(len(signal)):
        val = signal[i]
        if not np.isnan(val) and val > 0:
            if not in_event:
                in_event = True
                start = i
        else:
            if in_event:
                dur = i - start
                if dur >= min_bins:
                    events.append((start, i - 1, dur))
                in_event = False
    if in_event:
        dur = len(signal) - start
        if dur >= min_bins:
            events.append((start, len(signal) - 1, dur))
    return events


def which_pot_at_time(behav, t):
    """Return which pot(s) the mouse is at during time bin t."""
    pots = []
    for p in ['Pot-1', 'Pot-2', 'Pot-3', 'Pot-4']:
        if p in behav:
            v = behav[p]
            if t < len(v) and not np.isnan(v[t]) and v[t] > 0:
                pots.append(p)
    # Also check pot zones
    for p in ['Pot-1 zone', 'Pot-2 zone', 'Pot-3 zone', 'Pot-4 zone']:
        if p in behav:
            v = behav[p]
            if t < len(v) and not np.isnan(v[t]) and v[t] > 0:
                pots.append(p)
    return pots


def process_session(sess_num):
    behav, n_bins = load_behavior_data(sess_num)
    if behav is None:
        print(f"  Session {sess_num}: No behavior data")
        return None

    state = "Fed" if sess_num <= 4 else "Fasted"
    print(f"\n{'='*70}")
    print(f"  Session {sess_num} ({state}, Foraging)")
    print(f"  {n_bins} bins = {n_bins * 0.1:.1f}s")
    print(f"{'='*70}")

    # Available variables
    print(f"\n  Available behavior keys:")
    for k in sorted(behav.keys()):
        print(f"    {k}")

    # =====================================================================
    # 1. POT-2 DWELL EVENTS (>= 1s)
    # =====================================================================
    print(f"\n  --- Pot-2 Visits (dwell >= 1s) ---")
    # Combine Pot-2 and Pot-2 zone (mouse is "at" pot-2 if either is active)
    pot2_signal = np.zeros(n_bins)
    if 'Pot-2' in behav:
        p2 = behav['Pot-2']
        pot2_signal = np.where(~np.isnan(p2) & (p2 > 0), 1, pot2_signal)
    if 'Pot-2 zone' in behav:
        p2z = behav['Pot-2 zone']
        pot2_signal = np.where(~np.isnan(p2z) & (p2z > 0), 1, pot2_signal)

    pot2_events = find_dwell_events(pot2_signal, MIN_DWELL_BINS)
    print(f"  Found {len(pot2_events)} Pot-2 dwell events (>= 1s)")
    for i, (s, e, d) in enumerate(pot2_events):
        t_start = s * 0.1
        t_end = e * 0.1
        dur = d * 0.1
        # Check if feeding or digging during this visit
        feed_during = 0
        dig_during = 0
        if 'Feeding' in behav:
            feed_during = np.nansum(behav['Feeding'][s:e+1] > 0)
        if 'Digging' in behav:
            dig_during = np.nansum(behav['Digging'][s:e+1] > 0)
        print(f"    Visit {i+1}: {t_start:.1f}-{t_end:.1f}s ({dur:.1f}s) "
              f"feed={feed_during} dig={dig_during} bins")

    # =====================================================================
    # 2. ALL POT DWELL EVENTS (>= 1s) for each pot
    # =====================================================================
    print(f"\n  --- All Pot Visits (dwell >= 1s) ---")
    pot_events = {}
    for pot_name in ['Pot-1', 'Pot-2', 'Pot-3', 'Pot-4']:
        pot_signal = np.zeros(n_bins)
        if pot_name in behav:
            p = behav[pot_name]
            pot_signal = np.where(~np.isnan(p) & (p > 0), 1, pot_signal)
        zone_name = f'{pot_name} zone'
        if zone_name in behav:
            pz = behav[zone_name]
            pot_signal = np.where(~np.isnan(pz) & (pz > 0), 1, pot_signal)
        events = find_dwell_events(pot_signal, MIN_DWELL_BINS)
        pot_events[pot_name] = events
        print(f"  {pot_name}: {len(events)} dwell events")
        for i, (s, e, d) in enumerate(events):
            print(f"    Visit {i+1}: {s*0.1:.1f}-{e*0.1:.1f}s ({d*0.1:.1f}s)")

    # =====================================================================
    # 3. DIGGING EVENTS — time, duration, which pot
    # =====================================================================
    print(f"\n  --- Digging Events ---")
    dig_events = []
    if 'Digging' in behav:
        raw_digs = find_behavior_events(behav['Digging'], min_bins=1)
        for s, e, d in raw_digs:
            # Determine which pot(s) mouse is at during digging
            pot_counts = {}
            for t in range(s, e + 1):
                pots_at_t = which_pot_at_time(behav, t)
                for p in pots_at_t:
                    if 'zone' not in p:  # only count direct pot contact
                        pot_counts[p] = pot_counts.get(p, 0) + 1
            # Also check zones if no direct pot
            if not pot_counts:
                for t in range(s, e + 1):
                    pots_at_t = which_pot_at_time(behav, t)
                    for p in pots_at_t:
                        if 'zone' in p:
                            pot_counts[p] = pot_counts.get(p, 0) + 1
            primary_pot = max(pot_counts, key=pot_counts.get) if pot_counts else "Unknown"
            primary_pot = primary_pot.replace(' zone', '')  # "Pot-4 zone" -> "Pot-4"
            dig_events.append((s, e, d, primary_pot))
            print(f"    Dig: {s*0.1:.1f}-{e*0.1:.1f}s ({d*0.1:.1f}s) at {primary_pot}")
    else:
        print(f"    No 'Digging' variable found")

    # =====================================================================
    # 4. FEEDING EVENTS — time, duration, which pot
    # =====================================================================
    print(f"\n  --- Feeding Events ---")
    feed_events = []
    if 'Feeding' in behav:
        raw_feeds = find_behavior_events(behav['Feeding'], min_bins=1)
        for s, e, d in raw_feeds:
            pot_counts = {}
            for t in range(s, e + 1):
                pots_at_t = which_pot_at_time(behav, t)
                for p in pots_at_t:
                    if 'zone' not in p:
                        pot_counts[p] = pot_counts.get(p, 0) + 1
            if not pot_counts:
                for t in range(s, e + 1):
                    pots_at_t = which_pot_at_time(behav, t)
                    for p in pots_at_t:
                        if 'zone' in p:
                            pot_counts[p] = pot_counts.get(p, 0) + 1
            primary_pot = max(pot_counts, key=pot_counts.get) if pot_counts else "Unknown"
            primary_pot = primary_pot.replace(' zone', '')  # "Pot-4 zone" -> "Pot-4"
            feed_events.append((s, e, d, primary_pot))
            print(f"    Feed: {s*0.1:.1f}-{e*0.1:.1f}s ({d*0.1:.1f}s) at {primary_pot}")
    else:
        print(f"    No 'Feeding' variable found")

    # =====================================================================
    # 5. POT-2 -> OTHER POT TRANSITIONS
    # =====================================================================
    print(f"\n  --- Pot-2 -> Other Pot Transitions ---")
    transitions = []
    for p2_s, p2_e, p2_d in pot2_events:
        # Look for the next pot visit after this Pot-2 visit ends
        for other_pot in ['Pot-1', 'Pot-3', 'Pot-4']:
            for op_s, op_e, op_d in pot_events[other_pot]:
                if op_s > p2_e and (op_s - p2_e) * 0.1 < 120:  # within 2 min
                    gap = (op_s - p2_e) * 0.1
                    transitions.append({
                        'from_pot': 'Pot-2',
                        'from_start': p2_s * 0.1,
                        'from_end': p2_e * 0.1,
                        'to_pot': other_pot,
                        'to_start': op_s * 0.1,
                        'to_end': op_e * 0.1,
                        'gap_s': gap,
                    })
                    break  # only first visit to this pot after Pot-2

    for tr in transitions:
        print(f"    Pot-2 ({tr['from_start']:.1f}-{tr['from_end']:.1f}s) -> "
              f"{tr['to_pot']} ({tr['to_start']:.1f}-{tr['to_end']:.1f}s) "
              f"gap={tr['gap_s']:.1f}s")

    # =====================================================================
    # 6. TIMELINE FIGURE
    # =====================================================================
    fig, axes = plt.subplots(4, 1, figsize=(20, 10), sharex=True)
    fig.suptitle(f'Session {sess_num} ({state}, Foraging) — Behavioral Event Timeline\n'
                 f'Pot visits, digging, feeding across session',
                 fontsize=14, fontweight='bold')

    time_axis = np.arange(n_bins) * 0.1

    # Panel 1: Pot occupancy
    ax = axes[0]
    pot_colors = {'Pot-1': '#FF9800', 'Pot-2': '#E53935', 'Pot-3': '#1E88E5', 'Pot-4': '#43A047'}
    for pot_name, color in pot_colors.items():
        signal = np.zeros(n_bins)
        if pot_name in behav:
            p = behav[pot_name]
            signal = np.where(~np.isnan(p) & (p > 0), 1, 0)
        zone_name = f'{pot_name} zone'
        if zone_name in behav:
            pz = behav[zone_name]
            # Zone = 0.5 height, pot = 1.0 height
            signal = np.where(~np.isnan(pz) & (pz > 0) & (signal < 1), 0.5, signal)
        pot_idx = int(pot_name[-1]) - 1
        ax.fill_between(time_axis, pot_idx, pot_idx + signal * 0.9,
                         color=color, alpha=0.6, step='mid')
    ax.set_yticks([0.45, 1.45, 2.45, 3.45])
    ax.set_yticklabels(['Pot-1', 'Pot-2', 'Pot-3', 'Pot-4'])
    ax.set_ylabel('Pot occupancy')
    ax.set_title('Pot visits (filled = on pot, half = in zone)')

    # Panel 2: Digging
    ax = axes[1]
    if 'Digging' in behav:
        dig = behav['Digging']
        dig_clean = np.where(np.isnan(dig), 0, dig)
        ax.fill_between(time_axis, 0, dig_clean, color='brown', alpha=0.7, step='mid')
        # Color-code by which pot
        for s, e, d, pot in dig_events:
            color = pot_colors.get(pot, 'gray')
            ax.axvspan(s * 0.1, e * 0.1, color=color, alpha=0.3)
            ax.text((s + e) / 2 * 0.1, 1.1, pot[-1], ha='center', fontsize=8,
                    color=color, fontweight='bold')
    ax.set_ylabel('Digging')
    ax.set_ylim(-0.1, 1.5)

    # Panel 3: Feeding
    ax = axes[2]
    if 'Feeding' in behav:
        feed = behav['Feeding']
        feed_clean = np.where(np.isnan(feed), 0, feed)
        ax.fill_between(time_axis, 0, feed_clean, color='green', alpha=0.7, step='mid')
        for s, e, d, pot in feed_events:
            color = pot_colors.get(pot, 'gray')
            ax.axvspan(s * 0.1, e * 0.1, color=color, alpha=0.3)
            ax.text((s + e) / 2 * 0.1, 1.1, pot[-1], ha='center', fontsize=8,
                    color=color, fontweight='bold')
    ax.set_ylabel('Feeding')
    ax.set_ylim(-0.1, 1.5)

    # Panel 4: Home, Arena, Distance to Pot-2
    ax = axes[3]
    if 'Home' in behav:
        home = behav['Home']
        home_clean = np.where(np.isnan(home), 0, home)
        ax.fill_between(time_axis, 0, home_clean * 0.5, color='gray', alpha=0.3,
                         step='mid', label='Home')
    if 'Foraging arena' in behav:
        arena = behav['Foraging arena']
        arena_clean = np.where(np.isnan(arena), 0, arena)
        ax.fill_between(time_axis, 0.5, 0.5 + arena_clean * 0.5, color='steelblue',
                         alpha=0.3, step='mid', label='Arena')
    ax.set_ylabel('Location')
    ax.set_ylim(-0.1, 1.2)
    ax.legend(fontsize=8, loc='upper right')

    axes[-1].set_xlabel('Time in session (s)')
    axes[-1].set_xlim(0, n_bins * 0.1)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(f'figures/foraging_events_s{sess_num}.png', dpi=100, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: figures/foraging_events_s{sess_num}.png")

    return {
        'session': sess_num,
        'state': state,
        'pot2_events': pot2_events,
        'pot_events': pot_events,
        'dig_events': dig_events,
        'feed_events': feed_events,
        'transitions': transitions,
    }


def main():
    print("=" * 70)
    print("  FORAGING BEHAVIORAL EVENTS — Sessions 2, 4, 6, 8")
    print("  Pot visits, digging, feeding, strategy shifts")
    print("=" * 70)
    sys.stdout.flush()

    all_results = {}
    for sess in [2, 4, 6, 8]:
        result = process_session(sess)
        if result:
            all_results[sess] = result
        sys.stdout.flush()

    # =====================================================================
    # CROSS-SESSION SUMMARY
    # =====================================================================
    print(f"\n\n{'='*70}")
    print(f"  CROSS-SESSION SUMMARY")
    print(f"{'='*70}")
    print(f"\n  {'Session':<10} {'Pot-2 visits':<14} {'First Pot-2':<12} "
          f"{'Dig events':<12} {'Feed events':<12} {'First dig':<12} {'First feed':<12}")
    print(f"  {'-'*80}")

    for sess in [2, 4, 6, 8]:
        if sess not in all_results:
            continue
        r = all_results[sess]
        n_p2 = len(r['pot2_events'])
        first_p2 = f"{r['pot2_events'][0][0]*0.1:.1f}s" if n_p2 > 0 else "N/A"
        n_dig = len(r['dig_events'])
        n_feed = len(r['feed_events'])
        first_dig = f"{r['dig_events'][0][0]*0.1:.1f}s" if n_dig > 0 else "N/A"
        first_feed = f"{r['feed_events'][0][0]*0.1:.1f}s" if n_feed > 0 else "N/A"
        print(f"  S{sess:<9} {n_p2:<14} {first_p2:<12} {n_dig:<12} {n_feed:<12} "
              f"{first_dig:<12} {first_feed:<12}")

    # Dig location summary
    print(f"\n  Digging locations:")
    for sess in [2, 4, 6, 8]:
        if sess not in all_results:
            continue
        r = all_results[sess]
        pot_digs = {}
        for s, e, d, pot in r['dig_events']:
            pot_digs[pot] = pot_digs.get(pot, 0) + 1
        print(f"    S{sess}: {pot_digs}")

    print("\nDone!")


if __name__ == "__main__":
    main()
