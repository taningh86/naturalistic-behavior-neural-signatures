"""
Detect hesitant explorations from zone and velocity data.

Hesitant exploration: animal leaves Home, enters Ladder/Transition zone
corridor, may briefly enter Foraging arena, then retreats back toward
Home without committing to a task (no Feeding, Digging, or sustained
pot interaction).

Anchored to excursion starts (Home exit). Classifies each excursion
as hesitant or not based on zone sequence and velocity patterns.

Step 1: Characterize zone/velocity signatures during labeled hesitant
exploration bouts in sessions 1-2, then apply detection to sessions 1-8.
"""

import yaml
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

with open("paths.yaml") as f:
    paths_config = yaml.safe_load(f)


def load_behavior_data(session_num):
    """Load all behavior variables for a session, return dict of arrays."""
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


def get_zone_at_time(behav, t):
    """Get list of active zones at time bin t."""
    zones_to_check = ['Home', 'Ladder', 'Transition zone', 'Foraging arena',
                      'Pot-1', 'Pot-2', 'Pot-3', 'Pot-4',
                      'Pot-1 zone', 'Pot-2 zone', 'Pot-3 zone', 'Pot-4 zone',
                      'Arna center', 'Right corner', 'Left corner']
    active = []
    for z in zones_to_check:
        if z in behav:
            v = behav[z]
            if t < len(v) and not np.isnan(v[t]) and v[t] > 0:
                active.append(z)
    return active if active else ['NONE']


def get_simplified_zone(active_zones):
    """Collapse zone list to a single simplified label."""
    # Priority order for simplified labeling
    if 'Home' in active_zones:
        return 'Home'
    if any(p in active_zones for p in ['Pot-1', 'Pot-2', 'Pot-3', 'Pot-4']):
        return 'Pot'
    if 'Arna center' in active_zones:
        return 'Arena-center'
    if any(p in active_zones for p in ['Pot-1 zone', 'Pot-2 zone', 'Pot-3 zone', 'Pot-4 zone']):
        return 'Pot-zone'
    if 'Foraging arena' in active_zones:
        return 'Arena'
    if 'Transition zone' in active_zones:
        return 'Transition'
    if 'Ladder' in active_zones:
        return 'Ladder'
    if any(c in active_zones for c in ['Right corner', 'Left corner']):
        return 'Corner'
    return 'NONE'


def get_zone_sequence(behav, start_bin, end_bin):
    """Get simplified zone sequence between start and end bins."""
    seq = []
    prev = None
    for t in range(start_bin, end_bin):
        active = get_zone_at_time(behav, t)
        simplified = get_simplified_zone(active)
        if simplified != prev:
            seq.append((t, simplified))
            prev = simplified
    return seq


def find_excursions_from_zones(behav, n_bins):
    """Detect excursions from zone data: Home exit to Home return."""
    if 'Home' not in behav:
        return []

    home = behav['Home']
    home_binary = np.zeros(n_bins, dtype=bool)
    for t in range(min(n_bins, len(home))):
        if not np.isnan(home[t]) and home[t] > 0:
            home_binary[t] = True

    # Find home exits and returns
    excursions = []
    in_excursion = False
    exc_start = None

    # Find first home entry to anchor
    first_home = -1
    for t in range(n_bins):
        if home_binary[t]:
            first_home = t
            break

    if first_home < 0:
        return []

    for t in range(first_home, n_bins):
        if not in_excursion:
            # Look for home exit
            if home_binary[t] and t + 1 < n_bins and not home_binary[t + 1]:
                exc_start = t + 1
                in_excursion = True
        else:
            # Look for home return
            if home_binary[t]:
                excursions.append((exc_start, t))
                in_excursion = False

    # If still in excursion at end of session
    if in_excursion:
        excursions.append((exc_start, n_bins - 1))

    return excursions


def compute_excursion_features(behav, start_bin, end_bin):
    """Compute zone and velocity features for an excursion."""
    n_pts = end_bin - start_bin

    # Zone sequence
    zone_seq = get_zone_sequence(behav, start_bin, end_bin)
    zone_names = [z for _, z in zone_seq]

    # Time spent in each zone
    zone_times = {}
    for t in range(start_bin, end_bin):
        active = get_zone_at_time(behav, t)
        z = get_simplified_zone(active)
        zone_times[z] = zone_times.get(z, 0) + 1

    # Farthest zone reached
    zone_order = ['Home', 'Ladder', 'Transition', 'Arena', 'Pot-zone',
                  'Arena-center', 'Pot']
    farthest_idx = 0
    for z in zone_names:
        if z in zone_order:
            farthest_idx = max(farthest_idx, zone_order.index(z))
    farthest = zone_order[farthest_idx]

    # Did the animal reach arena?
    reached_arena = any(z in ['Arena', 'Arena-center', 'Pot-zone', 'Pot']
                        for z in zone_names)

    # Did the animal interact with pots?
    pot_interaction = any(z in ['Pot-zone', 'Pot'] for z in zone_names)

    # Task engagement: Feeding or Digging during this excursion?
    feeding_bins = 0
    digging_bins = 0
    if 'Feeding' in behav:
        feed = behav['Feeding']
        for t in range(start_bin, min(end_bin, len(feed))):
            if not np.isnan(feed[t]) and feed[t] > 0:
                feeding_bins += 1
    if 'Digging' in behav:
        dig = behav['Digging']
        for t in range(start_bin, min(end_bin, len(dig))):
            if not np.isnan(dig[t]) and dig[t] > 0:
                digging_bins += 1

    # Number of direction reversals (zone transitions back toward home)
    reversals = 0
    for i in range(1, len(zone_names)):
        curr_idx = zone_order.index(zone_names[i]) if zone_names[i] in zone_order else 0
        prev_idx = zone_order.index(zone_names[i-1]) if zone_names[i-1] in zone_order else 0
        if curr_idx < prev_idx:
            reversals += 1

    # Velocity stats
    vel_mean = np.nan
    vel_std = np.nan
    vel_min = np.nan
    vel_at_transition = np.nan
    if 'Velocity' in behav:
        vel = behav['Velocity']
        vel_slice = vel[start_bin:min(end_bin, len(vel))]
        valid_vel = vel_slice[~np.isnan(vel_slice)]
        if len(valid_vel) > 0:
            vel_mean = np.mean(valid_vel)
            vel_std = np.std(valid_vel)
            vel_min = np.min(valid_vel)

        # Velocity at transition zone entry (if any)
        for t in range(start_bin, min(end_bin, len(vel))):
            active = get_zone_at_time(behav, t)
            if 'Transition zone' in active and not np.isnan(vel[t]):
                vel_at_transition = vel[t]
                break

    # Duration
    duration = n_pts * 0.1  # seconds

    return {
        'start_bin': start_bin,
        'end_bin': end_bin,
        'start_time': start_bin * 0.1,
        'end_time': end_bin * 0.1,
        'duration': duration,
        'n_zone_transitions': len(zone_seq),
        'farthest_zone': farthest,
        'reached_arena': reached_arena,
        'pot_interaction': pot_interaction,
        'feeding_bins': feeding_bins,
        'digging_bins': digging_bins,
        'reversals': reversals,
        'zone_sequence': ' -> '.join(zone_names),
        'pct_ladder': zone_times.get('Ladder', 0) / n_pts * 100 if n_pts > 0 else 0,
        'pct_transition': zone_times.get('Transition', 0) / n_pts * 100 if n_pts > 0 else 0,
        'pct_arena': sum(zone_times.get(z, 0) for z in ['Arena', 'Arena-center', 'Pot-zone', 'Pot']) / n_pts * 100 if n_pts > 0 else 0,
        'vel_mean': vel_mean,
        'vel_std': vel_std,
        'vel_min': vel_min,
        'vel_at_transition': vel_at_transition,
    }


def check_labeled_hesitant_overlap(behav, start_bin, end_bin):
    """Check if this excursion overlaps with labeled 'Hesitant exploration'."""
    if 'Hesitant exploration' not in behav:
        return 0
    hes = behav['Hesitant exploration']
    count = 0
    for t in range(start_bin, min(end_bin, len(hes))):
        if not np.isnan(hes[t]) and hes[t] > 0:
            count += 1
    return count


def main():
    print("=" * 80)
    print("  Hesitant Exploration Detection from Zone/Velocity Data")
    print("  Mouse01, Coordinate 1, Sessions 1-8")
    print("=" * 80)

    all_excursions = []

    for snum in range(1, 9):
        state = "Fed" if snum <= 4 else "Fasted"
        phase = "Exploration" if snum % 2 == 1 else "Foraging"

        behav, n_bins = load_behavior_data(snum)
        if behav is None:
            print(f"\nSession {snum}: No behavior data")
            continue

        duration_s = n_bins * 0.1
        print(f"\n{'='*80}")
        print(f"  Session {snum} ({state}, {phase}) — {duration_s:.0f}s, {n_bins} bins")
        print(f"{'='*80}")
        sys.stdout.flush()

        # Find excursions from zone data
        excursions = find_excursions_from_zones(behav, n_bins)
        print(f"  {len(excursions)} excursions detected from Home exits/returns")

        for ei, (s, e) in enumerate(excursions):
            features = compute_excursion_features(behav, s, e)
            features['session'] = snum
            features['state'] = state
            features['phase'] = phase
            features['excursion_idx'] = ei + 1

            # Check overlap with labeled hesitant exploration
            hes_overlap = check_labeled_hesitant_overlap(behav, s, e)
            features['labeled_hesitant_bins'] = hes_overlap
            features['labeled_hesitant'] = hes_overlap > 0

            all_excursions.append(features)

        # Print summary for this session
        exc_df = pd.DataFrame([f for f in all_excursions if f['session'] == snum])
        n_labeled_hes = exc_df['labeled_hesitant'].sum() if len(exc_df) > 0 else 0

        print(f"  Labeled hesitant: {n_labeled_hes} excursions")
        print(f"\n  {'Idx':<5} {'Time':<14} {'Dur':<7} {'Farthest':<14} "
              f"{'Revrsl':<7} {'Feed':<5} {'Dig':<5} {'VelMn':<7} {'Hes?':<5} "
              f"{'Zone Sequence'}")
        print(f"  {'-'*120}")

        for _, row in exc_df.iterrows():
            hes_mark = "YES" if row['labeled_hesitant'] else ""
            seq = row['zone_sequence']
            if len(seq) > 55:
                seq = seq[:55] + "..."
            print(f"  {int(row['excursion_idx']):<5} "
                  f"{row['start_time']:>6.1f}-{row['end_time']:<6.1f} "
                  f"{row['duration']:<7.1f} {row['farthest_zone']:<14} "
                  f"{int(row['reversals']):<7} {int(row['feeding_bins']):<5} "
                  f"{int(row['digging_bins']):<5} "
                  f"{row['vel_mean']:<7.1f} {hes_mark:<5} "
                  f"{seq}")
        sys.stdout.flush()

    # Save all excursion features
    full_df = pd.DataFrame(all_excursions)
    outpath = Path("data") / "excursion_features_all_sessions.csv"
    full_df.to_csv(outpath, index=False, float_format='%.3f')
    print(f"\n\nSaved: {outpath}")
    print(f"Total excursions: {len(full_df)}")

    # Summary stats
    print(f"\n{'='*80}")
    print(f"  Summary across sessions")
    print(f"{'='*80}")
    for snum in range(1, 9):
        sdf = full_df[full_df['session'] == snum]
        n_exc = len(sdf)
        n_hes = sdf['labeled_hesitant'].sum()
        n_arena = sdf['reached_arena'].sum()
        n_feed = (sdf['feeding_bins'] > 0).sum()
        n_dig = (sdf['digging_bins'] > 0).sum()
        n_rev = sdf[sdf['reversals'] > 0].shape[0]

        # Timing of hesitant excursions
        if n_hes > 0:
            hes_times = sdf[sdf['labeled_hesitant']]['start_time'].values
            hes_str = f"  hes at t={hes_times.min():.0f}-{hes_times.max():.0f}s"
        else:
            hes_str = ""

        state = sdf['state'].iloc[0] if len(sdf) > 0 else "?"
        phase = sdf['phase'].iloc[0] if len(sdf) > 0 else "?"
        print(f"  S{snum} ({state[:3]},{phase[:3]}): {n_exc:3d} exc, "
              f"{n_hes:2d} labeled-hes, {n_arena:3d} reached-arena, "
              f"{n_feed:2d} fed, {n_dig:2d} dug, {n_rev:3d} with-reversals{hes_str}")

    # Characterize labeled hesitant vs non-hesitant
    labeled = full_df[full_df['labeled_hesitant']]
    unlabeled = full_df[~full_df['labeled_hesitant']]

    if len(labeled) > 0:
        print(f"\n  Labeled hesitant excursions (n={len(labeled)}):")
        print(f"    Duration:    {labeled['duration'].median():.1f}s median "
              f"(range {labeled['duration'].min():.1f}-{labeled['duration'].max():.1f})")
        print(f"    Reversals:   {labeled['reversals'].median():.1f} median "
              f"(range {labeled['reversals'].min():.0f}-{labeled['reversals'].max():.0f})")
        print(f"    Vel mean:    {labeled['vel_mean'].median():.1f} median")
        print(f"    Reached arena: {labeled['reached_arena'].sum()}/{len(labeled)}")
        print(f"    Pot interact:  {labeled['pot_interaction'].sum()}/{len(labeled)}")
        print(f"    Feeding:       {(labeled['feeding_bins']>0).sum()}/{len(labeled)}")
        print(f"    Farthest zones: {labeled['farthest_zone'].value_counts().to_dict()}")

        print(f"\n  Non-hesitant excursions (n={len(unlabeled)}):")
        print(f"    Duration:    {unlabeled['duration'].median():.1f}s median "
              f"(range {unlabeled['duration'].min():.1f}-{unlabeled['duration'].max():.1f})")
        print(f"    Reversals:   {unlabeled['reversals'].median():.1f} median")
        print(f"    Vel mean:    {unlabeled['vel_mean'].median():.1f} median")
        print(f"    Reached arena: {unlabeled['reached_arena'].sum()}/{len(unlabeled)}")
        print(f"    Feeding:       {(unlabeled['feeding_bins']>0).sum()}/{len(unlabeled)}")

    print("\nDone!")


if __name__ == "__main__":
    main()
