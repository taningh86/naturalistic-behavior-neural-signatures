"""
Dual-Probe: Behavioral Approach Event Extraction
=================================================
Extracts approach trajectories defined by straight paths to target zones:

1. Pot approaches — direct path ending at pot (P1/P2/P3/P4), not pot-zone
2. Ladder from Home — leave Home, arrive at Ladder
3. Ladder from Arena — leave arena-side, arrive at Ladder (retreat)
4. Pre-dig approach — direct path leading to digging bout onset
5. Pre-feed approach — direct path leading to feeding bout onset

Approach window = zone-to-zone straight path (not fixed time window).
Validity: no detour zones, tortuosity < threshold, minimum duration.

Output: CSV with per-event metadata + .npy with x,y trajectories for neural alignment.
"""

import yaml
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import warnings

warnings.filterwarnings('ignore')

with open("paths.yaml") as f:
    cfg = yaml.safe_load(f)

# ---- Constants ----
MAX_TORTUOSITY = 1.8       # path_length / straight_line < this (direct paths only)
MIN_DURATION = 0.2         # seconds — exclude trivially short approaches
MIN_STRAIGHT_DIST = 1.0    # cm — exclude approaches where origin ~= target
MAX_DURATION = 3.0         # seconds — direct one-shot approaches only

SKIP_SESSIONS = {23, 24}

# Zone classification
HOME_ZONES = {'H', 'HCL', 'HCR'}
LADDER_ZONES = {'L'}
TRANSITION_ZONES = {'T'}
ARENA_ZONES = {'FA', 'CA'}
POT_ZONES = {'P1z', 'P2z', 'P3z', 'P4z'}  # pot surroundings (transit)
POT_TARGETS = {'P1', 'P2', 'P3', 'P4'}     # actual pots (targets)
ARENA_SIDE = ARENA_ZONES | POT_ZONES | POT_TARGETS | TRANSITION_ZONES

# For pot approaches: detour zones that invalidate the approach
POT_APPROACH_DETOUR = HOME_ZONES | LADDER_ZONES  # can't visit Home or Ladder en route

# Zone priority for building zone array (same as other scripts)
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
    'Lever choice zone': 'LCZ', 'Lever Zone': 'LZ',
    'Pot choice zone': 'PCZ', 'Sand Pots Zone': 'SPZ',
    'Lever1 food zone': 'L1F', 'Lever2 food zone': 'L2F',
}

# Manual behavior columns
BEHAVIOR_COLS = ['Feeding', 'Digging sand']

sessions_cfg = cfg["double_probe"]["coordinates_1"]["mouse01"]["sessions"]


# ---- Helper functions ----

def load_behavior_xlsx(path):
    """Load dual-probe behavior xlsx. Returns time, x, y, velocity, zones, behaviors."""
    df = pd.read_excel(path, header=None)
    col_names = df.iloc[34].tolist()
    data = df.iloc[36:].reset_index(drop=True)
    data.columns = col_names

    time_vals = pd.to_numeric(data['Recording time'], errors='coerce').values
    x = pd.to_numeric(data['X center'], errors='coerce').values
    y = pd.to_numeric(data['Y center'], errors='coerce').values
    vel = pd.to_numeric(data['Velocity(Center-point)'], errors='coerce').values
    vel = np.nan_to_num(vel, nan=0.0)
    vel = np.clip(vel, 0, None)

    # Fill NaN in x,y with interpolation
    valid = ~np.isnan(x)
    if valid.sum() > 10:
        x = np.interp(np.arange(len(x)), np.where(valid)[0], x[valid])
        y = np.interp(np.arange(len(y)), np.where(~np.isnan(y))[0], y[~np.isnan(y)])

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

    # Manual labels
    behaviors = {}
    for bcol in BEHAVIOR_COLS:
        if bcol in col_names:
            bvals = pd.to_numeric(data[bcol], errors='coerce').values
            bvals = np.nan_to_num(bvals, nan=0.0)
            behaviors[bcol] = bvals
        else:
            behaviors[bcol] = np.zeros(len(time_vals))

    return time_vals, x, y, vel, zones, behaviors


def compute_path_length(x_seg, y_seg):
    """Cumulative path length from x,y arrays."""
    dx = np.diff(x_seg)
    dy = np.diff(y_seg)
    return np.sum(np.sqrt(dx**2 + dy**2))


def compute_straight_dist(x_seg, y_seg):
    """Straight-line distance from first to last point."""
    return np.sqrt((x_seg[-1] - x_seg[0])**2 + (y_seg[-1] - y_seg[0])**2)


def find_zone_transitions(zones):
    """Find indices where zone changes. Returns list of (index, from_zone, to_zone)."""
    transitions = []
    for i in range(1, len(zones)):
        if zones[i] != zones[i-1]:
            transitions.append((i, zones[i-1], zones[i]))
    return transitions


def get_zone_category(zone):
    """Classify a zone into a category."""
    if zone in HOME_ZONES:
        return 'home'
    if zone in LADDER_ZONES:
        return 'ladder'
    if zone in TRANSITION_ZONES:
        return 'transition'
    if zone in ARENA_ZONES:
        return 'arena'
    if zone in POT_ZONES:
        return 'pot_zone'
    if zone in POT_TARGETS:
        return 'pot'
    return 'other'


POT_ZONE_OF = {'P1': 'P1z', 'P2': 'P2z', 'P3': 'P3z', 'P4': 'P4z'}


def allowed_pot_zones(origin, target):
    """Pot zones allowed along the path: only those belonging to origin or target pot."""
    allowed = set()
    if origin in POT_TARGETS:
        allowed.add(POT_ZONE_OF[origin])
    if target in POT_TARGETS:
        allowed.add(POT_ZONE_OF[target])
    return allowed


def path_has_other_pot_detour(zones_seg, origin, target):
    """True if the path visits any pot zone or pot target not equal to origin/target."""
    allowed = allowed_pot_zones(origin, target)
    allowed_targets = {z for z in [origin, target] if z in POT_TARGETS}
    seen = set(zones_seg)
    # any pot zone not in allowed
    for z in seen:
        if z in POT_ZONES and z not in allowed:
            return True
        if z in POT_TARGETS and z not in allowed_targets:
            return True
    return False


def window_contains_behavior(beh_arr, start_idx, end_idx):
    """True if behavior array is active anywhere in [start_idx, end_idx]."""
    if end_idx <= start_idx:
        return False
    return np.any(beh_arr[start_idx:end_idx] > 0.5)


def find_approach_start_simple(zones, target_entry_idx, origin_anchors, detour_zones=None):
    """Walk backward from target_entry_idx to find approach start.

    Approach starts when the mouse LEAVES an anchor zone (origin_anchors)
    and begins its transit toward the target. Transit zones (FA, CA, pot zones)
    are just the path — we skip through them.

    Returns: (start_idx, origin_zone) or (None, None).
    """
    if detour_zones is None:
        detour_zones = set()

    for i in range(target_entry_idx - 1, -1, -1):
        z = zones[i]
        # If we hit a detour zone, this approach is invalid
        if z in detour_zones:
            return None, None
        # If we find an anchor zone, the approach started when mouse left it
        if z in origin_anchors:
            origin_zone = z
            # Scan forward to find first frame OUTSIDE this anchor
            for j in range(i + 1, target_entry_idx + 1):
                if zones[j] not in origin_anchors:
                    return j, origin_zone
            return None, None
    return None, None


def find_behavior_onsets(behavior_arr, min_gap_bins=5):
    """Find onset indices of behavior bouts (0->1 transitions).
    Merge bouts separated by < min_gap_bins."""
    active = behavior_arr > 0.5
    onsets = []
    offsets = []

    in_bout = False
    for i in range(len(active)):
        if active[i] and not in_bout:
            onsets.append(i)
            in_bout = True
        elif not active[i] and in_bout:
            offsets.append(i)
            in_bout = False
    if in_bout:
        offsets.append(len(active))

    # Merge bouts with small gaps
    if len(onsets) < 2:
        return onsets

    merged_onsets = [onsets[0]]
    merged_offsets = []
    for i in range(1, len(onsets)):
        if onsets[i] - offsets[i-1] < min_gap_bins:
            continue  # merge: don't start new bout
        else:
            merged_offsets.append(offsets[i-1])
            merged_onsets.append(onsets[i])
    merged_offsets.append(offsets[-1])

    return merged_onsets


def validate_approach(time_vals, x, y, start_idx, end_idx):
    """Validate approach: duration, distance, tortuosity. Returns dict or None."""
    duration = time_vals[end_idx] - time_vals[start_idx]
    if duration < MIN_DURATION or duration > MAX_DURATION:
        return None

    x_seg = x[start_idx:end_idx+1]
    y_seg = y[start_idx:end_idx+1]

    path_len = compute_path_length(x_seg, y_seg)
    straight_dist = compute_straight_dist(x_seg, y_seg)

    if straight_dist < MIN_STRAIGHT_DIST:
        return None

    tortuosity = path_len / straight_dist if straight_dist > 0 else 999
    if tortuosity > MAX_TORTUOSITY:
        return None

    mean_vel = np.nanmean(np.sqrt(np.diff(x_seg)**2 + np.diff(y_seg)**2) /
                          np.diff(time_vals[start_idx:end_idx+1]))

    return {
        'duration': duration,
        'path_length': path_len,
        'straight_line_dist': straight_dist,
        'tortuosity': tortuosity,
        'mean_velocity': mean_vel,
    }


# ========================================================================
# MAIN EXTRACTION
# ========================================================================
print("=" * 110)
print("DUAL-PROBE: BEHAVIORAL APPROACH EVENT EXTRACTION")
print("=" * 110)

session_meta = {}
for skey, sval in sessions_cfg.items():
    snum = int(skey.split('_')[1])
    if snum in SKIP_SESSIONS:
        continue
    behav = sval.get('behavior')
    if not behav or not Path(behav).exists():
        continue
    session_meta[snum] = {
        'state': sval['state'], 'phase': sval['phase'],
        'behavior': behav,
    }

print(f"Found {len(session_meta)} sessions with behavior data\n")

all_events = []
all_trajectories = {}  # keyed by event index

for snum in sorted(session_meta.keys()):
    meta = session_meta[snum]
    state, phase = meta['state'], meta['phase']
    print(f"  S{snum} ({state}/{phase}): loading...", end='', flush=True)

    time_vals, x, y, vel, zones, behaviors = load_behavior_xlsx(meta['behavior'])
    dt = np.median(np.diff(time_vals))
    n_samples = len(time_vals)
    print(f" {n_samples} samples, dt={dt:.3f}s", end='')

    transitions = find_zone_transitions(zones)
    n_pot = 0
    n_ladder_home = 0
    n_ladder_arena = 0
    n_pre_dig = 0
    n_pre_feed = 0

    # ------------------------------------------------------------------
    # 1. POT APPROACHES
    # Target = actual pot (P1/P2/P3/P4), not pot zone
    # Origin anchors = Home, Ladder, Transition, or another POT (not pot zone)
    # Transit = FA, CA, pot zones (P1z etc.) — just passing through
    # Detour = none for now (anchor-to-anchor is already direct)
    # ------------------------------------------------------------------
    # Anchor zones for pot approach origins: where the mouse was "at" before approaching
    POT_ORIGIN_ANCHORS = HOME_ZONES | LADDER_ZONES | TRANSITION_ZONES | POT_TARGETS

    for tr_idx, from_z, to_z in transitions:
        if to_z not in POT_TARGETS:
            continue

        # Find approach start: walk backward to last anchor zone
        start_idx, origin_zone = find_approach_start_simple(
            zones, tr_idx, POT_ORIGIN_ANCHORS - {to_z})

        if start_idx is None:
            continue

        # Check for detours: did the mouse visit Home, Ladder, or other pot zones?
        path_seg = zones[start_idx:tr_idx]
        path_zones = set(path_seg)
        if path_zones & (HOME_ZONES | LADDER_ZONES):
            continue
        if path_has_other_pot_detour(path_seg, origin_zone, to_z):
            continue
        # No intervening dig/feed mid-approach
        if window_contains_behavior(behaviors['Digging sand'], start_idx, tr_idx):
            continue
        if window_contains_behavior(behaviors['Feeding'], start_idx, tr_idx):
            continue

        metrics = validate_approach(time_vals, x, y, start_idx, tr_idx)
        if metrics is None:
            continue

        evt_idx = len(all_events)
        all_events.append({
            'event_idx': evt_idx,
            'session': snum, 'state': state, 'phase': phase,
            'approach_type': 'pot_approach',
            'target_zone': to_z,
            'origin_zone': origin_zone,
            't_start': time_vals[start_idx],
            't_end': time_vals[tr_idx],
            'start_idx': start_idx,
            'end_idx': tr_idx,
            **metrics,
        })
        all_trajectories[evt_idx] = {
            'x': x[start_idx:tr_idx+1].copy(),
            'y': y[start_idx:tr_idx+1].copy(),
            'time': time_vals[start_idx:tr_idx+1].copy(),
        }
        n_pot += 1

    # ------------------------------------------------------------------
    # 2. LADDER FROM HOME
    # The mouse is in Home and moves to Ladder (to enter the arena).
    # Home and Ladder are adjacent, so the H->L transition itself is trivial.
    # Use the last LADDER_HOME_FRAMES frames in Home before the transition
    # to keep approach windows comparable across events.
    # ------------------------------------------------------------------
    LADDER_HOME_FRAMES = 5

    for tr_idx, from_z, to_z in transitions:
        if to_z not in LADDER_ZONES:
            continue
        # Must come from Home side (from_z is the zone just before Ladder)
        if from_z not in HOME_ZONES:
            continue

        # Start = 5 frames before Ladder entry (clamped to stay in Home)
        start_idx = max(0, tr_idx - LADDER_HOME_FRAMES)
        # Verify all frames in this window are Home zones
        if not all(zones[i] in HOME_ZONES for i in range(start_idx, tr_idx)):
            # If not all Home, find first Home frame in the window
            for i in range(start_idx, tr_idx):
                if zones[i] in HOME_ZONES:
                    start_idx = i
                    break
            else:
                continue

        if start_idx >= tr_idx:
            continue

        metrics = validate_approach(time_vals, x, y, start_idx, tr_idx)
        if metrics is None:
            continue

        origin_zone = zones[start_idx]

        evt_idx = len(all_events)
        all_events.append({
            'event_idx': evt_idx,
            'session': snum, 'state': state, 'phase': phase,
            'approach_type': 'ladder_from_home',
            'target_zone': 'L',
            'origin_zone': origin_zone,
            't_start': time_vals[start_idx],
            't_end': time_vals[tr_idx],
            'start_idx': start_idx,
            'end_idx': tr_idx,
            **metrics,
        })
        all_trajectories[evt_idx] = {
            'x': x[start_idx:tr_idx+1].copy(),
            'y': y[start_idx:tr_idx+1].copy(),
            'time': time_vals[start_idx:tr_idx+1].copy(),
        }
        n_ladder_home += 1

    # ------------------------------------------------------------------
    # 3. LADDER FROM ARENA (retreat)
    # The mouse comes from the arena side back to Ladder.
    # T (Transition) is adjacent to L, so T->L is trivially short.
    # Use only POT_TARGETS as origin anchors — this captures the full
    # path from a pot through arena zones / T to Ladder.
    # ------------------------------------------------------------------
    for tr_idx, from_z, to_z in transitions:
        if to_z not in LADDER_ZONES:
            continue

        # Must come from arena side (not directly from Home)
        if from_z in HOME_ZONES:
            continue

        # Origin = last pot the mouse was at (not T — too close to L)
        start_idx, origin_zone = find_approach_start_simple(
            zones, tr_idx, POT_TARGETS,
            detour_zones=HOME_ZONES)  # can't pass through Home

        if start_idx is None:
            continue

        # Direct retreat: no other pot zones visited, no dig/feed mid-path
        path_seg = zones[start_idx:tr_idx]
        if path_has_other_pot_detour(path_seg, origin_zone, 'L'):
            continue
        if window_contains_behavior(behaviors['Digging sand'], start_idx, tr_idx):
            continue
        if window_contains_behavior(behaviors['Feeding'], start_idx, tr_idx):
            continue

        metrics = validate_approach(time_vals, x, y, start_idx, tr_idx)
        if metrics is None:
            continue

        evt_idx = len(all_events)
        all_events.append({
            'event_idx': evt_idx,
            'session': snum, 'state': state, 'phase': phase,
            'approach_type': 'ladder_from_arena',
            'target_zone': 'L',
            'origin_zone': origin_zone,
            't_start': time_vals[start_idx],
            't_end': time_vals[tr_idx],
            'start_idx': start_idx,
            'end_idx': tr_idx,
            **metrics,
        })
        all_trajectories[evt_idx] = {
            'x': x[start_idx:tr_idx+1].copy(),
            'y': y[start_idx:tr_idx+1].copy(),
            'time': time_vals[start_idx:tr_idx+1].copy(),
        }
        n_ladder_arena += 1

    # ------------------------------------------------------------------
    # 4. PRE-DIG APPROACHES
    # Find dig onsets, then trace back to the approach into the pot where digging occurs
    # The approach = from last anchor zone to the dig onset
    # ------------------------------------------------------------------
    dig_onsets = find_behavior_onsets(behaviors.get('Digging sand', np.zeros(n_samples)))
    PRE_ANCHORS = HOME_ZONES | LADDER_ZONES | TRANSITION_ZONES | POT_TARGETS

    for onset_idx in dig_onsets:
        # What pot is the mouse in at dig onset?
        dig_zone = zones[onset_idx]
        if dig_zone not in POT_TARGETS:
            found = False
            for offset in range(-5, 6):
                check_idx = onset_idx + offset
                if 0 <= check_idx < n_samples and zones[check_idx] in POT_TARGETS:
                    dig_zone = zones[check_idx]
                    found = True
                    break
            if not found:
                continue

        # Walk backward from onset to find the approach start
        # Origin = last anchor zone before reaching this pot (excluding the pot itself)
        start_idx, origin_zone = find_approach_start_simple(
            zones, onset_idx, PRE_ANCHORS - {dig_zone})

        if start_idx is None:
            continue

        # Direct approach only: no other pots, no mid-path dig/feed
        path_seg = zones[start_idx:onset_idx]
        if path_has_other_pot_detour(path_seg, origin_zone, dig_zone):
            continue
        if window_contains_behavior(behaviors['Digging sand'], start_idx, onset_idx):
            continue
        if window_contains_behavior(behaviors['Feeding'], start_idx, onset_idx):
            continue

        metrics = validate_approach(time_vals, x, y, start_idx, onset_idx)
        if metrics is None:
            continue

        evt_idx = len(all_events)
        all_events.append({
            'event_idx': evt_idx,
            'session': snum, 'state': state, 'phase': phase,
            'approach_type': 'pre_dig',
            'target_zone': dig_zone,
            'origin_zone': origin_zone,
            't_start': time_vals[start_idx],
            't_end': time_vals[onset_idx],
            'start_idx': start_idx,
            'end_idx': onset_idx,
            **metrics,
        })
        all_trajectories[evt_idx] = {
            'x': x[start_idx:onset_idx+1].copy(),
            'y': y[start_idx:onset_idx+1].copy(),
            'time': time_vals[start_idx:onset_idx+1].copy(),
        }
        n_pre_dig += 1

    # ------------------------------------------------------------------
    # 5. PRE-FEED APPROACHES
    # Find feeding onsets, trace back to approach into the pot where feeding occurs
    # ------------------------------------------------------------------
    feed_onsets = find_behavior_onsets(behaviors.get('Feeding', np.zeros(n_samples)))

    for onset_idx in feed_onsets:
        feed_zone = zones[onset_idx]
        if feed_zone not in POT_TARGETS:
            found = False
            for offset in range(-5, 6):
                check_idx = onset_idx + offset
                if 0 <= check_idx < n_samples and zones[check_idx] in POT_TARGETS:
                    feed_zone = zones[check_idx]
                    found = True
                    break
            if not found:
                continue

        start_idx, origin_zone = find_approach_start_simple(
            zones, onset_idx, PRE_ANCHORS - {feed_zone})

        if start_idx is None:
            continue

        path_seg = zones[start_idx:onset_idx]
        if path_has_other_pot_detour(path_seg, origin_zone, feed_zone):
            continue
        if window_contains_behavior(behaviors['Digging sand'], start_idx, onset_idx):
            continue
        if window_contains_behavior(behaviors['Feeding'], start_idx, onset_idx):
            continue

        metrics = validate_approach(time_vals, x, y, start_idx, onset_idx)
        if metrics is None:
            continue

        evt_idx = len(all_events)
        all_events.append({
            'event_idx': evt_idx,
            'session': snum, 'state': state, 'phase': phase,
            'approach_type': 'pre_feed',
            'target_zone': feed_zone,
            'origin_zone': origin_zone,
            't_start': time_vals[start_idx],
            't_end': time_vals[onset_idx],
            'start_idx': start_idx,
            'end_idx': onset_idx,
            **metrics,
        })
        all_trajectories[evt_idx] = {
            'x': x[start_idx:onset_idx+1].copy(),
            'y': y[start_idx:onset_idx+1].copy(),
            'time': time_vals[start_idx:onset_idx+1].copy(),
        }
        n_pre_feed += 1

    print(f" | pot={n_pot}, lad_home={n_ladder_home}, lad_arena={n_ladder_arena}, "
          f"pre_dig={n_pre_dig}, pre_feed={n_pre_feed}")

# ========================================================================
# SAVE
# ========================================================================
df = pd.DataFrame(all_events)
df.to_csv("data/dp_approach_events.csv", index=False)
print(f"\nSaved data/dp_approach_events.csv ({len(df)} rows)")

# Save trajectories as numpy archive
np.savez_compressed("data/dp_approach_trajectories.npz", **{
    str(k): np.column_stack([v['time'], v['x'], v['y']])
    for k, v in all_trajectories.items()
})
print(f"Saved data/dp_approach_trajectories.npz ({len(all_trajectories)} trajectories)")

# ========================================================================
# SUMMARY
# ========================================================================
print("\n" + "=" * 110)
print("APPROACH EVENT SUMMARY")
print("=" * 110)

for atype in ['pot_approach', 'ladder_from_home', 'ladder_from_arena', 'pre_dig', 'pre_feed']:
    sub = df[df['approach_type'] == atype]
    if len(sub) == 0:
        print(f"\n  {atype}: 0 events")
        continue
    print(f"\n  {atype}: {len(sub)} events")
    print(f"    Duration: {sub['duration'].mean():.2f}s (median {sub['duration'].median():.2f}s, "
          f"range [{sub['duration'].min():.2f}, {sub['duration'].max():.2f}])")
    print(f"    Path length: {sub['path_length'].mean():.1f}cm (median {sub['path_length'].median():.1f}cm)")
    print(f"    Straight dist: {sub['straight_line_dist'].mean():.1f}cm (median {sub['straight_line_dist'].median():.1f}cm)")
    print(f"    Tortuosity: {sub['tortuosity'].mean():.2f} (median {sub['tortuosity'].median():.2f})")
    print(f"    Mean velocity: {sub['mean_velocity'].mean():.1f}cm/s")

    # Per state
    for st in ['fed', 'fasted', 'fed-HFD']:
        st_sub = sub[sub['state'] == st]
        if len(st_sub) > 0:
            print(f"    {st:>8s}: {len(st_sub):>4d} events, dur={st_sub['duration'].mean():.2f}s, "
                  f"tort={st_sub['tortuosity'].mean():.2f}")

    # Per target (for pot approaches)
    if atype in ['pot_approach', 'pre_dig', 'pre_feed']:
        for pot in ['P1', 'P2', 'P3', 'P4']:
            p_sub = sub[sub['target_zone'] == pot]
            if len(p_sub) > 0:
                print(f"    -> {pot}: {len(p_sub)} events")

# Per-session counts
print("\n  Per-session counts:")
print(f"    {'Session':<18s}  {'pot':>5s}  {'lad_H':>5s}  {'lad_A':>5s}  {'dig':>5s}  {'feed':>5s}  {'total':>6s}")
for snum in sorted(session_meta.keys()):
    s_df = df[df['session'] == snum]
    counts = {}
    for atype in ['pot_approach', 'ladder_from_home', 'ladder_from_arena', 'pre_dig', 'pre_feed']:
        counts[atype] = len(s_df[s_df['approach_type'] == atype])
    meta = session_meta[snum]
    print(f"    S{snum:<2d} ({meta['state']:<7s}/{meta['phase']:<4s})  "
          f"{counts['pot_approach']:5d}  {counts['ladder_from_home']:5d}  "
          f"{counts['ladder_from_arena']:5d}  {counts['pre_dig']:5d}  "
          f"{counts['pre_feed']:5d}  {len(s_df):6d}")

# ========================================================================
# FIGURES
# ========================================================================
outdir = Path("figures")
outdir.mkdir(exist_ok=True)

# Figure 1: Summary bar chart — event counts by type and state
STATE_ORDER = ['fed', 'fasted', 'fed-HFD']
STATE_LABELS = {'fed': 'Fed', 'fasted': 'Fasted', 'fed-HFD': 'HFD'}
STATE_COLORS = {'fed': 'tab:blue', 'fasted': 'tab:red', 'fed-HFD': 'tab:purple'}
APPROACH_LABELS = {
    'pot_approach': 'Pot\napproach',
    'ladder_from_home': 'Ladder\nfrom Home',
    'ladder_from_arena': 'Ladder\nfrom Arena',
    'pre_dig': 'Pre-dig\napproach',
    'pre_feed': 'Pre-feed\napproach',
}

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Panel A: counts per session by state
ax = axes[0]
atypes = list(APPROACH_LABELS.keys())
x_pos = np.arange(len(atypes))
width = 0.25
for si, st in enumerate(STATE_ORDER):
    counts = []
    for atype in atypes:
        sub = df[(df['approach_type'] == atype) & (df['state'] == st)]
        n_sessions = len(df[df['state'] == st]['session'].unique())
        counts.append(len(sub) / max(n_sessions, 1))
    ax.bar(x_pos + si * width, counts, width, label=STATE_LABELS[st],
           color=STATE_COLORS[st], alpha=0.7, edgecolor='black', linewidth=0.5)

ax.set_xticks(x_pos + width)
ax.set_xticklabels([APPROACH_LABELS[a] for a in atypes], fontsize=10)
ax.set_ylabel('Events per session', fontsize=13)
ax.set_title('Approach Events by Type & State', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.tick_params(labelsize=10)

# Panel B: duration distributions by type
ax = axes[1]
type_colors = {'pot_approach': 'tab:green', 'ladder_from_home': 'tab:orange',
               'ladder_from_arena': 'tab:cyan', 'pre_dig': 'tab:brown', 'pre_feed': 'tab:red'}
box_data = []
box_labels = []
box_colors = []
for atype in atypes:
    sub = df[df['approach_type'] == atype]
    if len(sub) > 0:
        box_data.append(sub['duration'].values)
        box_labels.append(APPROACH_LABELS[atype].replace('\n', ' '))
        box_colors.append(type_colors[atype])

if box_data:
    bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True, widths=0.5)
    for i, color in enumerate(box_colors):
        bp['boxes'][i].set_facecolor(color)
        bp['boxes'][i].set_alpha(0.5)
    ax.set_ylabel('Duration (s)', fontsize=13)
    ax.set_title('Approach Duration by Type', fontsize=14, fontweight='bold')
    ax.tick_params(labelsize=9)

plt.suptitle('Behavioral Approach Events', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(outdir / "dp_approach_events_summary.png", dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved figures/dp_approach_events_summary.png")

# Figure 2: Tortuosity by type and state
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel A: tortuosity by approach type
ax = axes[0]
for si, st in enumerate(STATE_ORDER):
    tort_means = []
    tort_sems = []
    for atype in atypes:
        sub = df[(df['approach_type'] == atype) & (df['state'] == st)]
        if len(sub) > 0:
            tort_means.append(sub['tortuosity'].mean())
            tort_sems.append(sub['tortuosity'].sem())
        else:
            tort_means.append(0)
            tort_sems.append(0)
    ax.bar(x_pos + si * width, tort_means, width, yerr=tort_sems,
           label=STATE_LABELS[st], color=STATE_COLORS[st], alpha=0.7,
           edgecolor='black', linewidth=0.5, capsize=3)

ax.set_xticks(x_pos + width)
ax.set_xticklabels([APPROACH_LABELS[a] for a in atypes], fontsize=10)
ax.set_ylabel('Tortuosity (path/straight)', fontsize=13)
ax.set_title('Approach Tortuosity', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.tick_params(labelsize=10)

# Panel B: velocity by type and state
ax = axes[1]
for si, st in enumerate(STATE_ORDER):
    vel_means = []
    vel_sems = []
    for atype in atypes:
        sub = df[(df['approach_type'] == atype) & (df['state'] == st)]
        if len(sub) > 0:
            vel_means.append(sub['mean_velocity'].mean())
            vel_sems.append(sub['mean_velocity'].sem())
        else:
            vel_means.append(0)
            vel_sems.append(0)
    ax.bar(x_pos + si * width, vel_means, width, yerr=vel_sems,
           label=STATE_LABELS[st], color=STATE_COLORS[st], alpha=0.7,
           edgecolor='black', linewidth=0.5, capsize=3)

ax.set_xticks(x_pos + width)
ax.set_xticklabels([APPROACH_LABELS[a] for a in atypes], fontsize=10)
ax.set_ylabel('Mean velocity (cm/s)', fontsize=13)
ax.set_title('Approach Velocity', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.tick_params(labelsize=10)

plt.suptitle('Approach Kinematics', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(outdir / "dp_approach_events_kinematics.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved figures/dp_approach_events_kinematics.png")

print("\nDone.")
