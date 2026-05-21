"""
Excursion Detection from EthoVision Behavior Data
==================================================
Identifies home-to-home excursions for each session.

Zone order (home to arena):
    Home → Ladder → Transition Zone → Foraging Arena

Excursion logic:
    - Start: mouse exits Home zone (Home goes 1→0)
    - End: mouse re-enters Home zone (Home goes 0→1), or session ends
    - Complete: mouse entered the Foraging Arena during the excursion
    - Incomplete: mouse only reached Transition Zone or Ladder (never entered Arena)

If session doesn't start at Home, first excursion begins from the
first Home entry.

Output: one CSV per session in data/ with columns:
    excursion_id, start_time, end_time, duration, label, farthest_zone
"""

import yaml
import numpy as np
import pandas as pd
from pathlib import Path

with open("paths.yaml") as f:
    paths_config = yaml.safe_load(f)

SESSION_INFO = {
    1: {'state': 'Fed', 'phase': 'Exploration'},
    2: {'state': 'Fed', 'phase': 'Foraging'},
    3: {'state': 'Fed', 'phase': 'Exploration'},
    4: {'state': 'Fed', 'phase': 'Foraging'},
    5: {'state': 'Fasted', 'phase': 'Exploration'},
    6: {'state': 'Fasted', 'phase': 'Foraging'},
    7: {'state': 'Fasted', 'phase': 'Exploration'},
    8: {'state': 'Fasted', 'phase': 'Foraging'},
}


def load_behavior(beh_path: str) -> dict:
    """Load EthoVision CSV and extract zone time series.

    Returns dict with keys: times, home, ladder, transition, arena
    (all 1D float arrays, same length).
    """
    df = pd.read_csv(beh_path, header=None)
    labels = df.iloc[:, 0].values

    def get_row(name: str) -> np.ndarray:
        idx = np.where(labels == name)[0]
        if len(idx) == 0:
            raise ValueError(f"Row '{name}' not found in {beh_path}")
        return df.iloc[idx[0], 1:].astype(float).values

    return {
        'times': get_row('Recording time'),
        'home': get_row('Home'),
        'ladder': get_row('Ladder'),
        'transition': get_row('Transition zone'),
        'arena': get_row('Foraging arena'),
    }


def detect_excursions(data: dict) -> list[dict]:
    """Detect home-to-home excursions from zone time series.

    Returns list of dicts with keys:
        excursion_id, start_time, end_time, duration, label, farthest_zone
    """
    times = data['times']
    home = data['home']
    ladder = data['ladder']
    transition = data['transition']
    arena = data['arena']
    n = len(times)

    excursions = []
    exc_id = 0

    # State machine
    # States: 'waiting_for_home', 'at_home', 'out'
    state = 'at_home' if home[0] == 1 else 'waiting_for_home'

    start_time = None
    reached_transition = False
    reached_arena = False

    for i in range(n):
        if state == 'waiting_for_home':
            if home[i] == 1:
                state = 'at_home'

        elif state == 'at_home':
            if home[i] == 0:
                # Mouse left home — excursion starts
                start_time = times[i]
                reached_transition = False
                reached_arena = False
                state = 'out'
                # Check current bin zones
                if transition[i] == 1:
                    reached_transition = True
                if arena[i] == 1:
                    reached_arena = True
                    reached_transition = True

        elif state == 'out':
            # Track farthest zone
            if arena[i] == 1:
                reached_arena = True
                reached_transition = True
            if transition[i] == 1:
                reached_transition = True

            # Check for return to Home
            if home[i] == 1:
                end_time = times[i]
                exc_id += 1

                if reached_arena:
                    label = 'complete'
                    farthest = 'Foraging arena'
                elif reached_transition:
                    label = 'incomplete'
                    farthest = 'Transition zone'
                else:
                    label = 'incomplete'
                    farthest = 'Ladder'

                excursions.append({
                    'excursion_id': exc_id,
                    'start_time': round(start_time, 4),
                    'end_time': round(end_time, 4),
                    'duration': round(end_time - start_time, 4),
                    'label': label,
                    'farthest_zone': farthest,
                })
                state = 'at_home'

            # Note: Ladder is NOT an endpoint mid-session because the
            # mouse passes through it on every outbound/return trip.
            # Only Home re-entry ends an excursion.

    # Handle session end: if mouse is still out, record final excursion.
    # Per user: if mouse is at Home or Ladder at session end, it counts.
    if state == 'out' and start_time is not None:
        end_time = times[-1]
        exc_id += 1

        if reached_arena:
            label = 'complete'
            farthest = 'Foraging arena'
        elif reached_transition:
            label = 'incomplete'
            farthest = 'Transition zone'
        else:
            label = 'incomplete'
            farthest = 'Ladder'

        excursions.append({
            'excursion_id': exc_id,
            'start_time': round(start_time, 4),
            'end_time': round(end_time, 4),
            'duration': round(end_time - start_time, 4),
            'label': label,
            'farthest_zone': farthest,
        })

    return excursions


def main():
    print("Excursion Detection — Single-Probe Mouse01 Coord1")
    print("=" * 55)

    sp = paths_config['single_probe']['coordinates_1']['mouse01']['sessions']
    out_dir = Path("data")
    out_dir.mkdir(exist_ok=True)

    for sess_num, info in SESSION_INFO.items():
        key = f"session_{sess_num}"
        beh_path = sp[key].get('behavior')
        if not beh_path or not Path(beh_path).exists():
            print(f"\n  S{sess_num}: no behavior CSV, skipping")
            continue

        print(f"\n  Session {sess_num} ({info['state']}, {info['phase']})")

        data = load_behavior(beh_path)
        print(f"    Duration: {data['times'][-1]:.0f}s, "
              f"{len(data['times'])} bins (100ms)")

        # Check start zone
        start_zones = []
        if data['home'][0] == 1: start_zones.append('Home')
        if data['ladder'][0] == 1: start_zones.append('Ladder')
        if data['transition'][0] == 1: start_zones.append('Transition')
        if data['arena'][0] == 1: start_zones.append('Arena')
        print(f"    Starts in: {start_zones if start_zones else 'None (tracking startup)'}")

        excursions = detect_excursions(data)

        n_complete = sum(1 for e in excursions if e['label'] == 'complete')
        n_incomplete = sum(1 for e in excursions if e['label'] == 'incomplete')
        print(f"    Excursions: {len(excursions)} total "
              f"({n_complete} complete, {n_incomplete} incomplete)")

        if excursions:
            durations = [e['duration'] for e in excursions]
            print(f"    Duration: mean={np.mean(durations):.1f}s, "
                  f"median={np.median(durations):.1f}s, "
                  f"range=[{np.min(durations):.1f}, {np.max(durations):.1f}]s")

            # Farthest zone breakdown
            farthest_counts = {}
            for e in excursions:
                fz = e['farthest_zone']
                farthest_counts[fz] = farthest_counts.get(fz, 0) + 1
            print(f"    Farthest zone: {farthest_counts}")

        # Save CSV
        df_out = pd.DataFrame(excursions)
        out_path = out_dir / f"excursions_session_{sess_num}.csv"
        df_out.to_csv(out_path, index=False)
        print(f"    Saved: {out_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
