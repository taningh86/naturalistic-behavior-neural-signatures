"""
Dual-Probe: Home Visit Analysis
================================
Extracts all home visits from zone data, classifies them as:
  1. Quick loop at home (overlaps with manual label)
  2. Incomplete home return (overlaps with manual label)
  3. Regular home visit (no overlap with either label)

Computes per-visit: duration, timing, preceding zone, following zone.
Summarizes how home visits change across session time, phase, and state.
"""

import yaml
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

with open("paths.yaml") as f:
    cfg = yaml.safe_load(f)

SKIP_SESSIONS = {23, 24}
HOME_ZONES = {'H', 'HCL', 'HCR'}  # All home-related zones

# Zone mapping (same as other scripts)
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

BEHAV_QUICK_LOOP = 'Quick one loop at home'
BEHAV_INCOMPLETE = 'Incomplete home returns'


def load_behavior_xlsx(path):
    """Load dual-probe behavior xlsx."""
    df = pd.read_excel(path, header=None)
    col_names = df.iloc[34].tolist()
    data = df.iloc[36:].reset_index(drop=True)
    data.columns = col_names

    time_vals = pd.to_numeric(data['Recording time'], errors='coerce').values

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

    # Load behavior labels
    labels = {}
    for bcol in [BEHAV_QUICK_LOOP, BEHAV_INCOMPLETE]:
        if bcol in col_names:
            bvals = pd.to_numeric(data[bcol], errors='coerce').values
            labels[bcol] = np.nan_to_num(bvals, nan=0.0)
        else:
            labels[bcol] = np.zeros(len(time_vals))

    return time_vals, zones, labels


def extract_label_bouts(label_arr, time_vals, zones):
    """Extract contiguous bouts from a binary behavior label array.
    Returns list of dicts with start/end info."""
    mask = label_arr > 0.5
    diff = np.diff(mask.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1
    if mask[0]:
        starts = np.concatenate([[0], starts])
    if mask[-1]:
        ends = np.concatenate([ends, [len(mask)]])

    bouts = []
    for s, e in zip(starts, ends):
        start_t = time_vals[s]
        end_t = time_vals[min(e - 1, len(time_vals) - 1)]
        # Dominant zone during this bout
        bout_zones = zones[s:e]
        zone_counts = pd.Series(bout_zones).value_counts()
        dominant_zone = zone_counts.index[0] if len(zone_counts) > 0 else 'O'
        bouts.append({
            'start_idx': s, 'end_idx': e,
            'start_time': start_t, 'end_time': end_t,
            'duration': end_t - start_t,
            'dominant_zone': dominant_zone,
        })
    return bouts


def extract_home_visits(time_vals, zones, labels):
    """Extract three independent categories of home-related events:
      1. 'quick_loop' — from manual label (Quick one loop at home)
      2. 'incomplete_return' — from manual label (Incomplete home returns)
      3. 'regular' — zone-based home visits NOT overlapping either label

    Returns list of dicts.
    """
    visits = []

    # --- 1. Quick loops from manual label ---
    ql_bouts = extract_label_bouts(labels[BEHAV_QUICK_LOOP], time_vals, zones)
    for b in ql_bouts:
        visits.append({**b, 'category': 'quick_loop'})

    # --- 2. Incomplete returns from manual label ---
    ir_bouts = extract_label_bouts(labels[BEHAV_INCOMPLETE], time_vals, zones)
    for b in ir_bouts:
        visits.append({**b, 'category': 'incomplete_return'})

    # --- 3. Regular home visits from zone data (not overlapping labels) ---
    is_home = np.array([z in HOME_ZONES for z in zones])
    diff = np.diff(is_home.astype(int))
    entries = np.where(diff == 1)[0] + 1
    exits = np.where(diff == -1)[0] + 1
    if is_home[0]:
        entries = np.concatenate([[0], entries])
    if is_home[-1]:
        exits = np.concatenate([exits, [len(is_home)]])

    for entry, exit_ in zip(entries, exits):
        # Check if this zone-based visit overlaps with either manual label
        ql_overlap = np.any(labels[BEHAV_QUICK_LOOP][entry:exit_] > 0.5)
        ir_overlap = np.any(labels[BEHAV_INCOMPLETE][entry:exit_] > 0.5)
        if ql_overlap or ir_overlap:
            continue  # already counted from manual labels

        start_t = time_vals[entry]
        end_t = time_vals[min(exit_ - 1, len(time_vals) - 1)]
        visits.append({
            'start_idx': entry, 'end_idx': exit_,
            'start_time': start_t, 'end_time': end_t,
            'duration': end_t - start_t,
            'dominant_zone': 'H',
            'category': 'regular',
        })

    # Sort by start time
    visits.sort(key=lambda v: v['start_time'])
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
# Extract home visits for all sessions
# ============================================================
all_visits = []

for snum in target_sessions:
    skey = f"session_{snum}"
    sval = sessions_cfg[skey]
    behav_path = sval['behavior']
    state = sval['state']
    phase = sval['phase']

    print(f"\nS{snum} ({state}/{phase}): loading...", end=' ', flush=True)
    time_vals, zones, labels = load_behavior_xlsx(behav_path)
    session_duration = time_vals[-1] - time_vals[0]

    visits = extract_home_visits(time_vals, zones, labels)
    print(f"{len(visits)} home visits in {session_duration/60:.1f} min")

    # Categorize
    cats = {}
    for v in visits:
        cats.setdefault(v['category'], []).append(v)

    for cat, vlist in sorted(cats.items()):
        durations = [v['duration'] for v in vlist]
        print(f"  {cat:25s}: {len(vlist):3d} visits, "
              f"median={np.median(durations):.1f}s, "
              f"mean={np.mean(durations):.1f}s, "
              f"range=[{np.min(durations):.1f}, {np.max(durations):.1f}]s")

    # Store individual visit data
    for v in visits:
        # Compute session fraction (0=start, 1=end)
        session_frac = v['start_time'] / session_duration if session_duration > 0 else 0
        all_visits.append({
            'session': snum,
            'state': state,
            'phase': phase,
            'category': v['category'],
            'start_time': v['start_time'],
            'end_time': v['end_time'],
            'duration': v['duration'],
            'session_fraction': session_frac,
            'dominant_zone': v.get('dominant_zone', ''),
        })

# Save CSV
df = pd.DataFrame(all_visits)
csv_path = 'data/dp_home_visits.csv'
df.to_csv(csv_path, index=False)
print(f"\nSaved {csv_path} ({len(df)} rows)")

# ============================================================
# Summary statistics
# ============================================================
print("\n" + "=" * 80)
print("HOME VISIT SUMMARY")
print("=" * 80)

for state in ['fed', 'fasted', 'fed-HFD']:
    for phase in ['exploration', 'foraging']:
        sub = df[(df['state'] == state) & (df['phase'] == phase)]
        if len(sub) == 0:
            continue
        n_sessions = sub['session'].nunique()
        print(f"\n  {state} / {phase} ({n_sessions} sessions, {len(sub)} total visits):")
        for cat in ['regular', 'quick_loop', 'incomplete_return', 'quick_loop+incomplete']:
            csub = sub[sub['category'] == cat]
            if len(csub) == 0:
                continue
            per_session = csub.groupby('session').size()
            dur = csub['duration']
            print(f"    {cat:25s}: {len(csub):4d} total "
                  f"({per_session.mean():.1f}/session), "
                  f"dur median={dur.median():.1f}s mean={dur.mean():.1f}s")

# ============================================================
# Figure 1: Home visit counts by category, state, phase
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(24, 8), sharey=True)

categories = ['regular', 'quick_loop', 'incomplete_return']
cat_labels = ['Regular', 'Quick Loop', 'Incomplete Return']
cat_colors = ['#4e79a7', '#e15759', '#f28e2b']

for ax_idx, (cat, cat_label, color) in enumerate(zip(categories, cat_labels, cat_colors)):
    ax = axes[ax_idx]
    groups = []
    group_labels = []
    for state in ['fed', 'fasted', 'fed-HFD']:
        for phase in ['exploration', 'foraging']:
            sub = df[(df['state'] == state) & (df['phase'] == phase) &
                     (df['category'] == cat)]
            if sub['session'].nunique() == 0:
                continue
            per_session = sub.groupby('session').size()
            groups.append(per_session.values)
            state_short = {'fed': 'Fed', 'fasted': 'Fast', 'fed-HFD': 'HFD'}[state]
            phase_short = {'exploration': 'Exp', 'foraging': 'For'}[phase]
            group_labels.append(f'{state_short}\n{phase_short}')

    positions = np.arange(len(groups))
    bp = ax.boxplot(groups, positions=positions, widths=0.5,
                    patch_artist=True, showmeans=True,
                    meanprops=dict(marker='D', markerfacecolor='white',
                                   markeredgecolor='black', markersize=8))
    for patch in bp['boxes']:
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    # Overlay individual session points
    for i, vals in enumerate(groups):
        jitter = np.random.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(positions[i] + jitter, vals, color=color, s=50,
                   edgecolors='black', linewidths=0.5, zorder=3, alpha=0.8)

    ax.set_xticks(positions)
    ax.set_xticklabels(group_labels, fontsize=13)
    ax.set_title(cat_label, fontsize=16, fontweight='bold', color=color)
    ax.set_ylabel('Visits per Session' if ax_idx == 0 else '', fontsize=14, fontweight='bold')
    ax.tick_params(labelsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

fig.suptitle('Home Visit Counts by Category', fontsize=18, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('figures/dp_home_visit_counts.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved figures/dp_home_visit_counts.png")

# ============================================================
# Figure 2: Home visit duration by category, state, phase
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(24, 8), sharey=True)

for ax_idx, (cat, cat_label, color) in enumerate(zip(categories, cat_labels, cat_colors)):
    ax = axes[ax_idx]
    groups = []
    group_labels = []
    for state in ['fed', 'fasted', 'fed-HFD']:
        for phase in ['exploration', 'foraging']:
            sub = df[(df['state'] == state) & (df['phase'] == phase) &
                     (df['category'] == cat)]
            if len(sub) == 0:
                continue
            groups.append(sub['duration'].values)
            state_short = {'fed': 'Fed', 'fasted': 'Fast', 'fed-HFD': 'HFD'}[state]
            phase_short = {'exploration': 'Exp', 'foraging': 'For'}[phase]
            group_labels.append(f'{state_short}\n{phase_short}')

    if not groups:
        continue

    positions = np.arange(len(groups))
    bp = ax.boxplot(groups, positions=positions, widths=0.5,
                    patch_artist=True, showmeans=True, showfliers=False,
                    meanprops=dict(marker='D', markerfacecolor='white',
                                   markeredgecolor='black', markersize=8))
    for patch in bp['boxes']:
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_xticks(positions)
    ax.set_xticklabels(group_labels, fontsize=13)
    ax.set_title(cat_label, fontsize=16, fontweight='bold', color=color)
    ax.set_ylabel('Duration (s)' if ax_idx == 0 else '', fontsize=14, fontweight='bold')
    ax.tick_params(labelsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

fig.suptitle('Home Visit Duration by Category', fontsize=18, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('figures/dp_home_visit_durations.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved figures/dp_home_visit_durations.png")

# ============================================================
# Figure 3: Home visits over session time (per-session rasters)
# ============================================================
# Group sessions by state
state_order = ['fed', 'fasted', 'fed-HFD']
state_labels = {'fed': 'Fed', 'fasted': 'Fasted', 'fed-HFD': 'HFD'}

fig, axes = plt.subplots(1, len(state_order), figsize=(28, 10), sharey=False)

for ax_idx, state in enumerate(state_order):
    ax = axes[ax_idx]
    state_sessions = sorted(df[df['state'] == state]['session'].unique())
    y_pos = 0
    ytick_positions = []
    ytick_labels = []

    for snum in state_sessions:
        sval = sessions_cfg[f'session_{snum}']
        phase = sval['phase']
        sub = df[(df['session'] == snum)]
        session_dur = sub['end_time'].max() if len(sub) > 0 else 1800

        for _, row in sub.iterrows():
            cat = row['category']
            if cat == 'regular':
                color = '#4e79a7'
            elif cat == 'quick_loop':
                color = '#e15759'
            elif cat == 'incomplete_return':
                color = '#f28e2b'
            else:
                color = '#76b7b2'

            # Draw horizontal bar for each visit
            ax.barh(y_pos, row['duration'], left=row['start_time'] / 60,
                    height=0.7, color=color, alpha=0.8, edgecolor='none')

        ytick_positions.append(y_pos)
        phase_short = 'E' if phase == 'exploration' else 'F'
        ytick_labels.append(f'S{snum} ({phase_short})')
        y_pos += 1

    ax.set_yticks(ytick_positions)
    ax.set_yticklabels(ytick_labels, fontsize=12)
    ax.set_xlabel('Time (min)', fontsize=14, fontweight='bold')
    ax.set_title(state_labels[state], fontsize=16, fontweight='bold')
    ax.tick_params(labelsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.invert_yaxis()

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#4e79a7', label='Regular'),
    Patch(facecolor='#e15759', label='Quick Loop'),
    Patch(facecolor='#f28e2b', label='Incomplete Return'),
]
axes[-1].legend(handles=legend_elements, loc='upper left',
                bbox_to_anchor=(1.01, 1.0), fontsize=14,
                framealpha=0.9, edgecolor='black', title='Category',
                title_fontsize=15)

fig.suptitle('Home Visits Over Session Time', fontsize=18, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('figures/dp_home_visit_raster.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved figures/dp_home_visit_raster.png")

print("\nDone.")
