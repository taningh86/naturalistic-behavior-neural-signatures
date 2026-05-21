"""
Dual-Probe: Behavioral Metrics Comparison — Fed vs Fasted vs HFD
=================================================================
Extracts per-session behavioral metrics from behavior xlsx files and
compares across metabolic states. Goal: find what is unique to HFD.

Metrics:
  Digging:    bout count, total dig time, mean/median duration, fraction of session,
              latency to first dig, inter-bout interval, pot diversity, pot switching
  Locomotion: mean velocity, velocity during dig, velocity non-dig, distance traveled
  Zone usage: time at home, time at pots, time in transition/ladder
  Manual labels: time in each scored behavior
  Exploration: pot visit count (entering pot zone), unique pots visited per 5-min bin
"""

import yaml
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

with open("paths.yaml") as f:
    cfg = yaml.safe_load(f)

MIN_DIG_DURATION = 2.0
MIN_INTER_DIG = 10.0
SKIP_SESSIONS = {23, 24}

sessions_cfg = cfg["double_probe"]["coordinates_1"]["mouse01"]["sessions"]

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

MANUAL_LABELS = [
    'Digging sand', 'Feeding', 'Transition wall exploration',
    'Hiding in corners', 'Quick one loop at home', 'Incomplete home returns',
    'Contemplation at T-zone', 'Rearing', 'Hiding food at home',
]


def load_behavior_xlsx(path):
    df = pd.read_excel(path, header=None)
    col_names = df.iloc[34].tolist()
    data = df.iloc[36:].reset_index(drop=True)
    data.columns = col_names

    time_vals = pd.to_numeric(data['Recording time'], errors='coerce').values
    vel = pd.to_numeric(data['Velocity(Center-point)'], errors='coerce').values
    vel = np.nan_to_num(vel, nan=0.0)
    vel = np.clip(vel, 0, None)

    zones = np.full(len(time_vals), 'O', dtype=object)
    for zname in zone_priority:
        col_match = [c for c in col_names if isinstance(c, str) and
                     c.startswith('Zone(') and zname in c]
        if col_match:
            vals = pd.to_numeric(data[col_match[0]], errors='coerce').values
            mask = vals > 0.5
            short = zone_short.get(zname, zname[:3])
            zones[mask] = short

    # Manual behavior labels
    label_vals = {}
    for lbl in MANUAL_LABELS:
        if lbl in col_names:
            vals = pd.to_numeric(data[lbl], errors='coerce').values
            vals = np.nan_to_num(vals, nan=0.0)
            label_vals[lbl] = vals
        else:
            label_vals[lbl] = np.zeros(len(time_vals))

    return time_vals, vel, zones, label_vals, col_names


def extract_dig_bouts(dig_vals, time_vals, min_duration, min_inter_dig):
    mask = dig_vals > 0.5
    diff = np.diff(mask.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1
    if mask[0]:
        starts = np.concatenate([[0], starts])
    if mask[-1]:
        ends = np.concatenate([ends, [len(mask)]])
    if len(starts) == 0:
        return []
    bout_times = [(time_vals[s], time_vals[min(e - 1, len(time_vals) - 1)])
                  for s, e in zip(starts, ends)]
    merged = [bout_times[0]]
    for s, e in bout_times[1:]:
        if s - merged[-1][1] < min_inter_dig:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    bouts = []
    for s, e in merged:
        dur = e - s
        if dur >= min_duration:
            bouts.append({'start_time': s, 'end_time': e, 'duration': dur})
    return bouts


def get_pot_at_dig(zones, time_vals, dig_start):
    idx = np.searchsorted(time_vals, dig_start)
    dt = np.median(np.diff(time_vals))
    window = int(2.0 / dt)
    start = max(0, idx - window)
    end = min(len(zones), idx + window)
    segment = zones[start:end]
    pot_zones = [z for z in segment if z.startswith('P') and not z.endswith('z')]
    if pot_zones:
        return Counter(pot_zones).most_common(1)[0][0]
    return 'unknown'


def shannon_entropy(counts):
    """Shannon entropy of a count distribution."""
    total = sum(counts)
    if total == 0:
        return 0
    probs = [c / total for c in counts if c > 0]
    return -sum(p * np.log2(p) for p in probs)


def count_pot_visits(zones, time_vals):
    """Count transitions INTO pot zones (P1-P4)."""
    pot_visits = {'P1': 0, 'P2': 0, 'P3': 0, 'P4': 0}
    prev = zones[0]
    for z in zones[1:]:
        if z in pot_visits and z != prev:
            pot_visits[z] += 1
        prev = z
    return pot_visits


# ========================================================================
# Discover sessions
# ========================================================================
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

print(f"Found {len(session_meta)} sessions with behavior data")

# ========================================================================
# EXTRACT METRICS
# ========================================================================
all_metrics = []

for snum in sorted(session_meta.keys()):
    meta = session_meta[snum]
    state, phase = meta['state'], meta['phase']
    print(f"  S{snum} ({state}/{phase})...", end='', flush=True)

    time_vals, vel, zones, label_vals, col_names = load_behavior_xlsx(meta['behavior'])
    dt = np.median(np.diff(time_vals))
    session_dur = time_vals[-1] - time_vals[0]

    dig_vals = label_vals.get('Digging sand', np.zeros(len(time_vals)))
    has_digging = np.sum(dig_vals > 0.5) > 0

    row = {
        'session': snum, 'state': state, 'phase': phase,
        'session_duration': session_dur,
        'has_digging_labels': has_digging,
    }

    # ---- Digging metrics ----
    if has_digging:
        bouts = extract_dig_bouts(dig_vals, time_vals, MIN_DIG_DURATION, MIN_INTER_DIG)
        row['n_dig_bouts'] = len(bouts)

        if len(bouts) > 0:
            durations = [b['duration'] for b in bouts]
            row['total_dig_time'] = sum(durations)
            row['frac_digging'] = sum(durations) / session_dur
            row['mean_dig_duration'] = np.mean(durations)
            row['median_dig_duration'] = np.median(durations)
            row['max_dig_duration'] = np.max(durations)
            row['latency_first_dig'] = bouts[0]['start_time'] - time_vals[0]

            # Inter-bout intervals
            if len(bouts) > 1:
                ibis = [bouts[i+1]['start_time'] - bouts[i]['end_time']
                        for i in range(len(bouts)-1)]
                row['mean_ibi'] = np.mean(ibis)
                row['median_ibi'] = np.median(ibis)
            else:
                row['mean_ibi'] = np.nan
                row['median_ibi'] = np.nan

            # Pot diversity
            pots = [get_pot_at_dig(zones, time_vals, b['start_time']) for b in bouts]
            pot_counts = Counter(pots)
            known_pots = {k: v for k, v in pot_counts.items() if k != 'unknown'}
            row['n_unique_pots'] = len(known_pots)
            row['pot_entropy'] = shannon_entropy(list(known_pots.values()))
            row['most_common_pot'] = pot_counts.most_common(1)[0][0] if pot_counts else 'none'
            row['most_common_pot_frac'] = (pot_counts.most_common(1)[0][1] / len(bouts)
                                           if pot_counts else 0)

            # Pot switching: fraction of consecutive bouts at different pots
            if len(pots) > 1:
                switches = sum(1 for i in range(len(pots)-1) if pots[i] != pots[i+1])
                row['pot_switch_rate'] = switches / (len(pots) - 1)
            else:
                row['pot_switch_rate'] = np.nan
        else:
            for k in ['total_dig_time', 'frac_digging', 'mean_dig_duration',
                       'median_dig_duration', 'max_dig_duration', 'latency_first_dig',
                       'mean_ibi', 'median_ibi', 'n_unique_pots', 'pot_entropy',
                       'pot_switch_rate']:
                row[k] = np.nan
            row['most_common_pot'] = 'none'
            row['most_common_pot_frac'] = 0
    else:
        row['n_dig_bouts'] = 0
        for k in ['total_dig_time', 'frac_digging', 'mean_dig_duration',
                   'median_dig_duration', 'max_dig_duration', 'latency_first_dig',
                   'mean_ibi', 'median_ibi', 'n_unique_pots', 'pot_entropy',
                   'pot_switch_rate']:
            row[k] = np.nan
        row['most_common_pot'] = 'none'
        row['most_common_pot_frac'] = 0

    # ---- Velocity metrics ----
    row['mean_velocity'] = np.mean(vel)
    row['median_velocity'] = np.median(vel)
    row['max_velocity'] = np.max(vel)
    row['velocity_std'] = np.std(vel)

    # Velocity percentiles
    row['vel_p10'] = np.percentile(vel, 10)
    row['vel_p90'] = np.percentile(vel, 90)

    # Distance traveled (sum of velocity * dt)
    row['total_distance'] = np.sum(vel * dt)

    # Velocity during dig vs non-dig
    if has_digging and np.sum(dig_vals > 0.5) > 10:
        dig_mask = dig_vals > 0.5
        row['vel_during_dig'] = np.mean(vel[dig_mask])
        row['vel_non_dig'] = np.mean(vel[~dig_mask])
    else:
        row['vel_during_dig'] = np.nan
        row['vel_non_dig'] = np.nan

    # ---- Zone occupancy (fraction of time) ----
    zone_groups = {
        'home': ['H', 'HCL', 'HCR'],
        'pots': ['P1', 'P2', 'P3', 'P4'],
        'pot_zones': ['P1z', 'P2z', 'P3z', 'P4z'],
        'transition': ['T', 'L'],
        'arena': ['FA', 'CA'],
    }
    for gname, zone_list in zone_groups.items():
        mask = np.isin(zones, zone_list)
        row[f'frac_{gname}'] = np.sum(mask) / len(zones)

    # Individual pot fractions
    for pot in ['P1', 'P2', 'P3', 'P4']:
        mask = zones == pot
        row[f'frac_{pot}'] = np.sum(mask) / len(zones)

    # ---- Pot visits (transitions into pot zones) ----
    pot_visits = count_pot_visits(zones, time_vals)
    row['total_pot_visits'] = sum(pot_visits.values())
    for pot, count in pot_visits.items():
        row[f'visits_{pot}'] = count

    # Pot visit entropy
    row['pot_visit_entropy'] = shannon_entropy(list(pot_visits.values()))

    # ---- Manual behavior labels (fraction of time) ----
    for lbl in MANUAL_LABELS:
        vals = label_vals[lbl]
        frac = np.sum(vals > 0.5) / len(vals)
        row[f'behav_{lbl}'] = frac

    # ---- Movement patterns ----
    # Immobility: fraction of time with velocity < 1 cm/s
    row['frac_immobile'] = np.sum(vel < 1.0) / len(vel)

    # High-speed fraction: velocity > 15 cm/s (darting/running)
    row['frac_fast'] = np.sum(vel > 15.0) / len(vel)

    # Velocity autocorrelation at 5s lag (movement persistence)
    lag_bins = int(5.0 / dt)
    if lag_bins < len(vel) - 1:
        v1 = vel[:-lag_bins]
        v2 = vel[lag_bins:]
        if np.std(v1) > 0 and np.std(v2) > 0:
            row['vel_autocorr_5s'] = np.corrcoef(v1, v2)[0, 1]
        else:
            row['vel_autocorr_5s'] = np.nan
    else:
        row['vel_autocorr_5s'] = np.nan

    all_metrics.append(row)
    print(f" done")

df = pd.DataFrame(all_metrics)
df.to_csv('data/dp_behavior_hfd_comparison.csv', index=False)
print(f"\nSaved data/dp_behavior_hfd_comparison.csv ({len(df)} rows)")

# ========================================================================
# STATISTICAL COMPARISON: Fed vs Fasted vs HFD
# ========================================================================
print("\n" + "=" * 100)
print("BEHAVIORAL METRICS: FED vs FASTED vs HFD")
print("=" * 100)

# Only use sessions with digging labels for digging metrics
df_dig = df[df['has_digging_labels'] == True].copy()

state_map = {'fed': 'Fed', 'fasted': 'Fasted', 'fed-HFD': 'HFD'}

# Define which metrics to compare
compare_metrics = [
    # Digging
    ('n_dig_bouts', 'Dig bouts per session', True),
    ('total_dig_time', 'Total dig time (s)', True),
    ('frac_digging', 'Fraction of session digging', True),
    ('mean_dig_duration', 'Mean dig bout duration (s)', True),
    ('median_dig_duration', 'Median dig bout duration (s)', True),
    ('max_dig_duration', 'Max dig bout duration (s)', True),
    ('latency_first_dig', 'Latency to first dig (s)', True),
    ('mean_ibi', 'Mean inter-bout interval (s)', True),
    ('n_unique_pots', 'Unique pots dug', True),
    ('pot_entropy', 'Pot dig entropy (bits)', True),
    ('pot_switch_rate', 'Pot switching rate', True),
    # Velocity
    ('mean_velocity', 'Mean velocity (cm/s)', False),
    ('median_velocity', 'Median velocity (cm/s)', False),
    ('velocity_std', 'Velocity SD', False),
    ('total_distance', 'Total distance (cm)', False),
    ('vel_during_dig', 'Velocity during dig', True),
    ('vel_non_dig', 'Velocity non-dig', True),
    ('frac_immobile', 'Fraction immobile (<1 cm/s)', False),
    ('frac_fast', 'Fraction fast (>15 cm/s)', False),
    ('vel_autocorr_5s', 'Velocity autocorr (5s lag)', False),
    # Zone occupancy
    ('frac_home', 'Fraction at Home', False),
    ('frac_pots', 'Fraction at Pots', False),
    ('frac_transition', 'Fraction in Transition/Ladder', False),
    ('frac_arena', 'Fraction in Arena', False),
    ('frac_P1', 'Fraction at P1', False),
    ('frac_P2', 'Fraction at P2', False),
    ('frac_P3', 'Fraction at P3', False),
    ('frac_P4', 'Fraction at P4', False),
    # Pot visits
    ('total_pot_visits', 'Total pot visits', False),
    ('pot_visit_entropy', 'Pot visit entropy', False),
    # Manual behaviors
    ('behav_Digging sand', 'Digging (manual label frac)', True),
    ('behav_Feeding', 'Feeding frac', True),
    ('behav_Transition wall exploration', 'Transition wall exploration frac', True),
    ('behav_Hiding in corners', 'Hiding in corners frac', True),
    ('behav_Quick one loop at home', 'Quick home loop frac', True),
    ('behav_Incomplete home returns', 'Incomplete home returns frac', True),
    ('behav_Contemplation at T-zone', 'Contemplation at T-zone frac', True),
    ('behav_Rearing', 'Rearing frac', True),
    # Movement
    ('vel_p10', 'Velocity 10th percentile', False),
    ('vel_p90', 'Velocity 90th percentile', False),
]

results_rows = []
significant_metrics = []

states = ['fed', 'fasted', 'fed-HFD']

for metric, label, dig_only in compare_metrics:
    source = df_dig if dig_only else df
    groups = {}
    for st in states:
        vals = source.loc[source['state'] == st, metric].dropna().values
        if len(vals) > 0:
            groups[st] = vals

    if len(groups) < 2:
        continue

    # Print means
    means_str = ', '.join([f"{state_map.get(s,s)}={np.mean(v):.4f} (n={len(v)})"
                           for s, v in groups.items()])

    # Kruskal-Wallis if 3 groups
    kw_p = np.nan
    if len(groups) == 3:
        vals_list = [groups[s] for s in states if s in groups]
        if all(len(v) >= 2 for v in vals_list):
            try:
                _, kw_p = kruskal(*vals_list)
            except Exception:
                kw_p = 1.0

    # Pairwise MWU
    pairs = []
    for i in range(len(states)):
        for j in range(i+1, len(states)):
            s1, s2 = states[i], states[j]
            if s1 in groups and s2 in groups and len(groups[s1]) >= 2 and len(groups[s2]) >= 2:
                try:
                    _, p = mannwhitneyu(groups[s1], groups[s2], alternative='two-sided')
                except Exception:
                    p = 1.0
                pairs.append((s1, s2, p))

    # Check if HFD is significantly different from BOTH fed and fasted
    hfd_unique = False
    hfd_vs_fed_p = np.nan
    hfd_vs_fasted_p = np.nan
    for s1, s2, p in pairs:
        if 'fed-HFD' in (s1, s2) and 'fed' in (s1, s2):
            hfd_vs_fed_p = p
        if 'fed-HFD' in (s1, s2) and 'fasted' in (s1, s2):
            hfd_vs_fasted_p = p
    if hfd_vs_fed_p < 0.1 and hfd_vs_fasted_p < 0.1:
        hfd_unique = True

    any_sig = kw_p < 0.05 or any(p < 0.05 for _, _, p in pairs)
    sig_marker = '***' if hfd_unique else ('*' if any_sig else '')

    if any_sig or hfd_unique:
        print(f"\n  {sig_marker} {label}")
        print(f"    {means_str}")
        if not np.isnan(kw_p):
            print(f"    KW p={kw_p:.4f}{'*' if kw_p < 0.05 else ''}")
        for s1, s2, p in pairs:
            print(f"    {state_map[s1]} vs {state_map[s2]}: MWU p={p:.4f}{'*' if p < 0.05 else ''}")

        if hfd_unique:
            significant_metrics.append((label, metric, groups, kw_p, pairs))

    results_rows.append({
        'metric': metric, 'label': label,
        'fed_mean': np.mean(groups['fed']) if 'fed' in groups else np.nan,
        'fed_n': len(groups['fed']) if 'fed' in groups else 0,
        'fasted_mean': np.mean(groups['fasted']) if 'fasted' in groups else np.nan,
        'fasted_n': len(groups['fasted']) if 'fasted' in groups else 0,
        'hfd_mean': np.mean(groups['fed-HFD']) if 'fed-HFD' in groups else np.nan,
        'hfd_n': len(groups['fed-HFD']) if 'fed-HFD' in groups else 0,
        'kw_p': kw_p,
        'fed_vs_fasted_p': next((p for s1, s2, p in pairs if set([s1,s2]) == {'fed','fasted'}), np.nan),
        'fed_vs_hfd_p': hfd_vs_fed_p,
        'fasted_vs_hfd_p': hfd_vs_fasted_p,
        'hfd_unique': hfd_unique,
    })

df_results = pd.DataFrame(results_rows)
df_results.to_csv('data/dp_behavior_hfd_stats.csv', index=False)
print(f"\nSaved data/dp_behavior_hfd_stats.csv ({len(df_results)} rows)")

# ========================================================================
# SUMMARY: HFD-Unique Metrics
# ========================================================================
print("\n" + "=" * 100)
print("HFD-UNIQUE METRICS (p < 0.1 vs BOTH fed and fasted)")
print("=" * 100)

if len(significant_metrics) == 0:
    print("  None found at p < 0.1 threshold")
else:
    for label, metric, groups, kw_p, pairs in significant_metrics:
        print(f"\n  {label}:")
        for st, vals in groups.items():
            print(f"    {state_map[st]}: {np.mean(vals):.4f} ± {np.std(vals):.4f} (n={len(vals)})")

# ========================================================================
# FIGURE: Bar plots of key metrics by state
# ========================================================================
state_colors = {'fed': '#4e79a7', 'fasted': '#e15759', 'fed-HFD': '#f28e2b'}

# Select metrics for figure (mix of digging and non-digging)
fig_metrics = [
    ('n_dig_bouts', 'Dig Bouts', True),
    ('frac_digging', 'Frac. Digging', True),
    ('mean_dig_duration', 'Mean Dig Duration (s)', True),
    ('pot_switch_rate', 'Pot Switch Rate', True),
    ('pot_entropy', 'Dig Pot Entropy', True),
    ('latency_first_dig', 'Latency to 1st Dig (s)', True),
    ('mean_velocity', 'Mean Velocity', False),
    ('frac_immobile', 'Frac. Immobile', False),
    ('frac_home', 'Frac. at Home', False),
    ('frac_pots', 'Frac. at Pots', False),
    ('total_pot_visits', 'Total Pot Visits', False),
    ('pot_visit_entropy', 'Visit Entropy', False),
    ('vel_autocorr_5s', 'Vel. Autocorr (5s)', False),
    ('frac_fast', 'Frac. Fast (>15cm/s)', False),
    ('total_distance', 'Total Distance (cm)', False),
    ('behav_Incomplete home returns', 'Incomplete Returns', True),
]

n_metrics = len(fig_metrics)
n_cols = 4
n_rows = (n_metrics + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 5 * n_rows))
axes = axes.flatten()

for idx, (metric, label, dig_only) in enumerate(fig_metrics):
    ax = axes[idx]
    source = df_dig if dig_only else df

    x_pos = []
    x_labels = []
    for si, st in enumerate(states):
        vals = source.loc[source['state'] == st, metric].dropna().values
        if len(vals) == 0:
            continue
        bar_x = si
        x_pos.append(bar_x)
        x_labels.append(state_map[st])
        mean_v = np.mean(vals)
        sem_v = np.std(vals) / np.sqrt(len(vals)) if len(vals) > 1 else 0

        ax.bar(bar_x, mean_v, color=state_colors[st], alpha=0.7, width=0.6,
               edgecolor='black', linewidth=0.5)
        ax.errorbar(bar_x, mean_v, yerr=sem_v, color='black', capsize=5,
                    capthick=1.5, linewidth=1.5)

        # Individual data points
        jitter = np.random.uniform(-0.15, 0.15, len(vals))
        ax.scatter(bar_x + jitter, vals, color=state_colors[st],
                   edgecolor='black', linewidth=0.5, s=40, zorder=5, alpha=0.8)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, fontsize=13)
    ax.set_ylabel(label, fontsize=14)
    ax.tick_params(labelsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add significance from results
    row = df_results[df_results['metric'] == metric]
    if len(row) > 0:
        r = row.iloc[0]
        if r['hfd_unique']:
            ax.set_title(f'{label}\nHFD UNIQUE', fontsize=13, fontweight='bold',
                         color='#d62728')
        elif r['kw_p'] < 0.05:
            ax.set_title(f'{label}\nKW p={r["kw_p"]:.3f}*', fontsize=13)
        else:
            ax.set_title(label, fontsize=13)
    else:
        ax.set_title(label, fontsize=13)

# Remove empty axes
for idx in range(n_metrics, len(axes)):
    axes[idx].set_visible(False)

fig.suptitle('Behavioral Metrics: Fed vs Fasted vs HFD',
             fontsize=20, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('figures/dp_behavior_hfd_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved figures/dp_behavior_hfd_comparison.png")

# ========================================================================
# FIGURE 2: Exploration-only vs Foraging-only comparison
# ========================================================================
fig, axes = plt.subplots(2, 4, figsize=(24, 10))

phase_metrics = [
    ('frac_digging', 'Frac. Digging', True),
    ('mean_velocity', 'Mean Velocity', False),
    ('frac_home', 'Frac. at Home', False),
    ('frac_pots', 'Frac. at Pots', False),
]

for phase_idx, phase_name in enumerate(['exploration', 'foraging']):
    for m_idx, (metric, label, dig_only) in enumerate(phase_metrics):
        ax = axes[phase_idx, m_idx]
        source = df_dig if dig_only else df
        source_phase = source[source['phase'] == phase_name]

        for si, st in enumerate(states):
            vals = source_phase.loc[source_phase['state'] == st, metric].dropna().values
            if len(vals) == 0:
                continue
            mean_v = np.mean(vals)
            sem_v = np.std(vals) / np.sqrt(len(vals)) if len(vals) > 1 else 0

            ax.bar(si, mean_v, color=state_colors[st], alpha=0.7, width=0.6,
                   edgecolor='black', linewidth=0.5)
            ax.errorbar(si, mean_v, yerr=sem_v, color='black', capsize=5,
                        capthick=1.5, linewidth=1.5)
            jitter = np.random.uniform(-0.15, 0.15, len(vals))
            ax.scatter(si + jitter, vals, color=state_colors[st],
                       edgecolor='black', linewidth=0.5, s=40, zorder=5, alpha=0.8)

        ax.set_xticks(range(len(states)))
        ax.set_xticklabels([state_map[s] for s in states], fontsize=12)
        ax.set_title(f'{label} — {phase_name.capitalize()}', fontsize=13)
        ax.set_ylabel(label, fontsize=12)
        ax.tick_params(labelsize=11)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

fig.suptitle('Behavioral Metrics by Phase: Exploration vs Foraging',
             fontsize=18, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('figures/dp_behavior_hfd_by_phase.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved figures/dp_behavior_hfd_by_phase.png")

print("\nDone.")
