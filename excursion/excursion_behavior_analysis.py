"""
Behavioral Analysis of Topologically Prominent Excursions
=========================================================
What behaviors occurred during excursions with significant H1 loops
vs excursions without loops?

Extracts all EthoVision behavioral variables during each excursion
and compares prominent (RSP 200ms H1 gap > 3) vs non-prominent.
"""

import yaml
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

with open("paths.yaml") as f:
    paths_config = yaml.safe_load(f)

# Prominent excursions from statistical testing (RSP 200ms/500ms, p < 0.05)
SIGNIFICANT_EXCURSIONS = [90, 71, 89, 88]
# All prominent across any resolution (gap > 3)
ALL_PROMINENT = [90, 71, 89, 88, 5, 13, 4, 35]

# =============================================================================
# LOAD BEHAVIOR
# =============================================================================

def load_behavior_csv(session_num):
    """Load full behavior CSV, return as dict of variable_name -> time series."""
    sp = paths_config['single_probe']['coordinates_1']['mouse01']['sessions']
    sc = sp[f'session_{session_num}']
    behav_path = sc.get('behavior')
    if not behav_path or not Path(behav_path).exists():
        print(f"  No behavior file for session {session_num}")
        return None, None

    df = pd.read_csv(behav_path, header=None)

    # Row labels are in column 0
    variables = {}
    row_names = {}
    for i in range(df.shape[0]):
        name = str(df.iloc[i, 0]).strip()
        if name and name != 'nan':
            row_names[i] = name
            data = pd.to_numeric(df.iloc[i, 1:], errors='coerce').values
            variables[name] = data

    # Time axis: 100ms bins
    n_bins = len(list(variables.values())[0])
    time_sec = np.arange(n_bins) * 0.1

    return variables, time_sec


def get_behavior_during_excursion(variables, time_sec, start_time, end_time):
    """Extract all behavioral variables during an excursion time window."""
    mask = (time_sec >= start_time) & (time_sec <= end_time)
    result = {}
    for name, data in variables.items():
        segment = data[mask]
        result[name] = segment
    return result, time_sec[mask]


# =============================================================================
# ZONE VARIABLES (binary 0/1)
# =============================================================================
ZONE_VARS = ['Home', 'Ladder', 'Transition zone', 'Foraging arena',
             'Pot-1', 'Pot-2', 'Pot-3', 'Pot-4',
             'Pot-1 zone', 'Pot-2 zone', 'Pot-3 zone', 'Pot-4 zone',
             'Right corner', 'Left corner', 'Arna center']

# Binary behavior labels (scored by observer)
BEHAVIOR_VARS = ['Feeding', 'Digging', 'Grooming',
                 'Longer exploration at home', 'Quick and hasty exploration at home',
                 'Quick one loop at home', 'Incomplete home return',
                 'Contemplation at T-zone', 'Transition wall exploration',
                 'Arena wall exploration', 'Random switching between pots',
                 'Intentional switching between pots', 'Quick arena exploration',
                 'Hesitant exploration', 'Hiding in corners']

# Continuous variables
CONTINUOUS_VARS = ['Velocity', 'Distance moved', 'Areachange', 'Meander',
                   'Distance to Pot-2', 'Distance to Pot-4',
                   'Distance to Transition zone', 'Distance to Home',
                   'Distance to Foraging arena']

MOVEMENT_VARS = ['Movement(Moving / Center-point)', 'High acceleration',
                 'Low acceleration']


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("Behavioral Analysis of Topologically Prominent Excursions")
    print("=" * 60)

    # Load behavior
    variables, behav_time = load_behavior_csv(1)
    if variables is None:
        return

    print(f"  Loaded {len(variables)} behavioral variables, {len(behav_time)} time bins")
    print(f"  Time range: {behav_time[0]:.1f} - {behav_time[-1]:.1f} s")

    # Load excursions
    exc_df = pd.read_csv("data/excursions_session_1.csv")
    complete = exc_df[exc_df['label'] == 'complete'].copy()
    complete['is_significant'] = complete['excursion_id'].isin(SIGNIFICANT_EXCURSIONS)
    complete['is_prominent'] = complete['excursion_id'].isin(ALL_PROMINENT)
    print(f"  {len(complete)} complete excursions")
    print(f"  {complete['is_significant'].sum()} statistically significant (RSP 200ms p<0.05)")
    print(f"  {complete['is_prominent'].sum()} prominent across any resolution (gap>3)")

    # =========================================================================
    # Extract behavior for each excursion
    # =========================================================================
    all_exc_data = []

    for _, erow in complete.iterrows():
        eid = int(erow['excursion_id'])
        behav_seg, seg_time = get_behavior_during_excursion(
            variables, behav_time, erow['start_time'], erow['end_time'])

        if len(seg_time) < 2:
            continue

        row = {
            'excursion_id': eid,
            'duration': erow['duration'],
            'is_significant': erow['is_significant'],
            'is_prominent': erow['is_prominent'],
            'n_bins': len(seg_time),
        }

        # Binary behaviors: fraction of time engaged
        for var in BEHAVIOR_VARS:
            if var in behav_seg:
                data = behav_seg[var]
                valid = data[~np.isnan(data)]
                row[f'{var}_frac'] = np.mean(valid > 0) if len(valid) > 0 else 0
                row[f'{var}_any'] = 1 if (valid > 0).any() else 0
            else:
                row[f'{var}_frac'] = np.nan
                row[f'{var}_any'] = np.nan

        # Zone occupancy: fraction of time in each zone
        for var in ZONE_VARS:
            if var in behav_seg:
                data = behav_seg[var]
                valid = data[~np.isnan(data)]
                row[f'{var}_frac'] = np.mean(valid > 0) if len(valid) > 0 else 0
            else:
                row[f'{var}_frac'] = np.nan

        # Continuous variables: mean, std
        for var in CONTINUOUS_VARS:
            if var in behav_seg:
                data = behav_seg[var]
                valid = data[~np.isnan(data)]
                row[f'{var}_mean'] = np.mean(valid) if len(valid) > 0 else np.nan
                row[f'{var}_std'] = np.std(valid) if len(valid) > 0 else np.nan
            else:
                row[f'{var}_mean'] = np.nan
                row[f'{var}_std'] = np.nan

        # Movement: fraction moving
        for var in MOVEMENT_VARS:
            if var in behav_seg:
                data = behav_seg[var]
                valid = data[~np.isnan(data)]
                row[f'{var}_frac'] = np.mean(valid > 0) if len(valid) > 0 else 0
            else:
                row[f'{var}_frac'] = np.nan

        # Number of zone transitions (count zone changes)
        zone_seq = []
        for t_idx in range(len(seg_time)):
            for zone in ['Home', 'Ladder', 'Transition zone', 'Foraging arena']:
                if zone in behav_seg and not np.isnan(behav_seg[zone][t_idx]):
                    if behav_seg[zone][t_idx] > 0:
                        zone_seq.append(zone)
                        break
        n_transitions = sum(1 for i in range(1, len(zone_seq)) if zone_seq[i] != zone_seq[i-1])
        row['n_zone_transitions'] = n_transitions
        row['zone_transition_rate'] = n_transitions / erow['duration'] if erow['duration'] > 0 else 0

        # Number of distinct behaviors observed
        n_behaviors = sum(1 for var in BEHAVIOR_VARS
                          if f'{var}_any' in row and row[f'{var}_any'] == 1)
        row['n_distinct_behaviors'] = n_behaviors

        all_exc_data.append(row)

    exc_behav = pd.DataFrame(all_exc_data)
    exc_behav.to_csv("data/excursion_behavior_profiles.csv", index=False)
    print(f"\n  Saved: data/excursion_behavior_profiles.csv ({len(exc_behav)} excursions)")

    # =========================================================================
    # Print detailed profiles for significant excursions
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"  DETAILED BEHAVIOR PROFILES — Significant Excursions")
    print(f"{'='*60}")

    sig_exc = exc_behav[exc_behav['is_significant']].sort_values('excursion_id')
    for _, row in sig_exc.iterrows():
        eid = int(row['excursion_id'])
        print(f"\n  --- Excursion {eid} ({row['duration']:.1f}s) ---")

        # Zone occupancy
        print(f"    Zone occupancy:")
        for zone in ['Home', 'Ladder', 'Transition zone', 'Foraging arena']:
            frac = row.get(f'{zone}_frac', 0)
            if not np.isnan(frac) and frac > 0:
                print(f"      {zone}: {frac*100:.0f}%")

        # Pot zones
        for pot in ['Pot-1 zone', 'Pot-2 zone', 'Pot-3 zone', 'Pot-4 zone']:
            frac = row.get(f'{pot}_frac', 0)
            if not np.isnan(frac) and frac > 0:
                print(f"      {pot}: {frac*100:.0f}%")

        # Behaviors observed
        print(f"    Behaviors:")
        any_behavior = False
        for var in BEHAVIOR_VARS:
            frac = row.get(f'{var}_frac', 0)
            if not np.isnan(frac) and frac > 0:
                print(f"      {var}: {frac*100:.0f}% of time")
                any_behavior = True
        if not any_behavior:
            print(f"      (no scored behaviors)")

        # Movement
        print(f"    Movement:")
        vel = row.get('Velocity_mean', np.nan)
        if not np.isnan(vel):
            print(f"      Mean velocity: {vel:.2f}")
        dist = row.get('Distance moved_mean', np.nan)
        if not np.isnan(dist):
            print(f"      Mean distance/bin: {dist:.2f}")
        move_frac = row.get('Movement(Moving / Center-point)_frac', np.nan)
        if not np.isnan(move_frac):
            print(f"      Fraction moving: {move_frac*100:.0f}%")

        print(f"    Zone transitions: {int(row['n_zone_transitions'])} "
              f"(rate: {row['zone_transition_rate']:.2f}/s)")
        print(f"    Distinct behaviors: {int(row['n_distinct_behaviors'])}")

    # =========================================================================
    # Statistical comparison: significant vs non-significant
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"  STATISTICAL COMPARISON: Significant vs Other Excursions")
    print(f"{'='*60}")

    sig = exc_behav[exc_behav['is_significant']]
    nonsig = exc_behav[~exc_behav['is_significant']]

    comparison_vars = []

    # Binary behaviors
    print(f"\n  Binary Behaviors (fraction of time):")
    print(f"  {'Variable':<40} {'Sig (n=4)':>12} {'Other (n={})'.format(len(nonsig)):>12} {'U':>6} {'p':>8}")
    print(f"  {'-'*80}")

    for var in BEHAVIOR_VARS:
        col = f'{var}_frac'
        if col not in exc_behav.columns:
            continue
        sig_vals = sig[col].dropna()
        nonsig_vals = nonsig[col].dropna()
        if len(sig_vals) < 2 or len(nonsig_vals) < 2:
            continue

        sig_mean = sig_vals.mean()
        nonsig_mean = nonsig_vals.mean()

        try:
            U, p = stats.mannwhitneyu(sig_vals, nonsig_vals, alternative='two-sided')
        except:
            U, p = np.nan, np.nan

        marker = '*' if p < 0.05 else '**' if p < 0.01 else ''
        if p < 0.01:
            marker = '**'
        elif p < 0.05:
            marker = '*'
        else:
            marker = ''

        print(f"  {var:<40} {sig_mean:>10.3f}   {nonsig_mean:>10.3f}  {U:>5.0f}  {p:>7.4f} {marker}")
        comparison_vars.append({'variable': var, 'type': 'behavior',
                                'sig_mean': sig_mean, 'nonsig_mean': nonsig_mean,
                                'U': U, 'p': p})

    # Zone occupancy
    print(f"\n  Zone Occupancy (fraction of time):")
    print(f"  {'Variable':<40} {'Sig (n=4)':>12} {'Other (n={})'.format(len(nonsig)):>12} {'U':>6} {'p':>8}")
    print(f"  {'-'*80}")

    for zone in ZONE_VARS:
        col = f'{zone}_frac'
        if col not in exc_behav.columns:
            continue
        sig_vals = sig[col].dropna()
        nonsig_vals = nonsig[col].dropna()
        if len(sig_vals) < 2 or len(nonsig_vals) < 2:
            continue

        sig_mean = sig_vals.mean()
        nonsig_mean = nonsig_vals.mean()

        try:
            U, p = stats.mannwhitneyu(sig_vals, nonsig_vals, alternative='two-sided')
        except:
            U, p = np.nan, np.nan

        marker = '**' if p < 0.01 else '*' if p < 0.05 else ''
        print(f"  {zone:<40} {sig_mean:>10.3f}   {nonsig_mean:>10.3f}  {U:>5.0f}  {p:>7.4f} {marker}")
        comparison_vars.append({'variable': zone, 'type': 'zone',
                                'sig_mean': sig_mean, 'nonsig_mean': nonsig_mean,
                                'U': U, 'p': p})

    # Continuous variables
    print(f"\n  Continuous Variables (mean):")
    print(f"  {'Variable':<40} {'Sig (n=4)':>12} {'Other (n={})'.format(len(nonsig)):>12} {'U':>6} {'p':>8}")
    print(f"  {'-'*80}")

    for var in CONTINUOUS_VARS + ['n_zone_transitions', 'zone_transition_rate',
                                   'n_distinct_behaviors', 'duration']:
        col = f'{var}_mean' if var in CONTINUOUS_VARS else var
        if col not in exc_behav.columns:
            continue
        sig_vals = sig[col].dropna()
        nonsig_vals = nonsig[col].dropna()
        if len(sig_vals) < 2 or len(nonsig_vals) < 2:
            continue

        sig_mean = sig_vals.mean()
        nonsig_mean = nonsig_vals.mean()

        try:
            U, p = stats.mannwhitneyu(sig_vals, nonsig_vals, alternative='two-sided')
        except:
            U, p = np.nan, np.nan

        marker = '**' if p < 0.01 else '*' if p < 0.05 else ''
        print(f"  {var:<40} {sig_mean:>10.3f}   {nonsig_mean:>10.3f}  {U:>5.0f}  {p:>7.4f} {marker}")
        comparison_vars.append({'variable': var, 'type': 'continuous',
                                'sig_mean': sig_mean, 'nonsig_mean': nonsig_mean,
                                'U': U, 'p': p})

    # =========================================================================
    # Figure 1: Behavior profiles of significant excursions
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle("Behavioral Profiles of Topologically Significant Excursions\n"
                 "Session 1 (Fed) — RSP H1 loops with p < 0.05 vs time-shuffled null",
                 fontsize=14, fontweight='bold')

    # Panel A: Zone occupancy comparison
    ax = axes[0, 0]
    zones_to_plot = ['Home', 'Ladder', 'Transition zone', 'Foraging arena']
    x = np.arange(len(zones_to_plot))
    width = 0.35
    sig_fracs = [sig[f'{z}_frac'].mean() for z in zones_to_plot]
    nonsig_fracs = [nonsig[f'{z}_frac'].mean() for z in zones_to_plot]
    sig_errs = [sig[f'{z}_frac'].std() for z in zones_to_plot]
    nonsig_errs = [nonsig[f'{z}_frac'].std() for z in zones_to_plot]

    ax.bar(x - width/2, sig_fracs, width, yerr=sig_errs, capsize=4,
           color='#D32F2F', alpha=0.8, label=f'Significant (n={len(sig)})')
    ax.bar(x + width/2, nonsig_fracs, width, yerr=nonsig_errs, capsize=4,
           color='#90CAF9', alpha=0.8, label=f'Other (n={len(nonsig)})')
    ax.set_xticks(x)
    ax.set_xticklabels(zones_to_plot, fontsize=10)
    ax.set_ylabel('Fraction of excursion time', fontsize=11)
    ax.set_title('Zone Occupancy', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)

    # Panel B: Scored behaviors comparison
    ax = axes[0, 1]
    # Only show behaviors that occur in at least one excursion
    behav_to_plot = []
    for var in BEHAVIOR_VARS:
        col = f'{var}_frac'
        if col in exc_behav.columns and exc_behav[col].sum() > 0:
            behav_to_plot.append(var)

    if behav_to_plot:
        x = np.arange(len(behav_to_plot))
        sig_fracs = [sig[f'{v}_frac'].mean() for v in behav_to_plot]
        nonsig_fracs = [nonsig[f'{v}_frac'].mean() for v in behav_to_plot]

        ax.barh(x - width/2, sig_fracs, width, color='#D32F2F', alpha=0.8,
                label=f'Significant (n={len(sig)})')
        ax.barh(x + width/2, nonsig_fracs, width, color='#90CAF9', alpha=0.8,
                label=f'Other (n={len(nonsig)})')
        ax.set_yticks(x)
        ax.set_yticklabels(behav_to_plot, fontsize=8)
        ax.set_xlabel('Fraction of excursion time', fontsize=11)
        ax.set_title('Scored Behaviors', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
    else:
        ax.text(0.5, 0.5, 'No scored behaviors found', ha='center', va='center')

    # Panel C: Velocity and movement
    ax = axes[1, 0]
    # Boxplot of velocity, distance, zone transitions
    plot_vars = []
    plot_labels = []
    for var, label in [('Velocity_mean', 'Mean Velocity'),
                       ('zone_transition_rate', 'Zone Trans./s'),
                       ('n_distinct_behaviors', '# Behaviors'),
                       ('Movement(Moving / Center-point)_frac', 'Frac Moving')]:
        if var in exc_behav.columns:
            plot_vars.append(var)
            plot_labels.append(label)

    if plot_vars:
        positions = np.arange(len(plot_vars))
        bp_width = 0.3
        for i, var in enumerate(plot_vars):
            sig_vals = sig[var].dropna().values
            nonsig_vals = nonsig[var].dropna().values

            # Normalize for comparison
            all_vals = np.concatenate([sig_vals, nonsig_vals])
            vmax = np.max(np.abs(all_vals)) if len(all_vals) > 0 else 1
            if vmax == 0:
                vmax = 1

            bp1 = ax.boxplot([sig_vals], positions=[i - 0.2], widths=bp_width,
                             patch_artist=True,
                             boxprops=dict(facecolor='#D32F2F', alpha=0.6),
                             medianprops=dict(color='black'))
            bp2 = ax.boxplot([nonsig_vals], positions=[i + 0.2], widths=bp_width,
                             patch_artist=True,
                             boxprops=dict(facecolor='#90CAF9', alpha=0.6),
                             medianprops=dict(color='black'))

            # Individual points for significant
            ax.scatter([i - 0.2] * len(sig_vals), sig_vals, c='#D32F2F',
                       s=30, zorder=5, alpha=0.8)

        ax.set_xticks(positions)
        ax.set_xticklabels(plot_labels, fontsize=10)
        ax.set_title('Movement & Complexity', fontsize=12, fontweight='bold')

    # Panel D: Individual excursion timelines
    ax = axes[1, 1]
    sig_ids = sorted(SIGNIFICANT_EXCURSIONS)
    colors_behav = {
        'Feeding': '#D32F2F', 'Digging': '#FF9800', 'Grooming': '#4CAF50',
        'Arena wall exploration': '#9C27B0', 'Transition wall exploration': '#2196F3',
        'Quick arena exploration': '#00BCD4', 'Hesitant exploration': '#795548',
        'Random switching between pots': '#E91E63',
        'Intentional switching between pots': '#3F51B5',
    }

    y_pos = 0
    ytick_labels = []
    ytick_pos = []

    for eid in sig_ids:
        erow = complete[complete['excursion_id'] == eid].iloc[0]
        behav_seg, seg_time = get_behavior_during_excursion(
            variables, behav_time, erow['start_time'], erow['end_time'])

        t_rel = seg_time - seg_time[0]

        # Zone bar
        for zone, color in [('Home', '#4CAF50'), ('Ladder', '#FF9800'),
                             ('Transition zone', '#9C27B0'), ('Foraging arena', '#D32F2F')]:
            if zone in behav_seg:
                zone_data = behav_seg[zone]
                active = zone_data > 0
                if active.any():
                    starts = np.where(np.diff(np.concatenate([[0], active.astype(int)])) == 1)[0]
                    ends = np.where(np.diff(np.concatenate([active.astype(int), [0]])) == -1)[0]
                    for s, e in zip(starts, ends):
                        ax.barh(y_pos, (e - s) * 0.1, left=s * 0.1, height=0.4,
                                color=color, alpha=0.6)

        ytick_labels.append(f'Exc {eid} zones')
        ytick_pos.append(y_pos)
        y_pos += 0.6

        # Behavior bar
        for var, color in colors_behav.items():
            if var in behav_seg:
                bdata = behav_seg[var]
                active = bdata > 0
                if active.any():
                    starts = np.where(np.diff(np.concatenate([[0], active.astype(int)])) == 1)[0]
                    ends = np.where(np.diff(np.concatenate([active.astype(int), [0]])) == -1)[0]
                    for s, e in zip(starts, ends):
                        ax.barh(y_pos, (e - s) * 0.1, left=s * 0.1, height=0.4,
                                color=color, alpha=0.8)

        ytick_labels.append(f'Exc {eid} behav')
        ytick_pos.append(y_pos)
        y_pos += 1.0

    ax.set_yticks(ytick_pos)
    ax.set_yticklabels(ytick_labels, fontsize=8)
    ax.set_xlabel('Time within excursion (s)', fontsize=10)
    ax.set_title('Excursion Timelines\n(zones + scored behaviors)', fontsize=12, fontweight='bold')

    # Create legend for zones + behaviors
    from matplotlib.patches import Patch
    zone_handles = [Patch(facecolor='#4CAF50', alpha=0.6, label='Home'),
                    Patch(facecolor='#FF9800', alpha=0.6, label='Ladder'),
                    Patch(facecolor='#9C27B0', alpha=0.6, label='Transition'),
                    Patch(facecolor='#D32F2F', alpha=0.6, label='Foraging')]
    behav_handles = [Patch(facecolor=c, alpha=0.8, label=n)
                     for n, c in list(colors_behav.items())[:6]]
    ax.legend(handles=zone_handles + behav_handles, fontsize=6,
              loc='upper right', ncol=2)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig("figures/excursion_behavior_profiles.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: figures/excursion_behavior_profiles.png")

    # =========================================================================
    # Figure 2: Heatmap of all behaviors across ALL excursions
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(22, 10))
    fig.suptitle("Behavioral Heatmap — All Complete Excursions (Session 1)\n"
                 "Red borders = topologically significant excursions",
                 fontsize=14, fontweight='bold')

    # Sort by excursion_id
    exc_sorted = exc_behav.sort_values('excursion_id')

    for panel, (var_list, title) in enumerate([
        (BEHAVIOR_VARS, 'Scored Behaviors'),
        (ZONE_VARS[:4] + ['Pot-2 zone', 'Pot-3 zone', 'Pot-4 zone'], 'Zone Occupancy'),
    ]):
        ax = axes[panel]
        cols = [f'{v}_frac' for v in var_list if f'{v}_frac' in exc_sorted.columns]
        labels = [v for v in var_list if f'{v}_frac' in exc_sorted.columns]

        if not cols:
            continue

        matrix = exc_sorted[cols].values
        exc_ids = exc_sorted['excursion_id'].values
        is_sig = exc_sorted['is_significant'].values

        im = ax.imshow(matrix.T, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1,
                       interpolation='nearest')

        # Highlight significant excursions
        for i, eid in enumerate(exc_ids):
            if is_sig[i]:
                ax.axvline(i, color='blue', linewidth=2, alpha=0.6)
                ax.text(i, -0.7, f'{int(eid)}', ha='center', va='bottom',
                        fontsize=7, fontweight='bold', color='blue')

        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel('Excursion (ordered by ID)', fontsize=10)
        ax.set_title(title, fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=ax, label='Fraction of time', shrink=0.8)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig("figures/excursion_behavior_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: figures/excursion_behavior_heatmap.png")

    print("\nDone!")


if __name__ == "__main__":
    main()
