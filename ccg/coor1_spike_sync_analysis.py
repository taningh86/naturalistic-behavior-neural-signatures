"""
Continuous-time SPIKE-synchronization aligned with behavior.
M1 Coordinates-1, all 8 sessions.
Unit selection: KSLabel=='good' AND fr>0.3 AND amp>48

Uses PySpike's adaptive coincidence detection (no fixed window parameter).
Profiles binned to 100ms for behavior alignment.
"""

import yaml
from pathlib import Path
import numpy as np
import pandas as pd
import pyspike
import spikeinterface.extractors as se
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import warnings
import time as timer

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIG
# =============================================================================
with open("paths.yaml") as f:
    cfg = yaml.safe_load(f)

FS = 30000
LHA_DEPTH_MAX = 1300
MIN_FR = 0.3
MIN_AMP = 48
BIN_SEC = 0.1  # 100ms bins for behavior alignment
SMOOTH_SEC = 5.0  # smoothing window for visualization
SMOOTH_BINS = int(SMOOTH_SEC / BIN_SEC)

sessions_cfg = cfg["single_probe"]["coordinates_1"]["mouse01"]["sessions"]
session_meta = {
    1: ('fed', 'exploration'), 2: ('fed', 'foraging'),
    3: ('fed', 'exploration'), 4: ('fed', 'foraging'),
    5: ('fasted', 'exploration'), 6: ('fasted', 'foraging'),
    7: ('fasted', 'exploration'), 8: ('fasted', 'foraging'),
}

NETWORK_COLORS = {'LHA-LHA': '#e74c3c', 'RSP-RSP': '#27ae60', 'LHA-RSP': '#2980b9'}
STATE_COLORS = {'fed': '#3498db', 'fasted': '#e74c3c'}
ZONE_COLORS = {
    'Pot-1': '#ff9999', 'Pot-2': '#ff6666', 'Pot-3': '#ff3333', 'Pot-4': '#cc0000',
    'Pot-1 zone': '#ffcccc', 'Pot-2 zone': '#ffaaaa',
    'Pot-3 zone': '#ff8888', 'Pot-4 zone': '#ff5555',
    'Home': '#6699cc', 'Ladder': '#99cc66', 'Transition': '#cccc66',
    'Right corner': '#cc99cc', 'other': '#e0e0e0',
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def get_good_units(sorted_path):
    """Select units passing triple criteria, split by region."""
    ci = Path(sorted_path) / "cluster_info.tsv"
    df = pd.read_csv(ci, sep='\t')
    label_col = 'group' if ('group' in df.columns and df['group'].eq('good').any()) else 'KSLabel'
    good = df[(df[label_col] == 'good') & (df['fr'] > MIN_FR) & (df['amp'] > MIN_AMP)]
    lha = good[good['depth'] < LHA_DEPTH_MAX]['cluster_id'].values
    rsp = good[good['depth'] >= LHA_DEPTH_MAX]['cluster_id'].values
    print(f"    Units: {len(df)} total -> {len(good)} pass -> {len(lha)} LHA + {len(rsp)} RSP")
    return lha, rsp


def load_behavior(behav_path):
    """Load transposed EthoVision CSV. Returns dict of variable_name -> array."""
    df_raw = pd.read_csv(behav_path, header=None)
    var_names = df_raw.iloc[:, 0].values
    time_vals = df_raw.iloc[1, 1:].astype(float).values
    data = df_raw.iloc[:, 1:].values

    behav = {'time': time_vals}
    for i, name in enumerate(var_names):
        if isinstance(name, str):
            behav[name.strip()] = data[i].astype(float)
    return behav


def get_zone_labels(behav):
    """Assign a single zone label per time bin (highest priority wins)."""
    n = len(behav['time'])
    zones = np.full(n, 'other', dtype=object)

    # Apply in order: later entries overwrite earlier (higher priority)
    priority_order = [
        'Right corner', 'Home', 'Ladder', 'Transition zone',
        'Pot-1 zone', 'Pot-2 zone', 'Pot-3 zone', 'Pot-4 zone',
        'Pot-1', 'Pot-2', 'Pot-3', 'Pot-4',
    ]
    label_map = {'Transition zone': 'Transition'}

    for var_name in priority_order:
        if var_name in behav:
            mask = behav[var_name] > 0.5
            label = label_map.get(var_name, var_name)
            zones[mask] = label
    return zones


def smooth_trace(trace, window):
    """Centered moving average, NaN-safe."""
    if window <= 1:
        return trace
    kernel = np.ones(window) / window
    return np.convolve(trace, kernel, mode='same')


# =============================================================================
# MAIN ANALYSIS
# =============================================================================
all_pair_results = []
all_session_traces = {}
all_behavior_sync = []

for snum in range(1, 9):
    sc = sessions_cfg[f"session_{snum}"]
    sp = Path(sc['sorted'])
    state, phase = session_meta[snum]

    print(f"\n{'='*70}")
    print(f"Session {snum} ({state}/{phase})")
    print(f"{'='*70}")

    # --- Unit selection ---
    lha_ids, rsp_ids = get_good_units(sp)
    sorting = se.read_kilosort(sp)
    avail = set(sorting.get_unit_ids())
    lha_ids = np.array([u for u in lha_ids if u in avail])
    rsp_ids = np.array([u for u in rsp_ids if u in avail])
    all_ids = np.concatenate([lha_ids, rsp_ids])

    if len(all_ids) < 2:
        print("  SKIP — too few units")
        continue

    # --- Spike trains in seconds ---
    spike_trains = {}
    t_min, t_max = np.inf, 0
    for uid in all_ids:
        st = sorting.get_unit_spike_train(uid) / FS
        spike_trains[uid] = st
        if len(st) > 0:
            t_min = min(t_min, st[0])
            t_max = max(t_max, st[-1])

    edges = [t_min, t_max]
    py_trains = {}
    for uid in all_ids:
        py_trains[uid] = pyspike.SpikeTrain(spike_trains[uid], edges=edges)

    # --- Time bins for behavior alignment ---
    time_bins = np.arange(t_min, t_max, BIN_SEC)
    n_bins = len(time_bins)

    # Running accumulators: sum of coincidences / sum of possible coincidences
    net_y_sum = {nt: np.zeros(n_bins) for nt in ['LHA-LHA', 'RSP-RSP', 'LHA-RSP']}
    net_mp_sum = {nt: np.zeros(n_bins) for nt in ['LHA-LHA', 'RSP-RSP', 'LHA-RSP']}
    net_n_pairs = {nt: 0 for nt in ['LHA-LHA', 'RSP-RSP', 'LHA-RSP']}

    # --- Build pair list ---
    pair_list = []
    n_lha = len(lha_ids)
    for i in range(n_lha):
        for j in range(i + 1, n_lha):
            pair_list.append((lha_ids[i], lha_ids[j], 'LHA-LHA'))
    for i in range(len(rsp_ids)):
        for j in range(i + 1, len(rsp_ids)):
            pair_list.append((rsp_ids[i], rsp_ids[j], 'RSP-RSP'))
    for lid in lha_ids:
        for rid in rsp_ids:
            pair_list.append((lid, rid, 'LHA-RSP'))

    # --- Compute pairwise SPIKE-synchronization ---
    t0 = timer.time()
    for pi, (uid_a, uid_b, nt) in enumerate(pair_list):
        profile = pyspike.spike_sync_profile(py_trains[uid_a], py_trains[uid_b])
        overall_sync = profile.avrg()

        all_pair_results.append({
            'session': snum, 'state': state, 'phase': phase,
            'network': nt, 'unit_a': int(uid_a), 'unit_b': int(uid_b),
            'sync_value': overall_sync,
        })

        # Bin profile: x=spike times, y=coincidence count, mp=multiplicity
        x_arr = np.array(profile.x)
        y_arr = np.array(profile.y)
        mp_arr = np.array(profile.mp)

        if len(x_arr) > 0:
            bin_idx = np.digitize(x_arr, time_bins) - 1
            valid = (bin_idx >= 0) & (bin_idx < n_bins)
            bi = bin_idx[valid]
            yi = y_arr[valid]
            mi = mp_arr[valid]
            np.add.at(net_y_sum[nt], bi, yi)
            np.add.at(net_mp_sum[nt], bi, mi)

        net_n_pairs[nt] += 1

        if (pi + 1) % 2000 == 0:
            print(f"    {pi+1}/{len(pair_list)} pairs...", flush=True)

    elapsed = timer.time() - t0
    print(f"    {len(pair_list)} pairs in {elapsed:.1f}s")

    # --- Mean sync traces per network ---
    traces = {}
    for nt in ['LHA-LHA', 'RSP-RSP', 'LHA-RSP']:
        if net_n_pairs[nt] > 0 and np.sum(net_mp_sum[nt]) > 0:
            with np.errstate(divide='ignore', invalid='ignore'):
                trace = net_y_sum[nt] / net_mp_sum[nt]
            trace = np.nan_to_num(trace, nan=0.0)
            traces[nt] = trace

    all_session_traces[snum] = {
        'time_bins': time_bins, 'traces': traces,
        'state': state, 'phase': phase, 'n_pairs': net_n_pairs,
    }

    # --- Behavior alignment ---
    behav_path = sc.get('behavior')
    if behav_path and Path(behav_path).exists():
        behav = load_behavior(behav_path)
        behav_time = behav['time']
        zones = get_zone_labels(behav)

        # Map sync time bins to behavior indices
        sync_to_behav = np.searchsorted(behav_time, time_bins)
        sync_to_behav = np.clip(sync_to_behav, 0, len(behav_time) - 1)
        aligned_zones = zones[sync_to_behav]

        # Movement variable
        move_key = None
        for k in behav:
            if 'Movement' in str(k) or 'Moving' in str(k):
                move_key = k
                break

        for nt, trace in traces.items():
            # Sync by zone
            for zone in np.unique(aligned_zones):
                mask = aligned_zones == zone
                if mask.sum() > 10:
                    all_behavior_sync.append({
                        'session': snum, 'state': state, 'phase': phase,
                        'network': nt, 'behavior_type': 'zone',
                        'behavior_value': zone,
                        'mean_sync': np.nanmean(trace[mask]),
                        'std_sync': np.nanstd(trace[mask]),
                        'n_bins': int(mask.sum()),
                    })

            # Sync by movement
            if move_key:
                movement = behav[move_key][sync_to_behav]
                for mv_label, mv_mask in [('moving', movement > 0.5),
                                           ('stationary', movement <= 0.5)]:
                    if mv_mask.sum() > 10:
                        all_behavior_sync.append({
                            'session': snum, 'state': state, 'phase': phase,
                            'network': nt, 'behavior_type': 'movement',
                            'behavior_value': mv_label,
                            'mean_sync': np.nanmean(trace[mv_mask]),
                            'std_sync': np.nanstd(trace[mv_mask]),
                            'n_bins': int(mv_mask.sum()),
                        })

        print(f"    Behavior aligned: {len(np.unique(zones))} zones found")
    else:
        print(f"    No behavior data available")


# =============================================================================
# SAVE RESULTS
# =============================================================================
pairs_df = pd.DataFrame(all_pair_results)
pairs_df.to_csv("data/coor1_spike_sync_pairs.csv", index=False)

behav_sync_df = pd.DataFrame(all_behavior_sync)
behav_sync_df.to_csv("data/coor1_spike_sync_by_behavior.csv", index=False)

trace_data = {}
for snum, info in all_session_traces.items():
    trace_data[f's{snum}_time'] = info['time_bins']
    for nt, tr in info['traces'].items():
        trace_data[f's{snum}_{nt}'] = tr
np.savez("data/coor1_spike_sync_traces.npz", **trace_data)

print(f"\nSaved {len(pairs_df)} pair results to data/coor1_spike_sync_pairs.csv")
print(f"Saved {len(behav_sync_df)} behavior-sync rows to data/coor1_spike_sync_by_behavior.csv")
print(f"Saved traces to data/coor1_spike_sync_traces.npz")


# =============================================================================
# PRINT SUMMARIES
# =============================================================================
print("\n" + "=" * 80)
print("OVERALL SPIKE-SYNCHRONIZATION BY SESSION AND NETWORK")
print("=" * 80)
for snum in range(1, 9):
    state, phase = session_meta[snum]
    s_df = pairs_df[pairs_df['session'] == snum]
    print(f"\n  S{snum} ({state}/{phase}):")
    for nt in ['LHA-LHA', 'RSP-RSP', 'LHA-RSP']:
        sub = s_df[s_df['network'] == nt]
        if len(sub) > 0:
            print(f"    {nt}: n={len(sub)}, mean={sub['sync_value'].mean():.6f}, "
                  f"median={sub['sync_value'].median():.6f}, max={sub['sync_value'].max():.6f}")

print("\n" + "=" * 80)
print("SYNCHRONY BY ZONE (session means)")
print("=" * 80)
if len(behav_sync_df) > 0:
    zone_df = behav_sync_df[behav_sync_df['behavior_type'] == 'zone']
    for nt in ['LHA-LHA', 'RSP-RSP', 'LHA-RSP']:
        nt_df = zone_df[zone_df['network'] == nt]
        if len(nt_df) > 0:
            print(f"\n  {nt}:")
            pivot = nt_df.pivot_table(values='mean_sync', index='behavior_value',
                                       columns='state', aggfunc=['mean', 'count'])
            print(pivot.to_string())

print("\n" + "=" * 80)
print("SYNCHRONY: MOVING VS STATIONARY (session means)")
print("=" * 80)
if len(behav_sync_df) > 0:
    mv_df = behav_sync_df[behav_sync_df['behavior_type'] == 'movement']
    for nt in ['LHA-LHA', 'RSP-RSP', 'LHA-RSP']:
        nt_df = mv_df[mv_df['network'] == nt]
        if len(nt_df) > 0:
            print(f"\n  {nt}:")
            pivot = nt_df.pivot_table(values='mean_sync', index='behavior_value',
                                       columns='state', aggfunc=['mean', 'count'])
            print(pivot.to_string())


# =============================================================================
# FIGURE 1: Per-session continuous traces with behavior overlay
# =============================================================================
print("\n" + "=" * 80)
print("GENERATING FIGURES")
print("=" * 80)

for snum in range(1, 9):
    if snum not in all_session_traces:
        continue

    info = all_session_traces[snum]
    sc = sessions_cfg[f"session_{snum}"]
    state, phase = session_meta[snum]
    traces = info['traces']
    time_bins = info['time_bins']

    behav_path = sc.get('behavior')
    has_behav = behav_path and Path(behav_path).exists()
    n_rows = len(traces) + (1 if has_behav else 0)
    if n_rows == 0:
        continue

    fig, axes = plt.subplots(n_rows, 1, figsize=(16, 2.5 * n_rows),
                              sharex=True, gridspec_kw={'hspace': 0.15})
    if n_rows == 1:
        axes = [axes]

    row = 0
    for nt in ['LHA-LHA', 'RSP-RSP', 'LHA-RSP']:
        if nt not in traces:
            continue
        ax = axes[row]
        trace = traces[nt]
        smoothed = smooth_trace(trace, SMOOTH_BINS)

        ax.plot(time_bins, smoothed, color=NETWORK_COLORS[nt], linewidth=0.7, alpha=0.9)
        ax.axhline(np.nanmean(trace), color='gray', linestyle='--', alpha=0.4, linewidth=0.5)
        ax.set_ylabel(f'{nt}\nSync', fontsize=9)
        n_p = info['n_pairs'][nt]
        ax.set_title(f'{nt} ({n_p} pairs)', fontsize=10, loc='left')
        row += 1

    # Behavior color strip
    if has_behav:
        behav = load_behavior(behav_path)
        behav_time = behav['time']
        zones = get_zone_labels(behav)

        ax_beh = axes[row]
        prev_zone = zones[0]
        span_start = behav_time[0]
        for ti in range(1, len(zones)):
            if zones[ti] != prev_zone or ti == len(zones) - 1:
                c = ZONE_COLORS.get(prev_zone, '#e0e0e0')
                ax_beh.axvspan(span_start, behav_time[min(ti, len(behav_time)-1)],
                               alpha=0.7, color=c, linewidth=0)
                prev_zone = zones[ti]
                span_start = behav_time[ti] if ti < len(behav_time) else behav_time[-1]

        ax_beh.set_yticks([])
        ax_beh.set_ylabel('Zone', fontsize=9)
        unique_zones = sorted([z for z in np.unique(zones) if z != 'other'])
        patches = [Patch(color=ZONE_COLORS.get(z, '#e0e0e0'), label=z) for z in unique_zones]
        ax_beh.legend(handles=patches, loc='upper right', fontsize=7, ncol=min(6, len(patches)))

    axes[-1].set_xlabel('Time (s)', fontsize=10)
    fig.suptitle(f'S{snum} ({state}/{phase}) — SPIKE-Synchronization (smoothed {SMOOTH_SEC:.0f}s)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f'figures/spike_sync_continuous_s{snum}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved figures/spike_sync_continuous_s{snum}.png")


# =============================================================================
# FIGURE 2: Sync by zone, fed vs fasted
# =============================================================================
if len(behav_sync_df) > 0:
    zone_df = behav_sync_df[behav_sync_df['behavior_type'] == 'zone']

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ai, nt in enumerate(['LHA-LHA', 'RSP-RSP', 'LHA-RSP']):
        ax = axes[ai]
        nt_df = zone_df[zone_df['network'] == nt]

        # Zones present in at least 2 sessions
        zone_counts = nt_df.groupby('behavior_value')['session'].nunique()
        common_zones = zone_counts[zone_counts >= 2].index.tolist()
        zone_order = (nt_df[nt_df['behavior_value'].isin(common_zones)]
                      .groupby('behavior_value')['mean_sync']
                      .mean().sort_values(ascending=False).index.tolist())

        x_pos = np.arange(len(zone_order))
        width = 0.35

        for si, s in enumerate(['fed', 'fasted']):
            s_df = nt_df[(nt_df['state'] == s) & (nt_df['behavior_value'].isin(zone_order))]
            means, sems = [], []
            for z in zone_order:
                z_vals = s_df[s_df['behavior_value'] == z]['mean_sync']
                means.append(z_vals.mean() if len(z_vals) > 0 else 0)
                sems.append(z_vals.std() / np.sqrt(len(z_vals)) if len(z_vals) > 1 else 0)

            offset = (si - 0.5) * width
            bars = ax.bar(x_pos + offset, means, width, yerr=sems,
                          label=s, color=STATE_COLORS[s], alpha=0.8, capsize=3)

            # Individual session points
            for zi, z in enumerate(zone_order):
                z_vals = s_df[s_df['behavior_value'] == z]['mean_sync'].values
                ax.scatter([x_pos[zi] + offset] * len(z_vals), z_vals,
                           color='black', s=15, zorder=5, alpha=0.6)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(zone_order, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Mean SPIKE-Sync', fontsize=10)
        ax.set_title(nt, fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)

    fig.suptitle('SPIKE-Synchronization by Zone and Metabolic State',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('figures/spike_sync_by_zone.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved figures/spike_sync_by_zone.png")


# =============================================================================
# FIGURE 3: Moving vs stationary
# =============================================================================
if len(behav_sync_df) > 0:
    mv_df = behav_sync_df[behav_sync_df['behavior_type'] == 'movement']
    if len(mv_df) > 0:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for ai, nt in enumerate(['LHA-LHA', 'RSP-RSP', 'LHA-RSP']):
            ax = axes[ai]
            nt_df = mv_df[mv_df['network'] == nt]

            positions = []
            labels = []
            for si, s in enumerate(['fed', 'fasted']):
                for mi, mv in enumerate(['stationary', 'moving']):
                    vals = nt_df[(nt_df['state'] == s) &
                                (nt_df['behavior_value'] == mv)]['mean_sync']
                    x = si * 2.5 + mi
                    positions.append(x)
                    labels.append(f'{s[:3].title()}\n{mv[:4].title()}')
                    mean_v = vals.mean() if len(vals) > 0 else 0
                    sem_v = vals.std() / np.sqrt(len(vals)) if len(vals) > 1 else 0
                    alpha = 0.9 if mv == 'stationary' else 0.5
                    ax.bar(x, mean_v, 0.8, yerr=sem_v,
                           color=STATE_COLORS[s], alpha=alpha, capsize=3)
                    ax.scatter([x] * len(vals), vals, color='black', s=20, zorder=5, alpha=0.6)

            ax.set_xticks(positions)
            ax.set_xticklabels(labels, fontsize=9)
            ax.set_ylabel('Mean SPIKE-Sync', fontsize=10)
            ax.set_title(nt, fontsize=12, fontweight='bold')

        fig.suptitle('SPIKE-Synchronization: Moving vs Stationary',
                     fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig('figures/spike_sync_movement.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  Saved figures/spike_sync_movement.png")

print("\n[DONE]")
