"""
Kernel-smoothed SPIKE-synchronization using only top-sync pairs.
Filters to top 25% of pairs by overall SPIKE-sync value per session/network.
Removes the noise floor from non-synchronous pairs.
"""

import yaml
from pathlib import Path
import numpy as np
import pandas as pd
import pyspike
import spikeinterface.extractors as se
from scipy.ndimage import gaussian_filter1d
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

EVAL_DT = 0.05
SIGMA_SEC = 2.0
SIGMA_BINS = int(SIGMA_SEC / EVAL_DT)
EPS = 1e-10
TOP_PCT = 10  # keep top 10% of pairs by sync value

sessions_cfg = cfg["single_probe"]["coordinates_1"]["mouse01"]["sessions"]
session_meta = {
    1: ('fed', 'exploration'), 2: ('fed', 'foraging'),
    3: ('fed', 'exploration'), 4: ('fed', 'foraging'),
    5: ('fasted', 'exploration'), 6: ('fasted', 'foraging'),
    7: ('fasted', 'exploration'), 8: ('fasted', 'foraging'),
}

NETWORK_COLORS = {'LHA-LHA': '#e74c3c', 'RSP-RSP': '#27ae60', 'LHA-RSP': '#2980b9'}
ZONE_COLORS = {
    'Pot-1': '#ff9999', 'Pot-2': '#ff6666', 'Pot-3': '#ff3333', 'Pot-4': '#cc0000',
    'Pot-1 zone': '#ffcccc', 'Pot-2 zone': '#ffaaaa',
    'Pot-3 zone': '#ff8888', 'Pot-4 zone': '#ff5555',
    'Home': '#6699cc', 'Ladder': '#99cc66', 'Transition': '#cccc66',
    'Right corner': '#cc99cc', 'Left corner': '#bb88bb',
    'Arna center': '#dddd88', 'Foraging arena': '#88bb88',
    'other': '#e0e0e0',
}


# =============================================================================
# HELPERS
# =============================================================================
def get_good_units(sorted_path):
    ci = Path(sorted_path) / "cluster_info.tsv"
    df = pd.read_csv(ci, sep='\t')
    label_col = 'group' if ('group' in df.columns and df['group'].eq('good').any()) else 'KSLabel'
    good = df[(df[label_col] == 'good') & (df['fr'] > MIN_FR) & (df['amp'] > MIN_AMP)]
    lha = good[good['depth'] < LHA_DEPTH_MAX]['cluster_id'].values
    rsp = good[good['depth'] >= LHA_DEPTH_MAX]['cluster_id'].values
    return lha, rsp


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


def get_zone_labels(behav):
    n = len(behav['time'])
    zones = np.full(n, 'other', dtype=object)
    priority_order = [
        'Right corner', 'Left corner', 'Arna center', 'Foraging arena',
        'Home', 'Ladder', 'Transition zone',
        'Pot-1 zone', 'Pot-2 zone', 'Pot-3 zone', 'Pot-4 zone',
        'Pot-1', 'Pot-2', 'Pot-3', 'Pot-4',
    ]
    label_map = {'Transition zone': 'Transition'}
    for var_name in priority_order:
        if var_name in behav:
            mask = behav[var_name] > 0.5
            zones[mask] = label_map.get(var_name, var_name)
    return zones


# =============================================================================
# LOAD PER-PAIR SYNC VALUES (from previous analysis)
# =============================================================================
pairs_df = pd.read_csv("data/coor1_spike_sync_pairs.csv")
print(f"Loaded {len(pairs_df)} pair sync values")

# Determine threshold per session/network
print(f"\nTop {TOP_PCT}% thresholds:")
thresholds = {}
for snum in range(1, 9):
    state, phase = session_meta[snum]
    for nt in ['LHA-LHA', 'RSP-RSP', 'LHA-RSP']:
        sub = pairs_df[(pairs_df['session'] == snum) & (pairs_df['network'] == nt)]
        if len(sub) == 0:
            continue
        thresh = np.percentile(sub['sync_value'], 100 - TOP_PCT)
        n_keep = (sub['sync_value'] >= thresh).sum()
        thresholds[(snum, nt)] = thresh
        print(f"  S{snum} {nt}: threshold={thresh:.4f}, keeping {n_keep}/{len(sub)} pairs")

# Build set of top pairs for quick lookup
top_pairs = set()
for snum in range(1, 9):
    for nt in ['LHA-LHA', 'RSP-RSP', 'LHA-RSP']:
        if (snum, nt) not in thresholds:
            continue
        thresh = thresholds[(snum, nt)]
        sub = pairs_df[(pairs_df['session'] == snum) & (pairs_df['network'] == nt)
                       & (pairs_df['sync_value'] >= thresh)]
        for _, r in sub.iterrows():
            top_pairs.add((snum, int(r['unit_a']), int(r['unit_b'])))


# =============================================================================
# COMPUTE KERNEL-SMOOTHED TRACES (top pairs only)
# =============================================================================
all_session_data = {}

for snum in range(1, 9):
    sc = sessions_cfg[f"session_{snum}"]
    sp = Path(sc['sorted'])
    state, phase = session_meta[snum]

    print(f"\n{'='*70}")
    print(f"Session {snum} ({state}/{phase})")
    print(f"{'='*70}")

    lha_ids, rsp_ids = get_good_units(sp)
    sorting = se.read_kilosort(sp)
    avail = set(sorting.get_unit_ids())
    lha_ids = np.array([u for u in lha_ids if u in avail])
    rsp_ids = np.array([u for u in rsp_ids if u in avail])
    all_ids = np.concatenate([lha_ids, rsp_ids])

    if len(all_ids) < 2:
        print("  SKIP")
        continue

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

    eval_time = np.arange(t_min, t_max, EVAL_DT)
    n_eval = len(eval_time)

    coinc_hist = {nt: np.zeros(n_eval) for nt in ['LHA-LHA', 'RSP-RSP', 'LHA-RSP']}
    total_hist = {nt: np.zeros(n_eval) for nt in ['LHA-LHA', 'RSP-RSP', 'LHA-RSP']}
    net_n_pairs = {nt: 0 for nt in ['LHA-LHA', 'RSP-RSP', 'LHA-RSP']}
    net_n_total = {nt: 0 for nt in ['LHA-LHA', 'RSP-RSP', 'LHA-RSP']}

    # Build pair list
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

    t0 = timer.time()
    for pi, (uid_a, uid_b, nt) in enumerate(pair_list):
        net_n_total[nt] += 1

        # Skip if not a top pair
        if (snum, uid_a, uid_b) not in top_pairs and (snum, uid_b, uid_a) not in top_pairs:
            continue

        profile = pyspike.spike_sync_profile(py_trains[uid_a], py_trains[uid_b])
        x_arr = np.array(profile.x)
        y_arr = np.array(profile.y)
        mp_arr = np.array(profile.mp)

        if len(x_arr) == 0:
            continue

        bin_idx = np.floor((x_arr - t_min) / EVAL_DT).astype(int)
        valid = (bin_idx >= 0) & (bin_idx < n_eval)
        bi = bin_idx[valid]

        np.add.at(coinc_hist[nt], bi, y_arr[valid])
        np.add.at(total_hist[nt], bi, mp_arr[valid])
        net_n_pairs[nt] += 1

        if (pi + 1) % 2000 == 0:
            print(f"    {pi+1}/{len(pair_list)} checked...", flush=True)

    elapsed = timer.time() - t0
    for nt in ['LHA-LHA', 'RSP-RSP', 'LHA-RSP']:
        print(f"    {nt}: {net_n_pairs[nt]}/{net_n_total[nt]} pairs kept (top {TOP_PCT}%)")
    print(f"    Done in {elapsed:.1f}s")

    # Kernel smoothing
    traces = {}
    for nt in ['LHA-LHA', 'RSP-RSP', 'LHA-RSP']:
        if net_n_pairs[nt] == 0:
            continue
        smooth_coinc = gaussian_filter1d(coinc_hist[nt], sigma=SIGMA_BINS)
        smooth_total = gaussian_filter1d(total_hist[nt], sigma=SIGMA_BINS)
        sync_trace = smooth_coinc / (smooth_total + EPS)
        sync_trace[smooth_total < EPS * 100] = np.nan
        traces[nt] = sync_trace
        valid_vals = sync_trace[np.isfinite(sync_trace)]
        print(f"    {nt}: mean={np.nanmean(valid_vals):.4f}, "
              f"range=[{np.nanmin(valid_vals):.4f}, {np.nanmax(valid_vals):.4f}]")

    all_session_data[snum] = {
        'eval_time': eval_time, 'traces': traces,
        'state': state, 'phase': phase, 'n_pairs': net_n_pairs,
    }


# =============================================================================
# SAVE
# =============================================================================
trace_data = {}
for snum, info in all_session_data.items():
    trace_data[f's{snum}_time'] = info['eval_time']
    for nt, tr in info['traces'].items():
        trace_data[f's{snum}_{nt}'] = tr
np.savez("data/coor1_spike_sync_kernel_top10_traces.npz", **trace_data)
print(f"\nSaved to data/coor1_spike_sync_kernel_top10_traces.npz")


# =============================================================================
# FIGURES
# =============================================================================
print("\n" + "=" * 70)
print("GENERATING FIGURES")
print("=" * 70)

for snum in range(1, 9):
    if snum not in all_session_data:
        continue

    info = all_session_data[snum]
    sc = sessions_cfg[f"session_{snum}"]
    state, phase = session_meta[snum]
    traces = info['traces']
    eval_time = info['eval_time']

    behav_path = sc.get('behavior')
    has_behav = behav_path and Path(behav_path).exists()

    n_net = len(traces)
    n_rows = n_net + (2 if has_behav else 0)
    if n_rows == 0:
        continue

    height_ratios = [3] * n_net + ([1, 1] if has_behav else [])
    fig, axes = plt.subplots(n_rows, 1, figsize=(18, 2.5 * n_net + (2 if has_behav else 0)),
                              sharex=True,
                              gridspec_kw={'height_ratios': height_ratios, 'hspace': 0.08})
    if n_rows == 1:
        axes = [axes]

    # Load behavior for shading
    behav = None
    feeding_mask = None
    digging_mask = None

    if has_behav:
        behav = load_behavior(behav_path)
        behav_time = behav['time']
        if 'Feeding' in behav:
            feeding_mask = np.interp(eval_time, behav_time, behav['Feeding']) > 0.5
        if 'Digging' in behav:
            digging_mask = np.interp(eval_time, behav_time, behav['Digging']) > 0.5

    # Also load all-pairs traces for comparison
    all_traces_npz = np.load("data/coor1_spike_sync_kernel_traces.npz")

    row = 0
    for nt in ['LHA-LHA', 'RSP-RSP', 'LHA-RSP']:
        if nt not in traces:
            continue
        ax = axes[row]
        trace = traces[nt]

        # Plot all-pairs trace (faded) for comparison
        all_key = f's{snum}_{nt}'
        if all_key in all_traces_npz:
            all_time = all_traces_npz[f's{snum}_time']
            all_trace = all_traces_npz[all_key]
            ax.plot(all_time, all_trace, color='gray', linewidth=0.4, alpha=0.4,
                    label='All pairs')

        # Plot top-pairs trace
        ax.plot(eval_time, trace, color=NETWORK_COLORS[nt], linewidth=0.8, alpha=0.9,
                label=f'Top {TOP_PCT}%')

        # Shade feeding
        if feeding_mask is not None and feeding_mask.any():
            starts = np.where(np.diff(feeding_mask.astype(int)) == 1)[0]
            ends = np.where(np.diff(feeding_mask.astype(int)) == -1)[0]
            if feeding_mask[0]:
                starts = np.concatenate([[0], starts])
            if feeding_mask[-1]:
                ends = np.concatenate([ends, [len(feeding_mask) - 1]])
            for s, e in zip(starts, ends):
                ax.axvspan(eval_time[s], eval_time[min(e, len(eval_time)-1)],
                           alpha=0.2, color='orange', linewidth=0)

        # Shade digging
        if digging_mask is not None and digging_mask.any():
            starts = np.where(np.diff(digging_mask.astype(int)) == 1)[0]
            ends = np.where(np.diff(digging_mask.astype(int)) == -1)[0]
            if digging_mask[0]:
                starts = np.concatenate([[0], starts])
            if digging_mask[-1]:
                ends = np.concatenate([ends, [len(digging_mask) - 1]])
            for s, e in zip(starts, ends):
                ax.axvspan(eval_time[s], eval_time[min(e, len(eval_time)-1)],
                           alpha=0.2, color='purple', linewidth=0)

        n_p = info['n_pairs'][nt]
        ax.set_ylabel(f'{nt}\nSync', fontsize=9)
        ax.set_title(f'{nt} (top {TOP_PCT}%: {n_p} pairs)', fontsize=10, loc='left', pad=2)
        ax.legend(fontsize=7, loc='upper right')
        ax.tick_params(labelsize=8)
        row += 1

    # Behavior panels
    if has_behav:
        zones = get_zone_labels(behav)
        behav_time = behav['time']

        # Zone strip
        ax_zone = axes[row]
        prev_zone = zones[0]
        span_start = behav_time[0]
        for ti in range(1, len(zones)):
            if zones[ti] != prev_zone or ti == len(zones) - 1:
                c = ZONE_COLORS.get(prev_zone, '#e0e0e0')
                ax_zone.axvspan(span_start, behav_time[min(ti, len(behav_time)-1)],
                                alpha=0.7, color=c, linewidth=0)
                prev_zone = zones[ti]
                span_start = behav_time[ti] if ti < len(behav_time) else behav_time[-1]
        ax_zone.set_yticks([])
        ax_zone.set_ylabel('Zone', fontsize=9)
        unique_zones = sorted([z for z in np.unique(zones) if z != 'other'])
        patches = [Patch(color=ZONE_COLORS.get(z, '#e0e0e0'), label=z) for z in unique_zones]
        ax_zone.legend(handles=patches, loc='upper right', fontsize=6, ncol=min(7, len(patches)))
        row += 1

        # Feeding + Digging strip
        ax_act = axes[row]
        if feeding_mask is not None and feeding_mask.any():
            starts = np.where(np.diff(feeding_mask.astype(int)) == 1)[0]
            ends = np.where(np.diff(feeding_mask.astype(int)) == -1)[0]
            if feeding_mask[0]:
                starts = np.concatenate([[0], starts])
            if feeding_mask[-1]:
                ends = np.concatenate([ends, [len(feeding_mask) - 1]])
            for s, e in zip(starts, ends):
                ax_act.axvspan(eval_time[s], eval_time[min(e, len(eval_time)-1)],
                               alpha=0.7, color='orange', linewidth=0)
        if digging_mask is not None and digging_mask.any():
            starts = np.where(np.diff(digging_mask.astype(int)) == 1)[0]
            ends = np.where(np.diff(digging_mask.astype(int)) == -1)[0]
            if digging_mask[0]:
                starts = np.concatenate([[0], starts])
            if digging_mask[-1]:
                ends = np.concatenate([ends, [len(digging_mask) - 1]])
            for s, e in zip(starts, ends):
                ax_act.axvspan(eval_time[s], eval_time[min(e, len(eval_time)-1)],
                               alpha=0.7, color='purple', linewidth=0)
        ax_act.set_yticks([])
        ax_act.set_ylabel('Behav', fontsize=9)
        act_patches = [Patch(color='orange', label='Feeding'),
                       Patch(color='purple', label='Digging')]
        ax_act.legend(handles=act_patches, loc='upper right', fontsize=8, ncol=2)

    axes[-1].set_xlabel('Time (s)', fontsize=10)
    fig.suptitle(f'S{snum} ({state}/{phase}) — Top {TOP_PCT}% SPIKE-Sync pairs '
                 f'(kernel σ={SIGMA_SEC:.0f}s)',
                 fontsize=13, fontweight='bold')
    plt.savefig(f'figures/spike_sync_kernel_top10_s{snum}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved figures/spike_sync_kernel_top10_s{snum}.png")

print("\n[DONE]")
