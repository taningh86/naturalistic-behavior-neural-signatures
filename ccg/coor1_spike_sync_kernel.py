"""
Kernel-smoothed continuous SPIKE-synchronization traces with behavior overlay.
M1 Coordinates-1, all 8 sessions.

For each pair, the SPIKE-sync profile gives a coincidence value (0 or 1) at each
spike time. To create a continuous trace:
  - Numerator: Gaussian-smoothed density of coincident spikes
  - Denominator: Gaussian-smoothed density of all spikes
  - Ratio = instantaneous fraction of coincident spikes

No discrete binning — the fine evaluation grid (50ms) is just for computation,
with sigma >> grid resolution making the result continuous.
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

EVAL_DT = 0.05        # 50ms evaluation grid (computational, not analytical bins)
SIGMA_SEC = 2.0        # Gaussian kernel sigma in seconds
SIGMA_BINS = int(SIGMA_SEC / EVAL_DT)  # sigma in grid units
EPS = 1e-10            # avoid division by zero

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
    print(f"    Units: {len(lha)} LHA + {len(rsp)} RSP")
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
# MAIN: Compute kernel-smoothed sync traces
# =============================================================================
all_session_data = {}

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
        print("  SKIP")
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

    # --- Evaluation grid ---
    eval_time = np.arange(t_min, t_max, EVAL_DT)
    n_eval = len(eval_time)

    # Accumulators: coincident spike counts and total spike counts on eval grid
    coinc_hist = {nt: np.zeros(n_eval) for nt in ['LHA-LHA', 'RSP-RSP', 'LHA-RSP']}
    total_hist = {nt: np.zeros(n_eval) for nt in ['LHA-LHA', 'RSP-RSP', 'LHA-RSP']}
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

    # --- Process pairs ---
    t0 = timer.time()
    for pi, (uid_a, uid_b, nt) in enumerate(pair_list):
        profile = pyspike.spike_sync_profile(py_trains[uid_a], py_trains[uid_b])

        x_arr = np.array(profile.x)
        y_arr = np.array(profile.y)
        mp_arr = np.array(profile.mp)

        if len(x_arr) == 0:
            continue

        # Place spikes into eval grid (fine histogram, NOT analysis bins)
        bin_idx = np.floor((x_arr - t_min) / EVAL_DT).astype(int)
        valid = (bin_idx >= 0) & (bin_idx < n_eval)
        bi = bin_idx[valid]

        # Coincident spikes
        np.add.at(coinc_hist[nt], bi, y_arr[valid])
        # All spikes
        np.add.at(total_hist[nt], bi, mp_arr[valid])

        net_n_pairs[nt] += 1

        if (pi + 1) % 2000 == 0:
            print(f"    {pi+1}/{len(pair_list)} pairs...", flush=True)

    elapsed = timer.time() - t0
    print(f"    {len(pair_list)} pairs in {elapsed:.1f}s")

    # --- Gaussian kernel smoothing ---
    traces = {}
    for nt in ['LHA-LHA', 'RSP-RSP', 'LHA-RSP']:
        if net_n_pairs[nt] == 0:
            continue

        smooth_coinc = gaussian_filter1d(coinc_hist[nt], sigma=SIGMA_BINS)
        smooth_total = gaussian_filter1d(total_hist[nt], sigma=SIGMA_BINS)

        # Synchrony = fraction of coincident spikes
        sync_trace = smooth_coinc / (smooth_total + EPS)

        # Mask regions with negligible spike density
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
# SAVE TRACES
# =============================================================================
trace_data = {}
for snum, info in all_session_data.items():
    trace_data[f's{snum}_time'] = info['eval_time']
    for nt, tr in info['traces'].items():
        trace_data[f's{snum}_{nt}'] = tr
np.savez("data/coor1_spike_sync_kernel_traces.npz", **trace_data)
print(f"\nSaved kernel-smoothed traces to data/coor1_spike_sync_kernel_traces.npz")


# =============================================================================
# FIGURES: Per-session continuous traces with behavior overlay
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

    # Count panels: networks + behavior strip + feeding/digging strip
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

    # --- Sync traces with feeding/digging shading ---
    behav = None
    feeding_mask = None
    digging_mask = None
    behav_time = None

    if has_behav:
        behav = load_behavior(behav_path)
        behav_time = behav['time']

        # Get feeding and digging aligned to eval_time
        if 'Feeding' in behav:
            feed_interp = np.interp(eval_time, behav_time, behav['Feeding'])
            feeding_mask = feed_interp > 0.5
        if 'Digging' in behav:
            dig_interp = np.interp(eval_time, behav_time, behav['Digging'])
            digging_mask = dig_interp > 0.5

    row = 0
    for nt in ['LHA-LHA', 'RSP-RSP', 'LHA-RSP']:
        if nt not in traces:
            continue
        ax = axes[row]
        trace = traces[nt]

        # Plot sync trace
        ax.plot(eval_time, trace, color=NETWORK_COLORS[nt], linewidth=0.6, alpha=0.9)

        # Shade feeding episodes
        if feeding_mask is not None and feeding_mask.any():
            starts = np.where(np.diff(feeding_mask.astype(int)) == 1)[0]
            ends = np.where(np.diff(feeding_mask.astype(int)) == -1)[0]
            if feeding_mask[0]:
                starts = np.concatenate([[0], starts])
            if feeding_mask[-1]:
                ends = np.concatenate([ends, [len(feeding_mask) - 1]])
            for s, e in zip(starts, ends):
                ax.axvspan(eval_time[s], eval_time[min(e, len(eval_time)-1)],
                           alpha=0.25, color='orange', linewidth=0)

        # Shade digging episodes
        if digging_mask is not None and digging_mask.any():
            starts = np.where(np.diff(digging_mask.astype(int)) == 1)[0]
            ends = np.where(np.diff(digging_mask.astype(int)) == -1)[0]
            if digging_mask[0]:
                starts = np.concatenate([[0], starts])
            if digging_mask[-1]:
                ends = np.concatenate([ends, [len(digging_mask) - 1]])
            for s, e in zip(starts, ends):
                ax.axvspan(eval_time[s], eval_time[min(e, len(eval_time)-1)],
                           alpha=0.25, color='purple', linewidth=0)

        ax.set_ylabel(f'{nt}\nSync', fontsize=9)
        n_p = info['n_pairs'][nt]
        ax.set_title(f'{nt} ({n_p} pairs)', fontsize=10, loc='left', pad=2)
        ax.tick_params(labelsize=8)
        row += 1

    # --- Behavior panels ---
    if has_behav:
        zones = get_zone_labels(behav)

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
    fig.suptitle(f'S{snum} ({state}/{phase}) — SPIKE-Synchronization '
                 f'(kernel σ={SIGMA_SEC:.0f}s)',
                 fontsize=13, fontweight='bold')
    plt.savefig(f'figures/spike_sync_kernel_s{snum}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved figures/spike_sync_kernel_s{snum}.png")

print("\n[DONE]")
