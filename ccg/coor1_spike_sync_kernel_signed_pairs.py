"""
Kernel-smoothed SPIKE-synchronization split by CCG sign.
Reads significant pairs from CCG output, separates into positive and negative
cross-correlation groups, computes kernel-smoothed sync traces for each.

Depends on: data/coor1_significant_pairs.csv (from coor1_ccg_and_significant_pairs.py)
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
import sys

warnings.filterwarnings('ignore')

# Log output
log_path = Path("data/coor1_spike_sync_signed_run.log")
log_file = open(log_path, 'w')

class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()

sys.stdout = Tee(sys.__stdout__, log_file)
sys.stderr = Tee(sys.__stderr__, log_file)

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


def compute_kernel_trace(pair_set, py_trains, t_min, eval_time, n_eval):
    """Compute kernel-smoothed sync trace for a set of pairs."""
    coinc_hist = np.zeros(n_eval)
    total_hist = np.zeros(n_eval)
    n_computed = 0

    for uid_a, uid_b in pair_set:
        if uid_a not in py_trains or uid_b not in py_trains:
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

        np.add.at(coinc_hist, bi, y_arr[valid])
        np.add.at(total_hist, bi, mp_arr[valid])
        n_computed += 1

    if n_computed == 0:
        return None, 0

    smooth_coinc = gaussian_filter1d(coinc_hist, sigma=SIGMA_BINS)
    smooth_total = gaussian_filter1d(total_hist, sigma=SIGMA_BINS)
    sync_trace = smooth_coinc / (smooth_total + EPS)
    sync_trace[smooth_total < EPS * 100] = np.nan
    return sync_trace, n_computed


# =============================================================================
# LOAD SIGNIFICANT PAIRS FROM CCG
# =============================================================================
sig_pairs_path = Path("data/coor1_significant_pairs.csv")
if not sig_pairs_path.exists():
    print("ERROR: data/coor1_significant_pairs.csv not found.")
    print("The CCG analysis must complete first.")
    sys.exit(1)

sig_df = pd.read_csv(sig_pairs_path)
print(f"Loaded {len(sig_df)} significant pairs from CCG analysis")

# Exclude unit 1 (false unit — spill-over artifact)
n_before = len(sig_df)
sig_df = sig_df[(sig_df['unit_a'] != 1) & (sig_df['unit_b'] != 1)]
print(f"  Excluded unit 1: {n_before} -> {len(sig_df)} pairs")

# Split by sign
pos_df = sig_df[sig_df['peak_corr'] > 0]
neg_df = sig_df[sig_df['peak_corr'] < 0]
print(f"  Positive correlation: {len(pos_df)} pairs")
print(f"  Negative correlation: {len(neg_df)} pairs")

print("\nPer session/network breakdown:")
for snum in range(1, 9):
    state, phase = session_meta[snum]
    s_pos = pos_df[pos_df['session'] == snum]
    s_neg = neg_df[neg_df['session'] == snum]
    print(f"\n  S{snum} ({state}/{phase}):")
    for nt in ['LHA-LHA', 'RSP-RSP', 'LHA-RSP']:
        np_ = len(s_pos[s_pos['network'] == nt])
        nn_ = len(s_neg[s_neg['network'] == nt])
        print(f"    {nt}: {np_} positive, {nn_} negative")

# Build pair sets per session/network/sign
pair_sets = {}  # (snum, nt, sign) -> set of (unit_a, unit_b)
for _, r in sig_df.iterrows():
    snum = int(r['session'])
    nt = r['network']
    sign = 'pos' if r['peak_corr'] > 0 else 'neg'
    key = (snum, nt, sign)
    if key not in pair_sets:
        pair_sets[key] = set()
    pair_sets[key].add((int(r['unit_a']), int(r['unit_b'])))


# =============================================================================
# COMPUTE KERNEL-SMOOTHED TRACES
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

    traces = {}
    n_pairs_info = {}

    t0 = timer.time()
    for nt in ['LHA-LHA', 'RSP-RSP', 'LHA-RSP']:
        for sign in ['pos', 'neg']:
            key = (snum, nt, sign)
            ps = pair_sets.get(key, set())
            if len(ps) == 0:
                continue

            trace, n_computed = compute_kernel_trace(ps, py_trains, t_min, eval_time, n_eval)
            if trace is not None:
                trace_key = f"{nt}_{sign}"
                traces[trace_key] = trace
                n_pairs_info[trace_key] = n_computed
                valid_vals = trace[np.isfinite(trace)]
                print(f"    {nt} {sign}: {n_computed} pairs, "
                      f"mean={np.nanmean(valid_vals):.4f}, "
                      f"range=[{np.nanmin(valid_vals):.4f}, {np.nanmax(valid_vals):.4f}]")

    elapsed = timer.time() - t0
    print(f"    Done in {elapsed:.1f}s")

    all_session_data[snum] = {
        'eval_time': eval_time, 'traces': traces,
        'state': state, 'phase': phase, 'n_pairs': n_pairs_info,
    }


# =============================================================================
# SAVE
# =============================================================================
trace_data = {}
for snum, info in all_session_data.items():
    trace_data[f's{snum}_time'] = info['eval_time']
    for key, tr in info['traces'].items():
        trace_data[f's{snum}_{key}'] = tr
np.savez("data/coor1_spike_sync_kernel_signed_traces.npz", **trace_data)
print(f"\nSaved to data/coor1_spike_sync_kernel_signed_traces.npz")


# =============================================================================
# FIGURES
# =============================================================================
print("\n" + "=" * 70)
print("GENERATING FIGURES")
print("=" * 70)

POS_COLOR_MAP = {'LHA-LHA': '#e74c3c', 'RSP-RSP': '#27ae60', 'LHA-RSP': '#2980b9'}
NEG_COLOR_MAP = {'LHA-LHA': '#f5b7b1', 'RSP-RSP': '#a9dfbf', 'LHA-RSP': '#aed6f1'}

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

    # Determine which networks have data
    networks_present = set()
    for key in traces:
        nt = key.rsplit('_', 1)[0]
        networks_present.add(nt)
    networks_present = sorted(networks_present)

    n_net = len(networks_present)
    n_rows = n_net + (2 if has_behav else 0)
    if n_rows == 0:
        continue

    height_ratios = [3] * n_net + ([1, 1] if has_behav else [])
    fig, axes = plt.subplots(n_rows, 1, figsize=(18, 2.5 * n_net + (2 if has_behav else 0)),
                              sharex=True,
                              gridspec_kw={'height_ratios': height_ratios, 'hspace': 0.08})
    if n_rows == 1:
        axes = [axes]

    # Load behavior
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

    row = 0
    for nt in networks_present:
        ax = axes[row]

        pos_key = f"{nt}_pos"
        neg_key = f"{nt}_neg"

        # Plot positive pairs
        if pos_key in traces:
            n_p = info['n_pairs'].get(pos_key, 0)
            ax.plot(eval_time, traces[pos_key], color=POS_COLOR_MAP[nt],
                    linewidth=0.8, alpha=0.9, label=f'Positive ({n_p} pairs)')

        # Plot negative pairs
        if neg_key in traces:
            n_n = info['n_pairs'].get(neg_key, 0)
            ax.plot(eval_time, traces[neg_key], color=NEG_COLOR_MAP[nt],
                    linewidth=0.8, alpha=0.9, linestyle='--',
                    label=f'Negative ({n_n} pairs)')

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

        ax.set_ylabel(f'{nt}\nSync', fontsize=9)
        ax.set_title(f'{nt} — Positive vs Negative CCG pairs', fontsize=10, loc='left', pad=2)
        ax.legend(fontsize=7, loc='upper right')
        ax.tick_params(labelsize=8)
        row += 1

    # Behavior panels
    if has_behav:
        zones = get_zone_labels(behav)
        behav_time = behav['time']

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
    fig.suptitle(f'S{snum} ({state}/{phase}) — Signed CCG Pair SPIKE-Sync '
                 f'(kernel σ={SIGMA_SEC:.0f}s)',
                 fontsize=13, fontweight='bold')
    plt.savefig(f'figures/spike_sync_kernel_signed_s{snum}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved figures/spike_sync_kernel_signed_s{snum}.png")

print("\n[DONE]")
