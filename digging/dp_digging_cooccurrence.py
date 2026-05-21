"""
Dual-Probe: Spike Co-occurrence Around Digging Onset
=====================================================
Computes pairwise cross-correlation (z-scored dot product) at 10ms and 50ms
lags in 1s sliding windows around dig onset.

Pair categories: ACA-ACA, LHA-LHA, ACA-LHA
Windows: same peri-dig structure as dp_digging_neural_signatures.py

Per-session individual bout figures + pooled + state comparison.
"""

import yaml
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon
import spikeinterface.extractors as se
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
import time as timer

warnings.filterwarnings('ignore')

with open("paths.yaml") as f:
    cfg = yaml.safe_load(f)

# ---- Constants ----
FS = 30000
LHA_DEPTH_MIN = 0
LHA_DEPTH_MAX = 345
P0_MIN_FR = 0.2
P1_MIN_FR = 0.2
P1_MIN_AMP = 43

BIN_MS = 1               # 1ms bins for cross-correlation
PRE_SEC = 30              # full window for baseline
POST_SEC = 15
WINDOW_SEC = 1.0          # sliding window size for co-occurrence
MIN_DIG_DURATION = 2.0
MIN_INTER_DIG = 10.0
LAGS_MS = [10, 50]        # cross-correlation lags

SKIP_SESSIONS = {23, 24}
FIG_XLIM = (-5, 10)

sessions_cfg = cfg["double_probe"]["coordinates_1"]["mouse01"]["sessions"]

# Zone mapping (for pot detection)
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


def load_behavior_xlsx(path):
    df = pd.read_excel(path, header=None)
    col_names = df.iloc[34].tolist()
    data = df.iloc[36:].reset_index(drop=True)
    data.columns = col_names
    time_vals = pd.to_numeric(data['Recording time'], errors='coerce').values
    zones = np.full(len(time_vals), 'O', dtype=object)
    for zname in zone_priority:
        col_match = [c for c in col_names if isinstance(c, str) and
                     c.startswith('Zone(') and zname in c]
        if col_match:
            vals = pd.to_numeric(data[col_match[0]], errors='coerce').values
            mask = vals > 0.5
            short = zone_short.get(zname, zname[:3])
            zones[mask] = short
    dig_col = 'Digging sand'
    if dig_col in col_names:
        dig_vals = pd.to_numeric(data[dig_col], errors='coerce').values
        dig_vals = np.nan_to_num(dig_vals, nan=0.0)
    else:
        dig_vals = np.zeros(len(time_vals))
    return time_vals, zones, dig_vals


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


def get_good_units_p0(sorted_path):
    ci = Path(sorted_path) / "cluster_info.tsv"
    if not ci.exists():
        return np.array([])
    df = pd.read_csv(ci, sep='\t')
    label_col = 'group' if ('group' in df.columns and df['group'].eq('good').any()) else 'KSLabel'
    if label_col not in df.columns:
        return np.array([])
    good = df[(df[label_col] == 'good') & (df['fr'] > P0_MIN_FR)]
    return good['cluster_id'].values


def get_good_units_p1_lha(sorted_path):
    ci = Path(sorted_path) / "cluster_info.tsv"
    if not ci.exists():
        return np.array([])
    df = pd.read_csv(ci, sep='\t')
    label_col = 'group' if ('group' in df.columns and df['group'].eq('good').any()) else 'KSLabel'
    if label_col not in df.columns or 'depth' not in df.columns:
        return np.array([])
    good = df[(df[label_col] == 'good') &
              (df['fr'] > P1_MIN_FR) &
              (df['amp'] > P1_MIN_AMP) &
              (df['depth'] >= LHA_DEPTH_MIN) &
              (df['depth'] <= LHA_DEPTH_MAX)]
    return good['cluster_id'].values


def compute_cooccurrence_batch(z_trains, lag_bins, win_bins, n_windows):
    """Batch-compute mean pairwise cross-correlation for all windows at once.

    Reshapes z_trains into (n_units, n_windows, win_bins) and uses batch matmul
    to compute all windows simultaneously — ~45x faster than per-window loop.
    """
    n_units = z_trains.shape[0]
    if n_units < 2:
        return np.full(n_windows, np.nan)

    n_samp = win_bins - lag_bins
    if n_samp < 10:
        return np.full(n_windows, np.nan)

    # Reshape to windows: (N, W, win_bins)
    z_w = z_trains[:, :n_windows * win_bins].reshape(n_units, n_windows, win_bins)

    shifted = z_w[:, :, lag_bins:lag_bins + n_samp]  # (N, W, S)
    base = z_w[:, :, :n_samp]                        # (N, W, S)

    # Batch matmul: (W, N, S) @ (W, S, N) -> (W, N, N)
    s = shifted.transpose(1, 0, 2)    # (W, N, S)
    b = base.transpose(1, 2, 0)       # (W, S, N)
    r_batch = np.matmul(s, b) / n_samp  # (W, N, N)

    # Upper triangle mean for each window
    mask = np.triu(np.ones((n_units, n_units), dtype=bool), k=1)
    return r_batch[:, mask].mean(axis=1)  # (W,)


def compute_cross_cooccurrence_batch(z_aca, z_lha, lag_bins, win_bins, n_windows):
    """Batch-compute mean cross-region correlation for all windows at once."""
    n_aca, n_lha = z_aca.shape[0], z_lha.shape[0]
    if n_aca < 1 or n_lha < 1:
        return np.full(n_windows, np.nan)

    n_samp = win_bins - lag_bins
    if n_samp < 10:
        return np.full(n_windows, np.nan)

    aca_w = z_aca[:, :n_windows * win_bins].reshape(n_aca, n_windows, win_bins)
    lha_w = z_lha[:, :n_windows * win_bins].reshape(n_lha, n_windows, win_bins)

    aca_s = aca_w[:, :, lag_bins:lag_bins + n_samp].transpose(1, 0, 2)  # (W, n_aca, S)
    lha_b = lha_w[:, :, :n_samp].transpose(1, 2, 0)                    # (W, S, n_lha)

    r_cross = np.matmul(aca_s, lha_b) / n_samp  # (W, n_aca, n_lha)
    return r_cross.mean(axis=(1, 2))  # (W,)


# ========================================================================
# Discover sessions
# ========================================================================
session_meta = {}
for skey, sval in sessions_cfg.items():
    snum = int(skey.split('_')[1])
    if snum in SKIP_SESSIONS:
        continue
    p0_sorted = sval.get('probe_0_aca', {}).get('sorted')
    p1_sorted = sval.get('probe_1_lha_rsp', {}).get('sorted')
    behav = sval.get('behavior')
    if not p0_sorted or not p1_sorted or not behav:
        continue
    if not Path(p0_sorted).exists() or not Path(behav).exists():
        continue
    session_meta[snum] = {
        'state': sval['state'], 'phase': sval['phase'],
        'p0_sorted': p0_sorted, 'p1_sorted': p1_sorted,
        'behavior': behav,
    }

print(f"Found {len(session_meta)} sessions with sorted data + behavior")

# ========================================================================
# MAIN ANALYSIS
# ========================================================================
n_pre_bins_sec = int(PRE_SEC)
n_post_bins_sec = int(POST_SEC)
n_total_sec = n_pre_bins_sec + n_post_bins_sec  # 45 windows of 1s each
win_bins = int(WINDOW_SEC * 1000)  # 1000 bins per 1s window
time_axis = np.arange(-n_pre_bins_sec, n_post_bins_sec) + 0.5  # window centers in sec

# Slices for statistics (in 1s windows)
pre_slice = slice(n_pre_bins_sec - 3, n_pre_bins_sec)     # [-3s, 0s]
post_slice = slice(n_pre_bins_sec + 1, n_pre_bins_sec + 4) # [+1s, +4s]
baseline_slice = slice(0, n_pre_bins_sec - 15)              # [-30s, -15s]

pair_categories = ['ACA-ACA', 'LHA-LHA', 'ACA-LHA']

# Storage: {lag: {category: [list of 45-element arrays, one per bout]}}
pooled = {lag: {cat: [] for cat in pair_categories} for lag in LAGS_MS}
pooled_meta = []

per_session_dir = Path('figures/dp_digging_cooccurrence')
per_session_dir.mkdir(exist_ok=True)

results_rows = []

print("\n" + "=" * 100)
print("DUAL-PROBE: SPIKE CO-OCCURRENCE AROUND DIGGING ONSET")
print(f"Lags: {LAGS_MS} ms, {WINDOW_SEC}s sliding windows")
print(f"Pre: [-3s,0s], Post: [+1s,+4s], Baseline: [-30s,-15s]")
print("=" * 100)

for snum in sorted(session_meta.keys()):
    t0 = timer.time()
    meta = session_meta[snum]
    state, phase = meta['state'], meta['phase']

    # ---- Load behavior ----
    print(f"\n  S{snum} ({state}/{phase}): loading...", end='', flush=True)
    time_vals, zones, dig_vals = load_behavior_xlsx(meta['behavior'])

    if np.sum(dig_vals > 0.5) == 0:
        print(" no digging scored, skipping")
        continue

    bouts = extract_dig_bouts(dig_vals, time_vals, MIN_DIG_DURATION, MIN_INTER_DIG)
    if len(bouts) == 0:
        print(" no valid dig bouts, skipping")
        continue

    print(f" {len(bouts)} bouts", end='')

    # ---- Load neural data ----
    p0_path = Path(meta['p0_sorted'])
    p1_path = Path(meta['p1_sorted'])

    aca_ids = get_good_units_p0(p0_path)
    lha_ids = get_good_units_p1_lha(p1_path)

    try:
        sorting_p0 = se.read_kilosort(p0_path)
        avail_p0 = set(sorting_p0.get_unit_ids())
        aca_ids = np.array([u for u in aca_ids if u in avail_p0])
    except Exception:
        aca_ids = np.array([])

    try:
        sorting_p1 = se.read_kilosort(p1_path)
        avail_p1 = set(sorting_p1.get_unit_ids())
        lha_ids = np.array([u for u in lha_ids if u in avail_p1])
    except Exception:
        lha_ids = np.array([])

    has_aca = len(aca_ids) >= 2
    has_lha = len(lha_ids) >= 2
    has_cross = has_aca and has_lha
    print(f", ACA={len(aca_ids)}, LHA={len(lha_ids)}", end='')

    if not has_aca and not has_lha:
        print(" — skipping (too few units)")
        continue

    # ---- Precompute spike times in seconds ----
    aca_st = {u: sorting_p0.get_unit_spike_train(u) / FS for u in aca_ids} if has_aca else {}
    lha_st = {u: sorting_p1.get_unit_spike_train(u) / FS for u in lha_ids} if has_lha else {}

    # ---- Compute session-wide mean/std for z-scoring ----
    session_dur = time_vals[-1] + 1
    n_session_bins = int(session_dur * 1000)

    if has_aca:
        aca_means = np.array([len(aca_st[u]) / session_dur * 0.001 for u in aca_ids])  # mean per 1ms bin
        aca_stds = np.sqrt(aca_means * (1 - aca_means * 0.001))  # Poisson approx
        aca_stds = np.where(aca_stds > 1e-6, aca_stds, 1e-6)

    if has_lha:
        lha_means = np.array([len(lha_st[u]) / session_dur * 0.001 for u in lha_ids])
        lha_stds = np.sqrt(lha_means * (1 - lha_means * 0.001))
        lha_stds = np.where(lha_stds > 1e-6, lha_stds, 1e-6)

    # ---- Process each dig bout ----
    session_traces = {lag: {cat: [] for cat in pair_categories} for lag in LAGS_MS}
    n_valid = 0

    for bout in bouts:
        onset = bout['start_time']
        t_start = onset - PRE_SEC
        t_end = onset + POST_SEC

        if t_start < 0 or t_end > session_dur:
            continue

        # 1ms bin edges for this peri-dig window — exactly 45000 bins + 1 edges
        n_bins_needed = n_total_sec * 1000
        edges = np.linspace(t_start, t_start + n_total_sec, n_bins_needed + 1)
        if edges[-1] > session_dur:
            continue

        # Bin and z-score ACA trains
        if has_aca:
            aca_trains = np.array([np.histogram(aca_st[u], bins=edges)[0].astype(np.float32)
                                   for u in aca_ids])
            aca_z = (aca_trains - aca_means[:, None]) / aca_stds[:, None]

        # Bin and z-score LHA trains
        if has_lha:
            lha_trains = np.array([np.histogram(lha_st[u], bins=edges)[0].astype(np.float32)
                                   for u in lha_ids])
            lha_z = (lha_trains - lha_means[:, None]) / lha_stds[:, None]

        # Compute co-occurrence for all windows at once (batch matmul)
        for lag in LAGS_MS:
            lag_bins = lag  # 1ms bins, so lag in ms = lag in bins

            if has_aca:
                aca_trace = compute_cooccurrence_batch(
                    aca_z, lag_bins, win_bins, n_total_sec)
            else:
                aca_trace = np.full(n_total_sec, np.nan)

            if has_lha:
                lha_trace = compute_cooccurrence_batch(
                    lha_z, lag_bins, win_bins, n_total_sec)
            else:
                lha_trace = np.full(n_total_sec, np.nan)

            if has_cross:
                cross_trace = compute_cross_cooccurrence_batch(
                    aca_z, lha_z, lag_bins, win_bins, n_total_sec)
            else:
                cross_trace = np.full(n_total_sec, np.nan)

            session_traces[lag]['ACA-ACA'].append(aca_trace)
            session_traces[lag]['LHA-LHA'].append(lha_trace)
            session_traces[lag]['ACA-LHA'].append(cross_trace)

            pooled[lag]['ACA-ACA'].append(aca_trace)
            pooled[lag]['LHA-LHA'].append(lha_trace)
            pooled[lag]['ACA-LHA'].append(cross_trace)

        pooled_meta.append({
            'session': snum, 'state': state, 'phase': phase,
            'duration': bout['duration'],
            'has_aca': has_aca, 'has_lha': has_lha,
        })
        n_valid += 1

    elapsed = timer.time() - t0
    print(f", {n_valid} valid [{elapsed:.1f}s]")

    if n_valid == 0:
        continue

    # ---- Per-session stats: pre vs post ----
    print(f"    Session stats (pre [-3s,0s] vs post [+1s,+4s]):")
    for lag in LAGS_MS:
        for cat in pair_categories:
            traces = np.array(session_traces[lag][cat])
            if len(traces) < 2 or np.all(np.isnan(traces)):
                continue

            pre_vals = np.nanmean(traces[:, pre_slice], axis=1)
            post_vals = np.nanmean(traces[:, post_slice], axis=1)

            valid = ~(np.isnan(pre_vals) | np.isnan(post_vals))
            if np.sum(valid) < 2:
                continue

            pre_v = pre_vals[valid]
            post_v = post_vals[valid]

            try:
                _, wil_p = wilcoxon(pre_v - post_v)
            except Exception:
                wil_p = 1.0

            delta = np.mean(post_v) - np.mean(pre_v)
            sig = '*' if wil_p < 0.05 else ''
            print(f"      {cat} {lag}ms: pre={np.mean(pre_v):.6f}, "
                  f"post={np.mean(post_v):.6f}, delta={delta:+.6f}, "
                  f"Wilcox p={wil_p:.4f}{sig} (n={len(pre_v)})")

            results_rows.append({
                'session': snum, 'state': state, 'phase': phase,
                'lag_ms': lag, 'pair_category': cat,
                'n_bouts': len(pre_v),
                'pre_mean': np.mean(pre_v), 'post_mean': np.mean(post_v),
                'delta': delta, 'wilcoxon_p': wil_p,
                'group': f'S{snum}',
            })

    # ---- Per-session figure: individual bout traces ----
    bout_cmap = plt.cm.tab10

    for lag in LAGS_MS:
        fig, axes = plt.subplots(1, 3, figsize=(24, 7))

        for ax_idx, cat in enumerate(pair_categories):
            ax = axes[ax_idx]
            traces = session_traces[lag][cat]

            for bi, tr in enumerate(traces):
                if np.all(np.isnan(tr)):
                    continue
                color = bout_cmap(bi % 10)
                dur = bouts[bi]['duration'] if bi < len(bouts) else 0
                ax.plot(time_axis, tr, color=color, linewidth=1.2, alpha=0.7,
                        label=f'B{bi+1} {dur:.0f}s')

            # Mean trace
            all_tr = np.array(traces)
            if len(all_tr) > 0 and not np.all(np.isnan(all_tr)):
                mean_tr = np.nanmean(all_tr, axis=0)
                ax.plot(time_axis, mean_tr, color='black', linewidth=3,
                        label='Mean', zorder=10)

            ax.axvline(x=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
            ax.axvspan(-3, 0, color='#2196F3', alpha=0.08)
            ax.axvspan(1, 4, color='#4CAF50', alpha=0.08)
            ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
            ax.set_xlim(FIG_XLIM)
            ax.set_title(f'{cat} ({len(traces)} bouts)', fontsize=14, fontweight='bold')
            ax.set_xlabel('Time from dig onset (s)', fontsize=12)
            ax.set_ylabel(f'Cross-corr (r, {lag}ms lag)', fontsize=12)
            ax.tick_params(labelsize=11)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            if len(traces) <= 15:
                ax.legend(fontsize=8, loc='upper right', ncol=2)

        fig.suptitle(f'S{snum} — {state.capitalize()} / {phase.capitalize()} — '
                     f'Spike Co-occurrence ({lag}ms lag)',
                     fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        fname = per_session_dir / f'S{snum}_{state}_{phase}_cooccurrence_{lag}ms.png'
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    Saved {fname}")

# ========================================================================
# POOLED ANALYSIS
# ========================================================================
print("\n" + "=" * 100)
print("POOLED RESULTS")
print("=" * 100)

total_events = len(pooled_meta)
print(f"\nTotal dig events: {total_events}")

state_colors = {'fed': '#4e79a7', 'fasted': '#e15759', 'fed-HFD': '#f28e2b'}
state_labels = {'fed': 'Fed', 'fasted': 'Fasted', 'fed-HFD': 'HFD'}

for lag in LAGS_MS:
    print(f"\n  --- {lag}ms lag ---")

    # ---- All events ----
    for cat in pair_categories:
        traces = np.array(pooled[lag][cat])
        if len(traces) < 3 or np.all(np.isnan(traces)):
            continue

        pre_vals = np.nanmean(traces[:, pre_slice], axis=1)
        post_vals = np.nanmean(traces[:, post_slice], axis=1)
        valid = ~(np.isnan(pre_vals) | np.isnan(post_vals))
        pre_v, post_v = pre_vals[valid], post_vals[valid]
        if len(pre_v) < 3:
            continue

        try:
            _, wil_p = wilcoxon(pre_v - post_v)
        except Exception:
            wil_p = 1.0

        delta = np.mean(post_v) - np.mean(pre_v)
        sig = '*' if wil_p < 0.05 else ''
        print(f"    ALL {cat}: pre={np.mean(pre_v):.6f}, post={np.mean(post_v):.6f}, "
              f"delta={delta:+.6f}, Wilcox p={wil_p:.4f}{sig} (n={len(pre_v)})")

        results_rows.append({
            'lag_ms': lag, 'pair_category': cat,
            'n_bouts': len(pre_v),
            'pre_mean': np.mean(pre_v), 'post_mean': np.mean(post_v),
            'delta': delta, 'wilcoxon_p': wil_p,
            'group': 'all',
        })

    # ---- Per state ----
    for state_filter in ['fed', 'fasted', 'fed-HFD']:
        s_mask = np.array([m['state'] == state_filter for m in pooled_meta])
        n_state = np.sum(s_mask)
        if n_state < 3:
            continue

        print(f"\n    {state_filter.upper()} ({n_state} events):")
        for cat in pair_categories:
            traces = np.array(pooled[lag][cat])
            if 'ACA' in cat:
                c_mask = np.array([m['state'] == state_filter and m['has_aca']
                                   for m in pooled_meta])
            elif cat == 'LHA-LHA':
                c_mask = np.array([m['state'] == state_filter and m['has_lha']
                                   for m in pooled_meta])
            else:
                c_mask = s_mask

            t_f = traces[c_mask[:len(traces)]]
            if len(t_f) < 3 or np.all(np.isnan(t_f)):
                continue

            pre_vals = np.nanmean(t_f[:, pre_slice], axis=1)
            post_vals = np.nanmean(t_f[:, post_slice], axis=1)
            valid = ~(np.isnan(pre_vals) | np.isnan(post_vals))
            pre_v, post_v = pre_vals[valid], post_vals[valid]
            if len(pre_v) < 3:
                continue

            try:
                _, wil_p = wilcoxon(pre_v - post_v)
            except Exception:
                wil_p = 1.0

            delta = np.mean(post_v) - np.mean(pre_v)
            sig = '*' if wil_p < 0.05 else ''
            print(f"      {cat}: pre={np.mean(pre_v):.6f}, post={np.mean(post_v):.6f}, "
                  f"delta={delta:+.6f}, Wilcox p={wil_p:.4f}{sig} (n={len(pre_v)})")

            results_rows.append({
                'lag_ms': lag, 'pair_category': cat,
                'n_bouts': len(pre_v),
                'pre_mean': np.mean(pre_v), 'post_mean': np.mean(post_v),
                'delta': delta, 'wilcoxon_p': wil_p,
                'group': state_filter,
            })

    # ---- State comparison: pre-dig co-occurrence levels ----
    print(f"\n    STATE COMPARISON (pre-dig co-occurrence, {lag}ms):")
    for cat in pair_categories:
        traces = np.array(pooled[lag][cat])
        state_pre = {}
        for sf in ['fed', 'fasted', 'fed-HFD']:
            if 'ACA' in cat:
                c_mask = np.array([m['state'] == sf and m['has_aca'] for m in pooled_meta])
            elif cat == 'LHA-LHA':
                c_mask = np.array([m['state'] == sf and m['has_lha'] for m in pooled_meta])
            else:
                c_mask = np.array([m['state'] == sf for m in pooled_meta])

            t_f = traces[c_mask[:len(traces)]]
            if len(t_f) < 2:
                continue
            pre_vals = np.nanmean(t_f[:, pre_slice], axis=1)
            valid = pre_vals[~np.isnan(pre_vals)]
            if len(valid) >= 2:
                state_pre[sf] = valid

        if len(state_pre) < 2:
            continue

        # Pairwise MWU between states
        states = list(state_pre.keys())
        for i in range(len(states)):
            for j in range(i + 1, len(states)):
                s1, s2 = states[i], states[j]
                try:
                    _, p = mannwhitneyu(state_pre[s1], state_pre[s2], alternative='two-sided')
                except Exception:
                    p = 1.0
                sig = '*' if p < 0.05 else ''
                print(f"      {cat} {s1}({np.mean(state_pre[s1]):.6f}) vs "
                      f"{s2}({np.mean(state_pre[s2]):.6f}): MWU p={p:.4f}{sig}")

                results_rows.append({
                    'lag_ms': lag, 'pair_category': cat,
                    'pre_mean': np.mean(state_pre[s1]),
                    'post_mean': np.mean(state_pre[s2]),
                    'delta': np.mean(state_pre[s2]) - np.mean(state_pre[s1]),
                    'wilcoxon_p': p,
                    'group': f'{s1}_vs_{s2}',
                })


# ========================================================================
# POOLED FIGURES
# ========================================================================
for lag in LAGS_MS:
    # ---- Figure: pooled mean traces ----
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))

    for ax_idx, cat in enumerate(pair_categories):
        ax = axes[ax_idx]
        traces = np.array(pooled[lag][cat])
        if len(traces) < 3:
            ax.set_title(f'{cat} (no data)')
            continue

        # All events
        valid_mask = ~np.all(np.isnan(traces), axis=1)
        t_valid = traces[valid_mask]
        if len(t_valid) > 0:
            mean_tr = np.nanmean(t_valid, axis=0)
            sem_tr = np.nanstd(t_valid, axis=0) / np.sqrt(np.sum(~np.isnan(t_valid), axis=0))
            ax.plot(time_axis, mean_tr, color='black', linewidth=2.5,
                    label=f'All (n={len(t_valid)})')
            ax.fill_between(time_axis, mean_tr - sem_tr, mean_tr + sem_tr,
                            color='black', alpha=0.12)

        ax.axvline(x=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.axvspan(-3, 0, color='#2196F3', alpha=0.08)
        ax.axvspan(1, 4, color='#4CAF50', alpha=0.08)
        ax.set_xlim(FIG_XLIM)
        ax.set_title(cat, fontsize=14, fontweight='bold')
        ax.set_xlabel('Time from dig onset (s)', fontsize=12)
        ax.set_ylabel(f'Cross-corr (r, {lag}ms lag)', fontsize=12)
        ax.tick_params(labelsize=11)
        ax.legend(fontsize=11)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.suptitle(f'Spike Co-occurrence Around Dig Onset — Pooled ({lag}ms lag)',
                 fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    fname = f'figures/dp_digging_cooccurrence_pooled_{lag}ms.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {fname}")

    # ---- Figure: by state ----
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))

    for ax_idx, cat in enumerate(pair_categories):
        ax = axes[ax_idx]

        for state_filter, color in state_colors.items():
            if 'ACA' in cat:
                s_mask = np.array([m['state'] == state_filter and m['has_aca']
                                   for m in pooled_meta])
            elif cat == 'LHA-LHA':
                s_mask = np.array([m['state'] == state_filter and m['has_lha']
                                   for m in pooled_meta])
            else:
                s_mask = np.array([m['state'] == state_filter for m in pooled_meta])

            traces = np.array(pooled[lag][cat])
            t_f = traces[s_mask[:len(traces)]]
            valid_mask = ~np.all(np.isnan(t_f), axis=1) if len(t_f) > 0 else np.array([])
            if len(valid_mask) == 0 or np.sum(valid_mask) < 2:
                continue
            t_f = t_f[valid_mask]

            mean_tr = np.nanmean(t_f, axis=0)
            sem_tr = np.nanstd(t_f, axis=0) / np.sqrt(len(t_f))

            ax.plot(time_axis, mean_tr, color=color, linewidth=2,
                    label=f'{state_labels[state_filter]} (n={len(t_f)})')
            ax.fill_between(time_axis, mean_tr - sem_tr, mean_tr + sem_tr,
                            color=color, alpha=0.12)

        ax.axvline(x=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.axvspan(-3, 0, color='#2196F3', alpha=0.08)
        ax.axvspan(1, 4, color='#4CAF50', alpha=0.08)
        ax.set_xlim(FIG_XLIM)
        ax.set_title(cat, fontsize=14, fontweight='bold')
        ax.set_xlabel('Time from dig onset (s)', fontsize=12)
        ax.set_ylabel(f'Cross-corr (r, {lag}ms lag)', fontsize=12)
        ax.tick_params(labelsize=11)
        ax.legend(fontsize=11)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.suptitle(f'Spike Co-occurrence by State ({lag}ms lag)',
                 fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    fname = f'figures/dp_digging_cooccurrence_by_state_{lag}ms.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {fname}")

# Save CSV
df_results = pd.DataFrame(results_rows)
df_results.to_csv('data/dp_digging_cooccurrence_stats.csv', index=False)
print(f"\nSaved data/dp_digging_cooccurrence_stats.csv ({len(df_results)} rows)")
print("\nDone.")
