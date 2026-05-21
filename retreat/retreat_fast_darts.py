"""
Retreat Neural Signatures — FAST DARTS (top 25% velocity)
All 8 Sessions (M1, Coordinates 1)

Same analyses as retreat_advanced_metrics.py + retreat_neural_signatures_all_sessions.py
but filtered to only retreat transitions with velocity >= 75th percentile of retreat velocities.

Metrics:
- Population FR, PC1, PC2 (LHA + RSP)
- State velocity, population distance, sparseness, Fano factor
"""

import yaml
from pathlib import Path
import numpy as np
import pandas as pd
import spikeinterface.extractors as se
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter1d
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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

PRE_SEC = 2.0
POST_SEC = 5.0
BIN_SEC = 0.1
SMOOTH_SIGMA = 2
N_PCA = 6
VEL_WINDOW = 5  # bins (±0.5s around transition) to find max velocity

sessions_cfg = cfg["single_probe"]["coordinates_1"]["mouse01"]["sessions"]
session_meta = {
    1: ('fed', 'exploration'), 2: ('fed', 'foraging'),
    3: ('fed', 'exploration'), 4: ('fed', 'foraging'),
    5: ('fasted', 'exploration'), 6: ('fasted', 'foraging'),
    7: ('fasted', 'exploration'), 8: ('fasted', 'foraging'),
}

arena_zones = {'Pot-1', 'Pot-2', 'Pot-3', 'Pot-4',
               'Pot-1 zone', 'Pot-2 zone', 'Pot-3 zone', 'Pot-4 zone',
               'Arna center', 'Foraging arena', 'Transition zone',
               'Right corner', 'Left corner'}
retreat_destinations = {'Home', 'Ladder'}

priority_order = [
    'Right corner', 'Left corner', 'Arna center', 'Foraging arena',
    'Home', 'Ladder', 'Transition zone',
    'Pot-1 zone', 'Pot-2 zone', 'Pot-3 zone', 'Pot-4 zone',
    'Pot-1', 'Pot-2', 'Pot-3', 'Pot-4',
]


# =============================================================================
# HELPERS
# =============================================================================
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
    for var_name in priority_order:
        if var_name in behav:
            mask = behav[var_name] > 0.5
            zones[mask] = var_name
    return zones


def detect_retreats_with_velocity(zones, time_vals, behav):
    vel = behav.get('Velocity')
    transitions = []
    for i in range(1, len(zones)):
        if zones[i] != zones[i-1]:
            if zones[i-1] in arena_zones and zones[i] in retreat_destinations:
                # Max velocity in window around transition
                window = slice(max(0, i - VEL_WINDOW), min(len(vel), i + VEL_WINDOW))
                max_vel = np.nanmax(vel[window]) if vel is not None else np.nan
                transitions.append({
                    'time': time_vals[i],
                    'time_idx': i,
                    'from_zone': zones[i-1],
                    'to_zone': zones[i],
                    'velocity': max_vel,
                })
    return transitions


def get_good_units(sorted_path):
    ci = Path(sorted_path) / "cluster_info.tsv"
    df = pd.read_csv(ci, sep='\t')
    label_col = 'group' if ('group' in df.columns and df['group'].eq('good').any()) else 'KSLabel'
    good = df[(df[label_col] == 'good') & (df['fr'] > MIN_FR) & (df['amp'] > MIN_AMP)]
    lha = good[good['depth'] < LHA_DEPTH_MAX]['cluster_id'].values
    rsp = good[good['depth'] >= LHA_DEPTH_MAX]['cluster_id'].values
    return lha, rsp


def zscore_array(arr):
    mu = arr.mean(axis=1, keepdims=True)
    sd = arr.std(axis=1, keepdims=True)
    sd[sd < 1e-6] = 1
    return (arr - mu) / sd


def compute_state_velocity(scores):
    diff = np.diff(scores, axis=0)
    speed = np.linalg.norm(diff, axis=1) / BIN_SEC
    return np.concatenate([[speed[0]], speed])


def compute_pop_distance(scores):
    mean_state = scores.mean(axis=0)
    return np.linalg.norm(scores - mean_state, axis=1)


def compute_sparseness(z_arr):
    r = z_arr - z_arr.min(axis=1, keepdims=True)
    mean_r = r.mean(axis=0)
    mean_r2 = (r ** 2).mean(axis=0)
    return np.where(mean_r2 > 1e-10, (mean_r ** 2) / mean_r2, 1.0)


def compute_fano_factor(peri_fr_all_trials):
    mean_across = peri_fr_all_trials.mean(axis=0)
    var_across = peri_fr_all_trials.var(axis=0)
    fano = np.where(mean_across > 1e-6, var_across / mean_across, np.nan)
    return np.nanmean(fano, axis=0)


# =============================================================================
# MAIN LOOP
# =============================================================================
pre_bins = int(PRE_SEC / BIN_SEC)
post_bins = int(POST_SEC / BIN_SEC)
window_bins = pre_bins + post_bins
peri_time = np.arange(-pre_bins, post_bins) * BIN_SEC

all_session_results = {}
all_stats = []
all_transitions_df = []

for snum in range(1, 9):
    t0 = timer.time()
    state, phase = session_meta[snum]
    sc = sessions_cfg[f"session_{snum}"]
    sorted_path = Path(sc['sorted'])
    behav_path = sc.get('behavior')

    print(f"\n{'='*70}")
    print(f"SESSION {snum} ({state}/{phase})")
    print(f"{'='*70}")

    if not behav_path or not Path(behav_path).exists():
        print("  SKIP — no behavior data")
        continue

    # Load behavior + detect retreats with velocity
    behav = load_behavior(behav_path)
    time_vals = behav['time']
    zones = get_zone_labels(behav)
    all_retreats = detect_retreats_with_velocity(zones, time_vals, behav)
    print(f"  All retreats: {len(all_retreats)}")

    if len(all_retreats) < 5:
        print("  SKIP — too few retreats")
        continue

    # 75th percentile velocity filter
    all_vels = np.array([t['velocity'] for t in all_retreats])
    vel_p75 = np.nanpercentile(all_vels, 75)
    fast_retreats = [t for t in all_retreats if t['velocity'] >= vel_p75]
    slow_retreats = [t for t in all_retreats if t['velocity'] < vel_p75]

    print(f"  Velocity 75th percentile: {vel_p75:.1f}")
    print(f"  Fast darts (>= p75): {len(fast_retreats)}")
    print(f"  Slow returns (< p75): {len(slow_retreats)}")
    print(f"  Fast vel range: {min(t['velocity'] for t in fast_retreats):.1f} - "
          f"{max(t['velocity'] for t in fast_retreats):.1f}")

    if len(fast_retreats) < 5:
        print("  SKIP — too few fast retreats")
        continue

    # Load neural data
    lha_ids, rsp_ids = get_good_units(sorted_path)
    sorting = se.read_kilosort(sorted_path)
    avail = set(sorting.get_unit_ids())
    lha_ids = np.array([u for u in lha_ids if u in avail])
    rsp_ids = np.array([u for u in rsp_ids if u in avail])
    print(f"  LHA: {len(lha_ids)} units, RSP: {len(rsp_ids)} units")

    if len(lha_ids) < 2 or len(rsp_ids) < 2:
        print("  SKIP — insufficient units")
        continue

    # Bin spike trains
    rec_duration = time_vals[-1] + BIN_SEC
    bin_edges = np.arange(0, rec_duration + BIN_SEC, BIN_SEC)
    n_neural_bins = len(bin_edges) - 1
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    lha_fr = np.array([np.histogram(sorting.get_unit_spike_train(u) / FS, bins=bin_edges)[0] / BIN_SEC
                       for u in lha_ids])
    rsp_fr = np.array([np.histogram(sorting.get_unit_spike_train(u) / FS, bins=bin_edges)[0] / BIN_SEC
                       for u in rsp_ids])

    lha_z = zscore_array(lha_fr)
    rsp_z = zscore_array(rsp_fr)

    # PCA
    n_lha_pc = min(N_PCA, len(lha_ids) - 1)
    n_rsp_pc = min(N_PCA, len(rsp_ids) - 1)
    lha_scores = PCA(n_components=n_lha_pc).fit_transform(lha_z.T)
    rsp_scores = PCA(n_components=n_rsp_pc).fit_transform(rsp_z.T)

    # Session-wide metrics
    lha_velocity = compute_state_velocity(lha_scores)
    rsp_velocity = compute_state_velocity(rsp_scores)
    lha_distance = compute_pop_distance(lha_scores)
    rsp_distance = compute_pop_distance(rsp_scores)
    lha_sparseness = compute_sparseness(lha_z)
    rsp_sparseness = compute_sparseness(rsp_z)

    # =========================================================================
    # Extract peri-event windows for FAST and SLOW separately
    # =========================================================================
    def extract_peri(retreat_list):
        peri = {
            'lha_pop': [], 'rsp_pop': [],
            'lha_pc': [], 'rsp_pc': [],
            'lha_velocity': [], 'rsp_velocity': [],
            'lha_distance': [], 'rsp_distance': [],
            'lha_sparseness': [], 'rsp_sparseness': [],
            'lha_peri_raw': [], 'rsp_peri_raw': [],
        }
        valid = []
        for t in retreat_list:
            center_bin = int(np.searchsorted(bin_centers, t['time']))
            start = center_bin - pre_bins
            end = center_bin + post_bins
            if start < 0 or end > n_neural_bins:
                continue
            valid.append(t)
            peri['lha_pop'].append(lha_z[:, start:end].mean(axis=0))
            peri['rsp_pop'].append(rsp_z[:, start:end].mean(axis=0))
            peri['lha_pc'].append(lha_scores[start:end, :min(3, n_lha_pc)])
            peri['rsp_pc'].append(rsp_scores[start:end, :min(3, n_rsp_pc)])
            peri['lha_velocity'].append(lha_velocity[start:end])
            peri['rsp_velocity'].append(rsp_velocity[start:end])
            peri['lha_distance'].append(lha_distance[start:end])
            peri['rsp_distance'].append(rsp_distance[start:end])
            peri['lha_sparseness'].append(lha_sparseness[start:end])
            peri['rsp_sparseness'].append(rsp_sparseness[start:end])
            peri['lha_peri_raw'].append(lha_fr[:, start:end])
            peri['rsp_peri_raw'].append(rsp_fr[:, start:end])
        for key in peri:
            peri[key] = np.array(peri[key]) if peri[key] else np.array([])
        return valid, peri

    valid_fast, peri_fast = extract_peri(fast_retreats)
    valid_slow, peri_slow = extract_peri(slow_retreats)
    n_fast = len(valid_fast)
    n_slow = len(valid_slow)
    print(f"  Valid fast: {n_fast}, Valid slow: {n_slow}")

    # Fano factor
    lha_fano_fast = compute_fano_factor(peri_fast['lha_peri_raw']) if n_fast > 2 else np.full(window_bins, np.nan)
    rsp_fano_fast = compute_fano_factor(peri_fast['rsp_peri_raw']) if n_fast > 2 else np.full(window_bins, np.nan)
    lha_fano_slow = compute_fano_factor(peri_slow['lha_peri_raw']) if n_slow > 2 else np.full(window_bins, np.nan)
    rsp_fano_slow = compute_fano_factor(peri_slow['rsp_peri_raw']) if n_slow > 2 else np.full(window_bins, np.nan)

    # Store for cross-session
    for t in valid_fast:
        t['session'] = snum
        t['state'] = state
        t['phase'] = phase
        t['speed_group'] = 'fast'
    for t in valid_slow:
        t['session'] = snum
        t['state'] = state
        t['phase'] = phase
        t['speed_group'] = 'slow'
    all_transitions_df.extend(valid_fast)
    all_transitions_df.extend(valid_slow)

    all_session_results[snum] = {
        'state': state, 'phase': phase,
        'n_fast': n_fast, 'n_slow': n_slow,
        'n_lha': len(lha_ids), 'n_rsp': len(rsp_ids),
        'peri_fast': peri_fast, 'peri_slow': peri_slow,
        'lha_fano_fast': lha_fano_fast, 'rsp_fano_fast': rsp_fano_fast,
        'lha_fano_slow': lha_fano_slow, 'rsp_fano_slow': rsp_fano_slow,
        'vel_p75': vel_p75,
    }

    # =========================================================================
    # FIGURE: Fast darts — all metrics (8 rows: FR/PC1/PC2 + velocity/distance/sparseness/fano)
    # =========================================================================
    fig, axes = plt.subplots(7, 2, figsize=(14, 24), sharex=True)

    metric_plots = [
        ('Pop FR', 'lha_pop', 'rsp_pop', 'z-scored FR'),
        ('PC1', lambda p: p['lha_pc'][:, :, 0] if p['lha_pc'].ndim == 3 else None,
                lambda p: p['rsp_pc'][:, :, 0] if p['rsp_pc'].ndim == 3 else None, 'PC1 score'),
        ('State Velocity', 'lha_velocity', 'rsp_velocity', 'PCA speed (a.u./s)'),
        ('Pop Distance', 'lha_distance', 'rsp_distance', 'Euclidean dist'),
        ('Sparseness', 'lha_sparseness', 'rsp_sparseness', 'Treves-Rolls'),
    ]

    for row, (title, lha_key, rsp_key, ylabel) in enumerate(metric_plots):
        for col, (region, key, color_fast, color_slow) in enumerate([
            ('LHA', lha_key, '#c0392b', '#e6b0aa'),
            ('RSP', rsp_key, '#1e8449', '#a9dfbf'),
        ]):
            ax = axes[row, col]

            # Get data
            if callable(key):
                fast_data = key(peri_fast)
                slow_data = key(peri_slow)
            else:
                fast_data = peri_fast[key] if peri_fast[key].size > 0 else None
                slow_data = peri_slow[key] if peri_slow[key].size > 0 else None

            if fast_data is not None and len(fast_data) > 0:
                mean_f = gaussian_filter1d(fast_data.mean(axis=0), SMOOTH_SIGMA)
                sem_f = gaussian_filter1d(fast_data.std(axis=0) / np.sqrt(len(fast_data)), SMOOTH_SIGMA)
                ax.fill_between(peri_time, mean_f - sem_f, mean_f + sem_f,
                                alpha=0.3, color=color_fast)
                ax.plot(peri_time, mean_f, color=color_fast, linewidth=1.5,
                        label=f'Fast (n={n_fast})')

            if slow_data is not None and len(slow_data) > 0:
                mean_s = gaussian_filter1d(slow_data.mean(axis=0), SMOOTH_SIGMA)
                sem_s = gaussian_filter1d(slow_data.std(axis=0) / np.sqrt(len(slow_data)), SMOOTH_SIGMA)
                ax.fill_between(peri_time, mean_s - sem_s, mean_s + sem_s,
                                alpha=0.2, color=color_slow)
                ax.plot(peri_time, mean_s, color=color_slow, linewidth=1.2,
                        linestyle='--', label=f'Slow (n={n_slow})')

            ax.axvline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
            ax.set_ylabel(ylabel, fontsize=9)
            if row == 0:
                ax.set_title(f'{region}', fontsize=12, fontweight='bold')
            ax.text(0.02, 0.95, title, transform=ax.transAxes, fontsize=9,
                    va='top', fontweight='bold')
            ax.legend(fontsize=7)
            ax.tick_params(labelsize=8)

    # Fano factor row
    for col, (region, fano_f, fano_s, color_f, color_s) in enumerate([
        ('LHA', lha_fano_fast, lha_fano_slow, '#c0392b', '#e6b0aa'),
        ('RSP', rsp_fano_fast, rsp_fano_slow, '#1e8449', '#a9dfbf'),
    ]):
        ax = axes[5, col]
        if np.any(np.isfinite(fano_f)):
            ax.plot(peri_time, gaussian_filter1d(fano_f, SMOOTH_SIGMA), color=color_f,
                    linewidth=1.5, label=f'Fast (n={n_fast})')
        if np.any(np.isfinite(fano_s)):
            ax.plot(peri_time, gaussian_filter1d(fano_s, SMOOTH_SIGMA), color=color_s,
                    linewidth=1.2, linestyle='--', label=f'Slow (n={n_slow})')
        ax.axvline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.set_ylabel('Fano factor', fontsize=9)
        ax.text(0.02, 0.95, 'Fano Factor', transform=ax.transAxes, fontsize=9,
                va='top', fontweight='bold')
        ax.legend(fontsize=7)

    # Per-unit heatmap row (fast darts only)
    for col, (region, ids, peri_fr_key) in enumerate([
        ('LHA', lha_ids, 'lha_peri_raw'),
        ('RSP', rsp_ids, 'rsp_peri_raw'),
    ]):
        ax = axes[6, col]
        if peri_fast[peri_fr_key].size > 0:
            # z-score the raw FR for heatmap
            peri_raw = peri_fast[peri_fr_key]  # (n_fast, n_units, window)
            mean_per_unit = zscore_array(peri_raw.mean(axis=0))  # won't work — already per-unit
            # Use the z-scored version instead
            if peri_fr_key == 'lha_peri_raw':
                z_peri = np.array([lha_z[:, int(np.searchsorted(bin_centers, t['time'])) - pre_bins:
                                         int(np.searchsorted(bin_centers, t['time'])) + post_bins]
                                   for t in valid_fast if
                                   int(np.searchsorted(bin_centers, t['time'])) - pre_bins >= 0 and
                                   int(np.searchsorted(bin_centers, t['time'])) + post_bins <= n_neural_bins])
            else:
                z_peri = np.array([rsp_z[:, int(np.searchsorted(bin_centers, t['time'])) - pre_bins:
                                         int(np.searchsorted(bin_centers, t['time'])) + post_bins]
                                   for t in valid_fast if
                                   int(np.searchsorted(bin_centers, t['time'])) - pre_bins >= 0 and
                                   int(np.searchsorted(bin_centers, t['time'])) + post_bins <= n_neural_bins])
            mean_unit = z_peri.mean(axis=0)  # (n_units, window)
            peak_t = np.argmax(mean_unit, axis=1)
            sort_idx = np.argsort(peak_t)
            sorted_data = mean_unit[sort_idx].copy()
            for i in range(sorted_data.shape[0]):
                sorted_data[i] = gaussian_filter1d(sorted_data[i], SMOOTH_SIGMA)
            im = ax.imshow(sorted_data, aspect='auto', cmap='RdBu_r',
                           extent=[peri_time[0], peri_time[-1], sorted_data.shape[0]-0.5, -0.5],
                           vmin=-0.5, vmax=0.5, interpolation='nearest')
            ax.axvline(0, color='k', linestyle='--', linewidth=1, alpha=0.7)
            plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label='z-FR')
        ax.set_xlabel('Time from retreat (s)', fontsize=10)
        ax.set_ylabel('Unit', fontsize=9)
        ax.text(0.02, 0.95, f'{region} Unit Heatmap (fast only)', transform=ax.transAxes,
                fontsize=9, va='top', fontweight='bold')

    fig.suptitle(f'S{snum} ({state}/{phase}) — Fast Darts vs Slow Returns\n'
                 f'75th pct vel threshold: {vel_p75:.0f} | Fast: {n_fast}, Slow: {n_slow}',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f'figures/retreat_fast_darts_s{snum}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved figures/retreat_fast_darts_s{snum}.png")

    # =========================================================================
    # Pre vs Post stats — FAST DARTS ONLY
    # =========================================================================
    pre_sl = slice(0, pre_bins)
    post_sl = slice(pre_bins, window_bins)

    session_stats = []
    stat_metrics = [
        ('LHA Pop FR', peri_fast['lha_pop']),
        ('RSP Pop FR', peri_fast['rsp_pop']),
        ('LHA PC1', peri_fast['lha_pc'][:, :, 0] if peri_fast['lha_pc'].ndim == 3 else None),
        ('RSP PC1', peri_fast['rsp_pc'][:, :, 0] if peri_fast['rsp_pc'].ndim == 3 else None),
        ('LHA Velocity', peri_fast['lha_velocity']),
        ('RSP Velocity', peri_fast['rsp_velocity']),
        ('LHA Distance', peri_fast['lha_distance']),
        ('RSP Distance', peri_fast['rsp_distance']),
        ('LHA Sparseness', peri_fast['lha_sparseness']),
        ('RSP Sparseness', peri_fast['rsp_sparseness']),
    ]

    for name, data in stat_metrics:
        if data is None or data.size == 0 or len(data) < 5:
            continue
        pre_means = data[:, pre_sl].mean(axis=1)
        post_means = data[:, post_sl].mean(axis=1)
        try:
            stat, p_val = wilcoxon(pre_means, post_means)
        except ValueError:
            p_val = 1.0
        diff = post_means.mean() - pre_means.mean()
        pct = (diff / abs(pre_means.mean()) * 100) if abs(pre_means.mean()) > 1e-10 else 0
        session_stats.append({
            'session': snum, 'state': state, 'phase': phase,
            'metric': name, 'pre_mean': pre_means.mean(), 'post_mean': post_means.mean(),
            'diff': diff, 'pct_change': pct, 'wilcoxon_p': p_val, 'n_fast': n_fast,
        })
    all_stats.extend(session_stats)

    print(f"\n  FAST DARTS Pre ({PRE_SEC:.0f}s) vs Post ({POST_SEC:.0f}s):")
    print(f"  {'Metric':<18} {'Pre':>10} {'Post':>10} {'%chg':>8} {'p':>10} {'Sig':>4}")
    for r in session_stats:
        sig = '*' if r['wilcoxon_p'] < 0.05 else 'ns'
        print(f"  {r['metric']:<18} {r['pre_mean']:>10.4f} {r['post_mean']:>10.4f} "
              f"{r['pct_change']:>7.1f}% {r['wilcoxon_p']:>10.4f} {sig:>4}")

    elapsed = timer.time() - t0
    print(f"\n  Session {snum} done in {elapsed:.1f}s")


# =============================================================================
# CROSS-SESSION FIGURES
# =============================================================================
print(f"\n{'='*70}")
print("CROSS-SESSION SUMMARY")
print(f"{'='*70}")

# Save
trans_df = pd.DataFrame(all_transitions_df)
trans_df.to_csv('data/retreat_fast_slow_transitions.csv', index=False)
print(f"Saved data/retreat_fast_slow_transitions.csv ({len(trans_df)} events)")

stats_df = pd.DataFrame(all_stats)
stats_df.to_csv('data/retreat_fast_darts_stats.csv', index=False)
print("Saved data/retreat_fast_darts_stats.csv")

# =========================================================================
# Cross-session: Fed vs Fasted, FAST DARTS ONLY — all metrics
# =========================================================================
metric_configs = [
    ('Pop FR', 'pop', 'z-scored FR'),
    ('PC1', 'pc1', 'PC1 score'),
    ('State Velocity', 'velocity', 'PCA speed'),
    ('Pop Distance', 'distance', 'Euclidean dist'),
    ('Sparseness', 'sparseness', 'Treves-Rolls'),
]

fig, axes = plt.subplots(6, 2, figsize=(14, 22), sharex=True)
state_colors = {'fed': '#1f77b4', 'fasted': '#d62728'}

for row, (title, key_suffix, ylabel) in enumerate(metric_configs):
    for col, region in enumerate(['LHA', 'RSP']):
        ax = axes[row, col]

        for state_val, color in [('fed', '#1f77b4'), ('fasted', '#d62728')]:
            state_sessions = [s for s, info in all_session_results.items()
                              if info['state'] == state_val]
            if not state_sessions:
                continue

            traces = []
            for s in state_sessions:
                pf = all_session_results[s]['peri_fast']
                if key_suffix == 'pop':
                    d = pf[f'{region.lower()}_pop']
                elif key_suffix == 'pc1':
                    pc = pf[f'{region.lower()}_pc']
                    d = pc[:, :, 0] if pc.ndim == 3 and pc.size > 0 else np.array([])
                else:
                    d = pf[f'{region.lower()}_{key_suffix}']
                if d.size > 0:
                    traces.append(d)

            if not traces:
                continue
            all_traces = np.concatenate(traces)
            n_total = len(all_traces)
            mean_val = gaussian_filter1d(all_traces.mean(axis=0), SMOOTH_SIGMA)
            sem_val = gaussian_filter1d(all_traces.std(axis=0) / np.sqrt(n_total), SMOOTH_SIGMA)
            ax.fill_between(peri_time, mean_val - sem_val, mean_val + sem_val,
                            alpha=0.2, color=color)
            ax.plot(peri_time, mean_val, color=color, linewidth=1.5,
                    label=f'{state_val.capitalize()} (n={n_total})')

        ax.axvline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.legend(fontsize=8)
        if row == 0:
            ax.set_title(f'{region}', fontsize=12, fontweight='bold')
        ax.text(0.02, 0.95, title, transform=ax.transAxes, fontsize=9,
                va='top', fontweight='bold')

# Fano factor row
for col, region in enumerate(['LHA', 'RSP']):
    ax = axes[5, col]
    for state_val, color in [('fed', '#1f77b4'), ('fasted', '#d62728')]:
        state_sessions = [s for s, info in all_session_results.items()
                          if info['state'] == state_val]
        if not state_sessions:
            continue
        fano_traces = [all_session_results[s][f'{region.lower()}_fano_fast']
                       for s in state_sessions
                       if np.any(np.isfinite(all_session_results[s][f'{region.lower()}_fano_fast']))]
        if not fano_traces:
            continue
        mean_fano = gaussian_filter1d(np.mean(fano_traces, axis=0), SMOOTH_SIGMA)
        sem_fano = gaussian_filter1d(np.std(fano_traces, axis=0) / np.sqrt(len(fano_traces)), SMOOTH_SIGMA)
        ax.fill_between(peri_time, mean_fano - sem_fano, mean_fano + sem_fano,
                        alpha=0.2, color=color)
        ax.plot(peri_time, mean_fano, color=color, linewidth=1.5,
                label=f'{state_val.capitalize()} ({len(fano_traces)} sess)')
    ax.axvline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.set_ylabel('Fano factor', fontsize=9)
    ax.set_xlabel('Time from retreat (s)', fontsize=10)
    ax.legend(fontsize=8)
    ax.text(0.02, 0.95, 'Fano Factor', transform=ax.transAxes, fontsize=9,
            va='top', fontweight='bold')

fig.suptitle('Fast Darts (>= 75th pct velocity): Fed vs Fasted\n'
             f'Blue=Fed, Red=Fasted (±{PRE_SEC:.0f}/{POST_SEC:.0f}s)',
             fontsize=13, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('figures/retreat_fast_darts_fed_vs_fasted.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved figures/retreat_fast_darts_fed_vs_fasted.png")

# =========================================================================
# Summary table
# =========================================================================
print("\n\nSUMMARY — FAST DARTS Pre vs Post")
print(f"{'Sess':<5} {'State':<8} {'N':>4} {'v75':>6} | "
      f"{'LHA FR':>9} {'RSP FR':>9} {'LHA PC1':>9} {'RSP PC1':>9} "
      f"{'LHA Vel':>9} {'RSP Vel':>9} {'LHA Dist':>9} {'RSP Dist':>9} "
      f"{'LHA Spr':>9} {'RSP Spr':>9}")
print("-" * 120)

for snum in sorted(all_session_results.keys()):
    info = all_session_results[snum]
    s_stats = [s for s in all_stats if s['session'] == snum]
    row = f"S{snum:<4} {info['state']:<8} {info['n_fast']:>4} {info['vel_p75']:>5.0f} | "
    for metric in ['LHA Pop FR', 'RSP Pop FR', 'LHA PC1', 'RSP PC1',
                   'LHA Velocity', 'RSP Velocity', 'LHA Distance', 'RSP Distance',
                   'LHA Sparseness', 'RSP Sparseness']:
        st = next((s for s in s_stats if s['metric'] == metric), None)
        if st:
            p = st['wilcoxon_p']
            sig = '*' if p < 0.05 else ''
            row += f"{p:>8.3f}{sig} "
        else:
            row += f"{'N/A':>9} "
    print(row)

print("\n\nSIGNIFICANT RESULTS (p < 0.05, fast darts only):")
sig_stats = [s for s in all_stats if s['wilcoxon_p'] < 0.05]
if sig_stats:
    for s in sig_stats:
        direction = 'UP' if s['diff'] > 0 else 'DOWN'
        print(f"  S{s['session']} ({s['state']}/{s['phase']}) {s['metric']}: "
              f"{s['pre_mean']:.4f} -> {s['post_mean']:.4f} "
              f"({s['pct_change']:+.1f}% {direction}, p={s['wilcoxon_p']:.4f})")
else:
    print("  None")

print("\n[DONE]")
