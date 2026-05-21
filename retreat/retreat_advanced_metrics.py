"""
Retreat Advanced Neural Metrics — All 8 Sessions (M1, Coordinates 1)

Metrics beyond FR and PC1:
1. Population state velocity — speed of trajectory in PCA space
2. Population vector distance — Euclidean distance from session-mean state
3. Population sparseness — how concentrated vs distributed activity is
4. Trial-to-trial Fano factor — variability across retreats at each time bin

Per session: peri-retreat traces + pre vs post stats
Cross-session: fed vs fasted comparison, summary table
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
N_PCA = 6  # use more PCs for velocity/distance

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


def detect_retreats(zones, time_vals, behav):
    transitions = []
    for i in range(1, len(zones)):
        if zones[i] != zones[i-1]:
            if zones[i-1] in arena_zones and zones[i] in retreat_destinations:
                transitions.append({
                    'time': time_vals[i],
                    'time_idx': i,
                    'from_zone': zones[i-1],
                    'to_zone': zones[i],
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
    """Speed of neural trajectory in PCA space: ||dx/dt|| per time bin.
    scores: (n_time, n_components)
    Returns: (n_time,) velocity at each bin.
    """
    diff = np.diff(scores, axis=0)  # (n_time-1, n_components)
    speed = np.linalg.norm(diff, axis=1) / BIN_SEC  # units per second
    # Pad to same length (prepend first value)
    speed = np.concatenate([[speed[0]], speed])
    return speed


def compute_pop_distance(z_arr, scores):
    """Euclidean distance of population state from session mean in PCA space.
    scores: (n_time, n_components)
    Returns: (n_time,) distance.
    """
    mean_state = scores.mean(axis=0)
    dist = np.linalg.norm(scores - mean_state, axis=1)
    return dist


def compute_sparseness(z_arr):
    """Population sparseness at each time bin.
    Uses the Treves-Rolls sparseness: (sum(r/n))^2 / (sum(r^2/n))
    where r = firing rates shifted to be non-negative.
    z_arr: (n_units, n_time)
    Returns: (n_time,) sparseness in [0, 1]. Lower = sparser (fewer active units).
    """
    # Shift to non-negative (use raw z-scores + offset)
    r = z_arr - z_arr.min(axis=1, keepdims=True)  # shift each unit to >= 0
    n = r.shape[0]
    mean_r = r.mean(axis=0)  # (n_time,)
    mean_r2 = (r ** 2).mean(axis=0)  # (n_time,)
    # Treves-Rolls sparseness
    sparseness = np.where(mean_r2 > 1e-10,
                          (mean_r ** 2) / mean_r2,
                          1.0)
    return sparseness


def compute_fano_factor(peri_fr_all_trials):
    """Trial-to-trial Fano factor at each time bin.
    peri_fr_all_trials: (n_trials, n_units, n_time)
    Returns: (n_time,) mean Fano factor across units.
    """
    # Per unit: variance across trials / mean across trials at each time bin
    mean_across_trials = peri_fr_all_trials.mean(axis=0)  # (n_units, n_time)
    var_across_trials = peri_fr_all_trials.var(axis=0)    # (n_units, n_time)
    # Fano per unit per time
    fano = np.where(mean_across_trials > 1e-6,
                    var_across_trials / mean_across_trials,
                    np.nan)
    # Mean across units
    fano_mean = np.nanmean(fano, axis=0)  # (n_time,)
    return fano_mean


# =============================================================================
# MAIN LOOP
# =============================================================================
pre_bins = int(PRE_SEC / BIN_SEC)
post_bins = int(POST_SEC / BIN_SEC)
window_bins = pre_bins + post_bins
peri_time = np.arange(-pre_bins, post_bins) * BIN_SEC

all_session_results = {}
all_stats = []

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

    # Load behavior + detect retreats
    behav = load_behavior(behav_path)
    time_vals = behav['time']
    zones = get_zone_labels(behav)
    transitions = detect_retreats(zones, time_vals, behav)
    print(f"  Retreats: {len(transitions)}")

    if len(transitions) < 5:
        print("  SKIP — too few retreats")
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

    # PCA (more components for velocity/distance)
    n_lha_pc = min(N_PCA, len(lha_ids) - 1)
    n_rsp_pc = min(N_PCA, len(rsp_ids) - 1)
    lha_pca = PCA(n_components=n_lha_pc).fit(lha_z.T)
    lha_scores = lha_pca.transform(lha_z.T)  # (n_time, n_pc)
    rsp_pca = PCA(n_components=n_rsp_pc).fit(rsp_z.T)
    rsp_scores = rsp_pca.transform(rsp_z.T)

    lha_var = lha_pca.explained_variance_ratio_[:n_lha_pc].sum()
    rsp_var = rsp_pca.explained_variance_ratio_[:n_rsp_pc].sum()
    print(f"  LHA PCA ({n_lha_pc} PCs): {lha_var:.1%}, RSP PCA ({n_rsp_pc} PCs): {rsp_var:.1%}")

    # Compute session-wide metrics
    lha_velocity = compute_state_velocity(lha_scores)
    rsp_velocity = compute_state_velocity(rsp_scores)
    lha_distance = compute_pop_distance(lha_z, lha_scores)
    rsp_distance = compute_pop_distance(rsp_z, rsp_scores)
    lha_sparseness = compute_sparseness(lha_z)
    rsp_sparseness = compute_sparseness(rsp_z)

    # Extract peri-event windows
    valid_transitions = []
    peri = {
        'lha_velocity': [], 'rsp_velocity': [],
        'lha_distance': [], 'rsp_distance': [],
        'lha_sparseness': [], 'rsp_sparseness': [],
        'lha_peri_fr': [], 'rsp_peri_fr': [],  # for Fano factor
    }

    for t in transitions:
        center_bin = int(np.searchsorted(bin_centers, t['time']))
        start = center_bin - pre_bins
        end = center_bin + post_bins
        if start < 0 or end > n_neural_bins:
            continue
        valid_transitions.append(t)
        peri['lha_velocity'].append(lha_velocity[start:end])
        peri['rsp_velocity'].append(rsp_velocity[start:end])
        peri['lha_distance'].append(lha_distance[start:end])
        peri['rsp_distance'].append(rsp_distance[start:end])
        peri['lha_sparseness'].append(lha_sparseness[start:end])
        peri['rsp_sparseness'].append(rsp_sparseness[start:end])
        peri['lha_peri_fr'].append(lha_fr[:, start:end])
        peri['rsp_peri_fr'].append(rsp_fr[:, start:end])

    for key in peri:
        peri[key] = np.array(peri[key])

    n_valid = len(valid_transitions)
    print(f"  Valid retreats: {n_valid}")

    # Compute Fano factor
    lha_fano = compute_fano_factor(peri['lha_peri_fr'])  # (window_bins,)
    rsp_fano = compute_fano_factor(peri['rsp_peri_fr'])

    # Store
    all_session_results[snum] = {
        'state': state, 'phase': phase, 'n_valid': n_valid,
        'n_lha': len(lha_ids), 'n_rsp': len(rsp_ids),
        'peri': peri,
        'lha_fano': lha_fano, 'rsp_fano': rsp_fano,
    }

    # =========================================================================
    # PER-SESSION FIGURE: 4 metrics × 2 regions
    # =========================================================================
    metric_configs = [
        ('State Velocity', 'lha_velocity', 'rsp_velocity', 'PCA speed (a.u./s)'),
        ('Pop. Distance', 'lha_distance', 'rsp_distance', 'Euclidean dist from mean'),
        ('Sparseness', 'lha_sparseness', 'rsp_sparseness', 'Treves-Rolls sparseness'),
    ]

    fig, axes = plt.subplots(4, 2, figsize=(14, 16), sharex=True)
    region_colors = {'LHA': '#e74c3c', 'RSP': '#27ae60'}

    for row, (title, lha_key, rsp_key, ylabel) in enumerate(metric_configs):
        for col, (region, key, color) in enumerate([
            ('LHA', lha_key, '#e74c3c'), ('RSP', rsp_key, '#27ae60')
        ]):
            ax = axes[row, col]
            data = peri[key]
            mean_val = gaussian_filter1d(data.mean(axis=0), SMOOTH_SIGMA)
            sem_val = gaussian_filter1d(data.std(axis=0) / np.sqrt(n_valid), SMOOTH_SIGMA)
            ax.fill_between(peri_time, mean_val - sem_val, mean_val + sem_val,
                            alpha=0.3, color=color)
            ax.plot(peri_time, mean_val, color=color, linewidth=1.5)
            ax.axvline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
            ax.set_ylabel(ylabel, fontsize=9)
            if row == 0:
                ax.set_title(f'{region}', fontsize=12, fontweight='bold')
            ax.text(0.02, 0.95, title, transform=ax.transAxes, fontsize=9,
                    va='top', fontweight='bold')
            ax.tick_params(labelsize=8)

    # Fano factor row
    for col, (region, fano_data, color) in enumerate([
        ('LHA', lha_fano, '#e74c3c'), ('RSP', rsp_fano, '#27ae60')
    ]):
        ax = axes[3, col]
        smooth_fano = gaussian_filter1d(fano_data, SMOOTH_SIGMA)
        ax.plot(peri_time, smooth_fano, color=color, linewidth=1.5)
        ax.axvline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.set_ylabel('Fano factor', fontsize=9)
        ax.set_xlabel('Time from retreat (s)', fontsize=10)
        ax.text(0.02, 0.95, 'Fano Factor', transform=ax.transAxes, fontsize=9,
                va='top', fontweight='bold')
        ax.tick_params(labelsize=8)

    fig.suptitle(f'S{snum} ({state}/{phase}) — Advanced Peri-Retreat Metrics '
                 f'(n={n_valid} retreats, ±{PRE_SEC:.0f}/{POST_SEC:.0f}s)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f'figures/retreat_advanced_s{snum}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved figures/retreat_advanced_s{snum}.png")

    # =========================================================================
    # Pre vs Post stats for all metrics
    # =========================================================================
    pre_sl = slice(0, pre_bins)
    post_sl = slice(pre_bins, window_bins)

    session_stats = []
    metric_list = [
        ('LHA Velocity', peri['lha_velocity']),
        ('RSP Velocity', peri['rsp_velocity']),
        ('LHA Distance', peri['lha_distance']),
        ('RSP Distance', peri['rsp_distance']),
        ('LHA Sparseness', peri['lha_sparseness']),
        ('RSP Sparseness', peri['rsp_sparseness']),
    ]

    for name, data in metric_list:
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
            'diff': diff, 'pct_change': pct, 'wilcoxon_p': p_val, 'n_retreats': n_valid,
        })
    all_stats.extend(session_stats)

    # Print
    print(f"\n  Pre ({PRE_SEC:.0f}s) vs Post ({POST_SEC:.0f}s) Retreat:")
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

# Save stats
stats_df = pd.DataFrame(all_stats)
stats_df.to_csv('data/retreat_advanced_stats_all_sessions.csv', index=False)
print("Saved data/retreat_advanced_stats_all_sessions.csv")

# =========================================================================
# Fed vs Fasted overlay for each metric
# =========================================================================
metric_keys = [
    ('State Velocity', 'velocity', 'PCA speed (a.u./s)'),
    ('Pop. Distance', 'distance', 'Euclidean dist'),
    ('Sparseness', 'sparseness', 'Treves-Rolls'),
]

fig, axes = plt.subplots(4, 2, figsize=(14, 16), sharex=True)
state_colors = {'fed': '#1f77b4', 'fasted': '#d62728'}

for row, (title, key_suffix, ylabel) in enumerate(metric_keys):
    for col, region in enumerate(['LHA', 'RSP']):
        ax = axes[row, col]
        peri_key = f'{region.lower()}_{key_suffix}'

        for state_val, color in [('fed', '#1f77b4'), ('fasted', '#d62728')]:
            state_sessions = [s for s, info in all_session_results.items()
                              if info['state'] == state_val]
            if not state_sessions:
                continue
            all_traces = np.concatenate([all_session_results[s]['peri'][peri_key]
                                         for s in state_sessions])
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

# Fano factor row — compute per-state pooled Fano
for col, region in enumerate(['LHA', 'RSP']):
    ax = axes[3, col]
    fr_key = f'{region.lower()}_peri_fr'

    for state_val, color in [('fed', '#1f77b4'), ('fasted', '#d62728')]:
        state_sessions = [s for s, info in all_session_results.items()
                          if info['state'] == state_val]
        if not state_sessions:
            continue

        # For Fano, we need to compute per session then average
        fano_traces = []
        total_n = 0
        for s in state_sessions:
            info = all_session_results[s]
            fano_key = f'{region.lower()}_fano'
            fano_traces.append(info[fano_key])
            total_n += info['n_valid']

        mean_fano = gaussian_filter1d(np.mean(fano_traces, axis=0), SMOOTH_SIGMA)
        sem_fano = gaussian_filter1d(np.std(fano_traces, axis=0) / np.sqrt(len(fano_traces)), SMOOTH_SIGMA)
        ax.fill_between(peri_time, mean_fano - sem_fano, mean_fano + sem_fano,
                        alpha=0.2, color=color)
        ax.plot(peri_time, mean_fano, color=color, linewidth=1.5,
                label=f'{state_val.capitalize()} ({len(state_sessions)} sessions)')

    ax.axvline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.set_ylabel('Fano factor', fontsize=9)
    ax.set_xlabel('Time from retreat (s)', fontsize=10)
    ax.legend(fontsize=8)
    ax.text(0.02, 0.95, 'Fano Factor', transform=ax.transAxes, fontsize=9,
            va='top', fontweight='bold')

fig.suptitle('Peri-Retreat Advanced Metrics: Fed vs Fasted\n'
             f'Blue=Fed, Red=Fasted (pooled retreats, ±{PRE_SEC:.0f}/{POST_SEC:.0f}s)',
             fontsize=13, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig('figures/retreat_advanced_fed_vs_fasted.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved figures/retreat_advanced_fed_vs_fasted.png")

# =========================================================================
# Cross-session overlay (all 8 sessions per metric)
# =========================================================================
fig, axes = plt.subplots(4, 2, figsize=(16, 16), sharex=True)
state_line_colors = {'fed': '#1f77b4', 'fasted': '#d62728'}
phase_styles = {'exploration': '-', 'foraging': '--'}

for row, (title, key_suffix, ylabel) in enumerate(metric_keys):
    for col, region in enumerate(['LHA', 'RSP']):
        ax = axes[row, col]
        peri_key = f'{region.lower()}_{key_suffix}'

        for snum in sorted(all_session_results.keys()):
            info = all_session_results[snum]
            data = info['peri'][peri_key]
            mean_val = gaussian_filter1d(data.mean(axis=0), SMOOTH_SIGMA)
            color = state_line_colors[info['state']]
            ls = phase_styles[info['phase']]
            ax.plot(peri_time, mean_val, color=color, linestyle=ls, linewidth=1.0,
                    alpha=0.8, label=f"S{snum} (n={info['n_valid']})")

        ax.axvline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.set_ylabel(ylabel, fontsize=9)
        if row == 0:
            ax.set_title(f'{region}', fontsize=12, fontweight='bold')
        ax.text(0.02, 0.95, title, transform=ax.transAxes, fontsize=9,
                va='top', fontweight='bold')
        ax.legend(fontsize=6, loc='upper right')

# Fano row
for col, region in enumerate(['LHA', 'RSP']):
    ax = axes[3, col]
    for snum in sorted(all_session_results.keys()):
        info = all_session_results[snum]
        fano = gaussian_filter1d(info[f'{region.lower()}_fano'], SMOOTH_SIGMA)
        color = state_line_colors[info['state']]
        ls = phase_styles[info['phase']]
        ax.plot(peri_time, fano, color=color, linestyle=ls, linewidth=1.0,
                alpha=0.8, label=f"S{snum}")
    ax.axvline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.set_ylabel('Fano factor', fontsize=9)
    ax.set_xlabel('Time from retreat (s)', fontsize=10)
    ax.text(0.02, 0.95, 'Fano Factor', transform=ax.transAxes, fontsize=9,
            va='top', fontweight='bold')
    ax.legend(fontsize=6, loc='upper right')

from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='#1f77b4', linestyle='-', label='Fed/Exploration'),
    Line2D([0], [0], color='#1f77b4', linestyle='--', label='Fed/Foraging'),
    Line2D([0], [0], color='#d62728', linestyle='-', label='Fasted/Exploration'),
    Line2D([0], [0], color='#d62728', linestyle='--', label='Fasted/Foraging'),
]
fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=10,
           bbox_to_anchor=(0.5, -0.01))
fig.suptitle('Cross-Session Peri-Retreat Advanced Metrics\n'
             'Blue=Fed, Red=Fasted | Solid=Exploration, Dashed=Foraging',
             fontsize=13, fontweight='bold')
plt.tight_layout(rect=[0, 0.02, 1, 0.94])
plt.savefig('figures/retreat_advanced_cross_session.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved figures/retreat_advanced_cross_session.png")

# =========================================================================
# Print summary table
# =========================================================================
print("\n\nSUMMARY TABLE — Advanced Metrics Pre vs Post Retreat")
print(f"{'Sess':<5} {'State':<8} {'Phase':<6} {'N':>4} | "
      f"{'LHA Vel p':>10} {'RSP Vel p':>10} {'LHA Dist p':>10} {'RSP Dist p':>10} "
      f"{'LHA Spar p':>10} {'RSP Spar p':>10}")
print("-" * 100)

for snum in sorted(all_session_results.keys()):
    info = all_session_results[snum]
    s_stats = [s for s in all_stats if s['session'] == snum]
    row = f"S{snum:<4} {info['state']:<8} {info['phase'][:3]:<6} {info['n_valid']:>4} | "
    for metric in ['LHA Velocity', 'RSP Velocity', 'LHA Distance', 'RSP Distance',
                   'LHA Sparseness', 'RSP Sparseness']:
        st = next((s for s in s_stats if s['metric'] == metric), None)
        if st:
            p = st['wilcoxon_p']
            sig = '*' if p < 0.05 else ''
            row += f"{p:>9.4f}{sig} "
        else:
            row += f"{'N/A':>10} "
    print(row)

# Direction of change for significant results
print("\n\nSIGNIFICANT RESULTS (p < 0.05):")
sig_stats = [s for s in all_stats if s['wilcoxon_p'] < 0.05]
if sig_stats:
    for s in sig_stats:
        direction = 'increase' if s['diff'] > 0 else 'decrease'
        print(f"  S{s['session']} {s['metric']}: {s['pre_mean']:.4f} -> {s['post_mean']:.4f} "
              f"({s['pct_change']:+.1f}% {direction}, p={s['wilcoxon_p']:.4f})")
else:
    print("  None")

print("\n[DONE]")
