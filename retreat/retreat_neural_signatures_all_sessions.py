"""
Retreat Neural Signatures — All 8 Sessions (M1, Coordinates 1)

Detects time-points where mouse retreats from arena zones to Home/Ladder,
then examines peri-retreat neural activity in LHA and RSP.

Per session:
1. Detect retreat transitions (arena -> Home/Ladder)
2. Peri-retreat firing rate (population + per-unit heatmaps) for LHA and RSP
3. Peri-retreat PCA (PC1-3 trajectories)
4. Compare retreats by source zone (transition zone vs pots vs corners)
5. Pre vs Post retreat stats

Cross-session:
6. Summary comparison across sessions (fed vs fasted, exploration vs foraging)
"""

import yaml
from pathlib import Path
import numpy as np
import pandas as pd
import spikeinterface.extractors as se
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter1d
from scipy.stats import wilcoxon, mannwhitneyu
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

PRE_SEC = 2.0
POST_SEC = 5.0
BIN_SEC = 0.1
SMOOTH_SIGMA = 2

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

cat_labels = {'transition': 'Transition Zone', 'corner': 'Corners',
              'pot_area': 'Pot Areas', 'arena_center': 'Arena Center'}
cat_colors = {'transition': '#3498db', 'corner': '#e67e22',
              'pot_area': '#9b59b6', 'arena_center': '#2ecc71'}


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
                t = {
                    'time': time_vals[i],
                    'time_idx': i,
                    'from_zone': zones[i-1],
                    'to_zone': zones[i],
                    'velocity': behav['Velocity'][i] if 'Velocity' in behav else np.nan,
                }
                fz = t['from_zone']
                if 'Pot' in fz:
                    t['source_category'] = 'pot_area'
                elif fz == 'Transition zone':
                    t['source_category'] = 'transition'
                elif fz in ('Right corner', 'Left corner'):
                    t['source_category'] = 'corner'
                elif fz in ('Arna center', 'Foraging arena'):
                    t['source_category'] = 'arena_center'
                else:
                    t['source_category'] = 'other'
                transitions.append(t)
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


# =============================================================================
# MAIN LOOP
# =============================================================================
pre_bins = int(PRE_SEC / BIN_SEC)
post_bins = int(POST_SEC / BIN_SEC)
window_bins = pre_bins + post_bins
peri_time = np.arange(-pre_bins, post_bins) * BIN_SEC

all_session_results = {}
all_transitions_df = []
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

    # Load behavior
    behav = load_behavior(behav_path)
    time_vals = behav['time']
    zones = get_zone_labels(behav)

    # Detect retreats
    transitions = detect_retreats(zones, time_vals, behav)
    print(f"  Retreats: {len(transitions)}")

    if len(transitions) < 5:
        print("  SKIP — too few retreats")
        continue

    cat_counts = {}
    for t in transitions:
        cat_counts[t['source_category']] = cat_counts.get(t['source_category'], 0) + 1
    for cat, cnt in sorted(cat_counts.items(), key=lambda x: -x[1]):
        print(f"    {cat}: {cnt}")

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
    lha_pca = PCA(n_components=3).fit(lha_z.T)
    lha_scores = lha_pca.transform(lha_z.T)
    rsp_pca = PCA(n_components=3).fit(rsp_z.T)
    rsp_scores = rsp_pca.transform(rsp_z.T)

    print(f"  LHA PCA: {lha_pca.explained_variance_ratio_[:3].sum():.1%}, "
          f"RSP PCA: {rsp_pca.explained_variance_ratio_[:3].sum():.1%}")

    # Extract peri-event windows
    valid_transitions = []
    lha_peri_pop = []
    rsp_peri_pop = []
    lha_peri_pc = []
    rsp_peri_pc = []
    lha_peri_fr = []
    rsp_peri_fr = []

    for t in transitions:
        center_bin = int(np.searchsorted(bin_centers, t['time']))
        start = center_bin - pre_bins
        end = center_bin + post_bins
        if start < 0 or end > n_neural_bins:
            continue
        valid_transitions.append(t)
        lha_peri_pop.append(lha_z[:, start:end].mean(axis=0))
        rsp_peri_pop.append(rsp_z[:, start:end].mean(axis=0))
        lha_peri_pc.append(lha_scores[start:end, :3])
        rsp_peri_pc.append(rsp_scores[start:end, :3])
        lha_peri_fr.append(lha_z[:, start:end])
        rsp_peri_fr.append(rsp_z[:, start:end])

    lha_peri_pop = np.array(lha_peri_pop)
    rsp_peri_pop = np.array(rsp_peri_pop)
    lha_peri_pc = np.array(lha_peri_pc)
    rsp_peri_pc = np.array(rsp_peri_pc)
    lha_peri_fr = np.array(lha_peri_fr)
    rsp_peri_fr = np.array(rsp_peri_fr)
    n_valid = len(valid_transitions)
    print(f"  Valid retreats: {n_valid}/{len(transitions)}")

    # Add session info to transitions
    for t in valid_transitions:
        t['session'] = snum
        t['state'] = state
        t['phase'] = phase
    all_transitions_df.extend(valid_transitions)

    # Store for cross-session
    all_session_results[snum] = {
        'state': state, 'phase': phase, 'n_valid': n_valid,
        'n_lha': len(lha_ids), 'n_rsp': len(rsp_ids),
        'lha_peri_pop': lha_peri_pop, 'rsp_peri_pop': rsp_peri_pop,
        'lha_peri_pc': lha_peri_pc, 'rsp_peri_pc': rsp_peri_pc,
        'lha_peri_fr': lha_peri_fr, 'rsp_peri_fr': rsp_peri_fr,
        'lha_ids': lha_ids, 'rsp_ids': rsp_ids,
        'valid_transitions': valid_transitions,
    }

    # =========================================================================
    # FIGURE 1: Population peri-retreat FR + PC1
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    for col, (region, pop_data, pc_data, color) in enumerate([
        ('LHA', lha_peri_pop, lha_peri_pc, '#e74c3c'),
        ('RSP', rsp_peri_pop, rsp_peri_pc, '#27ae60'),
    ]):
        ax = axes[0, col]
        mean_fr = gaussian_filter1d(pop_data.mean(axis=0), SMOOTH_SIGMA)
        sem_fr = gaussian_filter1d(pop_data.std(axis=0) / np.sqrt(n_valid), SMOOTH_SIGMA)
        ax.fill_between(peri_time, mean_fr - sem_fr, mean_fr + sem_fr, alpha=0.3, color=color)
        ax.plot(peri_time, mean_fr, color=color, linewidth=1.5)
        ax.axvline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.set_ylabel('Population z-scored FR', fontsize=10)
        ax.set_title(f'{region} — Mean Population FR (n={n_valid})', fontsize=11)

        ax = axes[1, col]
        pc1 = pc_data[:, :, 0]
        mean_pc1 = gaussian_filter1d(pc1.mean(axis=0), SMOOTH_SIGMA)
        sem_pc1 = gaussian_filter1d(pc1.std(axis=0) / np.sqrt(n_valid), SMOOTH_SIGMA)
        ax.fill_between(peri_time, mean_pc1 - sem_pc1, mean_pc1 + sem_pc1, alpha=0.3, color=color)
        ax.plot(peri_time, mean_pc1, color=color, linewidth=1.5)
        ax.axvline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.set_ylabel('PC1 score', fontsize=10)
        ax.set_xlabel('Time from retreat (s)', fontsize=10)
        ax.set_title(f'{region} — PC1 (n={n_valid})', fontsize=11)

    fig.suptitle(f'S{snum} ({state}/{phase}) — Peri-Retreat Neural Activity '
                 f'(±{PRE_SEC:.0f}s, {BIN_SEC*1000:.0f}ms bins)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(f'figures/retreat_population_s{snum}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved figures/retreat_population_s{snum}.png")

    # =========================================================================
    # FIGURE 2: Per-unit heatmaps
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(16, 10))
    for col, (region, ids, peri_data) in enumerate([
        ('LHA', lha_ids, lha_peri_fr),
        ('RSP', rsp_ids, rsp_peri_fr),
    ]):
        ax = axes[col]
        mean_per_unit = peri_data.mean(axis=0)
        peak_times = np.argmax(mean_per_unit, axis=1)
        sort_idx = np.argsort(peak_times)
        sorted_data = mean_per_unit[sort_idx].copy()
        for i in range(sorted_data.shape[0]):
            sorted_data[i] = gaussian_filter1d(sorted_data[i], SMOOTH_SIGMA)
        im = ax.imshow(sorted_data, aspect='auto', cmap='RdBu_r',
                       extent=[peri_time[0], peri_time[-1], sorted_data.shape[0]-0.5, -0.5],
                       vmin=-0.5, vmax=0.5, interpolation='nearest')
        ax.axvline(0, color='k', linestyle='--', linewidth=1, alpha=0.7)
        ax.set_xlabel('Time from retreat (s)', fontsize=10)
        ax.set_ylabel('Unit (sorted by peak time)', fontsize=10)
        ax.set_title(f'{region} — {len(ids)} units, mean across {n_valid} retreats', fontsize=11)
        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label='z-scored FR')

    fig.suptitle(f'S{snum} ({state}/{phase}) — Per-Unit Peri-Retreat Heatmap',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(f'figures/retreat_unit_heatmap_s{snum}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved figures/retreat_unit_heatmap_s{snum}.png")

    # =========================================================================
    # FIGURE 3: By source category
    # =========================================================================
    cat_indices = {}
    for i, t in enumerate(valid_transitions):
        cat = t['source_category']
        if cat not in cat_indices:
            cat_indices[cat] = []
        cat_indices[cat].append(i)

    # Only plot if at least 2 categories with n>=3
    plottable_cats = [c for c in ['transition', 'corner', 'pot_area', 'arena_center']
                      if c in cat_indices and len(cat_indices[c]) >= 3]

    if len(plottable_cats) >= 2:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
        for col, (region, pop_data, pc_data) in enumerate([
            ('LHA', lha_peri_pop, lha_peri_pc),
            ('RSP', rsp_peri_pop, rsp_peri_pc),
        ]):
            ax = axes[0, col]
            for cat in plottable_cats:
                idx = cat_indices[cat]
                mean_val = gaussian_filter1d(pop_data[idx].mean(axis=0), SMOOTH_SIGMA)
                sem_val = gaussian_filter1d(pop_data[idx].std(axis=0) / np.sqrt(len(idx)), SMOOTH_SIGMA)
                ax.fill_between(peri_time, mean_val - sem_val, mean_val + sem_val,
                                alpha=0.2, color=cat_colors[cat])
                ax.plot(peri_time, mean_val, color=cat_colors[cat], linewidth=1.5,
                        label=f'{cat_labels[cat]} (n={len(idx)})')
            ax.axvline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
            ax.set_ylabel('Population z-scored FR', fontsize=10)
            ax.set_title(f'{region} — Population FR by Source', fontsize=11)
            ax.legend(fontsize=8)

            ax = axes[1, col]
            for cat in plottable_cats:
                idx = cat_indices[cat]
                cat_pc1 = pc_data[idx, :, 0]
                mean_val = gaussian_filter1d(cat_pc1.mean(axis=0), SMOOTH_SIGMA)
                sem_val = gaussian_filter1d(cat_pc1.std(axis=0) / np.sqrt(len(idx)), SMOOTH_SIGMA)
                ax.fill_between(peri_time, mean_val - sem_val, mean_val + sem_val,
                                alpha=0.2, color=cat_colors[cat])
                ax.plot(peri_time, mean_val, color=cat_colors[cat], linewidth=1.5,
                        label=f'{cat_labels[cat]} (n={len(idx)})')
            ax.axvline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
            ax.set_ylabel('PC1 score', fontsize=10)
            ax.set_xlabel('Time from retreat (s)', fontsize=10)
            ax.set_title(f'{region} — PC1 by Source', fontsize=11)
            ax.legend(fontsize=8)

        fig.suptitle(f'S{snum} ({state}/{phase}) — Peri-Retreat by Source Zone',
                     fontsize=13, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(f'figures/retreat_by_source_s{snum}.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved figures/retreat_by_source_s{snum}.png")
    else:
        print(f"  SKIP source category figure — fewer than 2 categories with n>=3")

    # =========================================================================
    # Pre vs Post stats
    # =========================================================================
    pre_window = slice(0, pre_bins)
    post_window = slice(pre_bins, window_bins)

    metrics = [
        ('LHA Pop FR', lha_peri_pop),
        ('RSP Pop FR', rsp_peri_pop),
        ('LHA PC1', lha_peri_pc[:, :, 0]),
        ('RSP PC1', rsp_peri_pc[:, :, 0]),
        ('LHA PC2', lha_peri_pc[:, :, 1]),
        ('RSP PC2', rsp_peri_pc[:, :, 1]),
    ]

    session_stats = []
    for name, data in metrics:
        pre_means = data[:, pre_window].mean(axis=1)
        post_means = data[:, post_window].mean(axis=1)
        try:
            stat, p_val = wilcoxon(pre_means, post_means)
        except ValueError:
            p_val = 1.0
        session_stats.append({
            'session': snum, 'state': state, 'phase': phase,
            'metric': name,
            'pre_mean': pre_means.mean(), 'post_mean': post_means.mean(),
            'diff': post_means.mean() - pre_means.mean(),
            'wilcoxon_p': p_val, 'n_retreats': n_valid,
        })
    all_stats.extend(session_stats)

    # Print session stats
    print(f"\n  Pre vs Post Retreat:")
    print(f"  {'Metric':<15} {'Pre':>8} {'Post':>8} {'Diff':>8} {'p':>8} {'Sig':>4}")
    for r in session_stats:
        sig = '*' if r['wilcoxon_p'] < 0.05 else 'ns'
        print(f"  {r['metric']:<15} {r['pre_mean']:>8.4f} {r['post_mean']:>8.4f} "
              f"{r['diff']:>8.4f} {r['wilcoxon_p']:>8.4f} {sig:>4}")

    elapsed = timer.time() - t0
    print(f"\n  Session {snum} done in {elapsed:.1f}s")


# =============================================================================
# CROSS-SESSION SUMMARY FIGURE
# =============================================================================
print(f"\n{'='*70}")
print("CROSS-SESSION SUMMARY")
print(f"{'='*70}")

# Save all transitions
trans_df = pd.DataFrame(all_transitions_df)
trans_df.to_csv('data/retreat_transitions_all_sessions.csv', index=False)
print(f"Saved data/retreat_transitions_all_sessions.csv ({len(trans_df)} events)")

stats_df = pd.DataFrame(all_stats)
stats_df.to_csv('data/retreat_pre_vs_post_stats_all_sessions.csv', index=False)
print("Saved data/retreat_pre_vs_post_stats_all_sessions.csv")

# Save peri-event data per session
save_dict = {'peri_time': peri_time}
for snum, info in all_session_results.items():
    save_dict[f's{snum}_lha_peri_pop'] = info['lha_peri_pop']
    save_dict[f's{snum}_rsp_peri_pop'] = info['rsp_peri_pop']
    save_dict[f's{snum}_lha_peri_pc'] = info['lha_peri_pc']
    save_dict[f's{snum}_rsp_peri_pc'] = info['rsp_peri_pc']
np.savez('data/retreat_peri_event_all_sessions.npz', **save_dict)
print("Saved data/retreat_peri_event_all_sessions.npz")

# =========================================================================
# CROSS-SESSION FIGURE: Overlay all sessions, colored by state
# =========================================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=True)
state_colors = {'fed': '#1f77b4', 'fasted': '#d62728'}
state_styles = {'exploration': '-', 'foraging': '--'}

for col, region in enumerate(['LHA', 'RSP']):
    pop_key = f'{region.lower()}_peri_pop'

    # Population FR
    ax = axes[0, col]
    for snum in sorted(all_session_results.keys()):
        info = all_session_results[snum]
        pop_data = info[f'{region.lower()}_peri_pop']
        mean_fr = gaussian_filter1d(pop_data.mean(axis=0), SMOOTH_SIGMA)
        color = state_colors[info['state']]
        ls = state_styles[info['phase']]
        ax.plot(peri_time, mean_fr, color=color, linestyle=ls, linewidth=1.2,
                alpha=0.8, label=f"S{snum} ({info['state'][:3]}/{info['phase'][:3]}, n={info['n_valid']})")
    ax.axvline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.set_ylabel('Population z-scored FR', fontsize=10)
    ax.set_title(f'{region} — Population FR', fontsize=12)
    ax.legend(fontsize=7, loc='upper right')

    # PC1
    ax = axes[1, col]
    for snum in sorted(all_session_results.keys()):
        info = all_session_results[snum]
        pc_data = info[f'{region.lower()}_peri_pc']
        mean_pc1 = gaussian_filter1d(pc_data[:, :, 0].mean(axis=0), SMOOTH_SIGMA)
        color = state_colors[info['state']]
        ls = state_styles[info['phase']]
        ax.plot(peri_time, mean_pc1, color=color, linestyle=ls, linewidth=1.2,
                alpha=0.8, label=f"S{snum}")
    ax.axvline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.set_ylabel('PC1 score', fontsize=10)
    ax.set_xlabel('Time from retreat (s)', fontsize=10)
    ax.set_title(f'{region} — PC1', fontsize=12)

# Custom legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='#1f77b4', linestyle='-', label='Fed/Exploration'),
    Line2D([0], [0], color='#1f77b4', linestyle='--', label='Fed/Foraging'),
    Line2D([0], [0], color='#d62728', linestyle='-', label='Fasted/Exploration'),
    Line2D([0], [0], color='#d62728', linestyle='--', label='Fasted/Foraging'),
]
fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=10,
           bbox_to_anchor=(0.5, -0.02))

fig.suptitle('Cross-Session Peri-Retreat Neural Activity\n'
             'Blue=Fed, Red=Fasted | Solid=Exploration, Dashed=Foraging',
             fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0.03, 1, 0.93])
plt.savefig('figures/retreat_cross_session_overlay.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved figures/retreat_cross_session_overlay.png")

# =========================================================================
# CROSS-SESSION FIGURE: Fed vs Fasted average
# =========================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)

for col, region in enumerate(['LHA', 'RSP']):
    for row, metric_label in enumerate(['Pop FR', 'PC1']):
        ax = axes[row, col]
        for state_val, color in [('fed', '#1f77b4'), ('fasted', '#d62728')]:
            state_sessions = [s for s, info in all_session_results.items()
                              if info['state'] == state_val]
            if not state_sessions:
                continue

            if metric_label == 'Pop FR':
                all_traces = np.concatenate([all_session_results[s][f'{region.lower()}_peri_pop']
                                             for s in state_sessions])
            else:
                all_traces = np.concatenate([all_session_results[s][f'{region.lower()}_peri_pc'][:, :, 0]
                                             for s in state_sessions])

            n_total = len(all_traces)
            mean_val = gaussian_filter1d(all_traces.mean(axis=0), SMOOTH_SIGMA)
            sem_val = gaussian_filter1d(all_traces.std(axis=0) / np.sqrt(n_total), SMOOTH_SIGMA)
            ax.fill_between(peri_time, mean_val - sem_val, mean_val + sem_val,
                            alpha=0.2, color=color)
            ax.plot(peri_time, mean_val, color=color, linewidth=1.5,
                    label=f'{state_val.capitalize()} (n={n_total})')

        ax.axvline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.set_ylabel(metric_label, fontsize=10)
        if row == 1:
            ax.set_xlabel('Time from retreat (s)', fontsize=10)
        ax.set_title(f'{region} — {metric_label}', fontsize=11)
        ax.legend(fontsize=9)

fig.suptitle('Peri-Retreat: Fed vs Fasted (pooled across sessions)',
             fontsize=13, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('figures/retreat_fed_vs_fasted.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved figures/retreat_fed_vs_fasted.png")

# =========================================================================
# Print summary table
# =========================================================================
print("\n\nSUMMARY TABLE — Pre vs Post Retreat (Wilcoxon signed-rank)")
print(f"{'Session':<10} {'State':<8} {'Phase':<12} {'N':>4} "
      f"{'LHA FR p':>10} {'RSP FR p':>10} {'LHA PC1 p':>10} {'RSP PC1 p':>10}")
print("-" * 80)
for snum in sorted(all_session_results.keys()):
    info = all_session_results[snum]
    s_stats = [s for s in all_stats if s['session'] == snum]
    row = f"S{snum:<9} {info['state']:<8} {info['phase']:<12} {info['n_valid']:>4} "
    for metric in ['LHA Pop FR', 'RSP Pop FR', 'LHA PC1', 'RSP PC1']:
        st = next((s for s in s_stats if s['metric'] == metric), None)
        if st:
            p = st['wilcoxon_p']
            sig = '*' if p < 0.05 else ''
            row += f"{p:>9.4f}{sig} "
        else:
            row += f"{'N/A':>10} "
    print(row)

print("\n[DONE]")
