"""
Retreat Neural Signatures — Session 1 (M1, Coordinates 1, Fed/Exploration)

Detects time-points where mouse retreats from arena zones to Home/Ladder,
then examines peri-retreat neural activity in LHA and RSP.

Analyses:
1. Detect retreat transitions (arena -> Home/Ladder)
2. Peri-retreat firing rate (population + per-unit heatmaps) for LHA and RSP
3. Peri-retreat PCA (PC1-3 trajectories)
4. Compare retreats by source zone (transition zone vs pots vs corners)
5. Velocity-matched controls (non-retreat movements)
"""

import yaml
from pathlib import Path
import numpy as np
import pandas as pd
import spikeinterface.extractors as se
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.gridspec as gridspec
import warnings

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

# Peri-event window
PRE_SEC = 5.0    # seconds before retreat
POST_SEC = 5.0   # seconds after retreat
BIN_SEC = 0.1    # 100ms bins for neural data
SMOOTH_SIGMA = 2  # Gaussian smoothing (in bins)

session_cfg = cfg["single_probe"]["coordinates_1"]["mouse01"]["sessions"]["session_1"]
sorted_path = Path(session_cfg['sorted'])
behav_path = session_cfg['behavior']

print("=" * 70)
print("RETREAT NEURAL SIGNATURES — Session 1 (Fed/Exploration)")
print("=" * 70)

# =============================================================================
# LOAD BEHAVIOR
# =============================================================================
print("\n[1] Loading behavior data...")
df_raw = pd.read_csv(behav_path, header=None)
var_names = df_raw.iloc[:, 0].values
time_vals = df_raw.iloc[1, 1:].astype(float).values
data = df_raw.iloc[:, 1:].values

behav = {'time': time_vals}
for i, name in enumerate(var_names):
    if isinstance(name, str):
        behav[name.strip()] = data[i].astype(float)

n_bins_behav = len(time_vals)
print(f"  Duration: {time_vals[-1]:.1f}s, {n_bins_behav} time bins")

# Build zone labels
zones = np.full(n_bins_behav, 'other', dtype=object)
priority_order = [
    'Right corner', 'Left corner', 'Arna center', 'Foraging arena',
    'Home', 'Ladder', 'Transition zone',
    'Pot-1 zone', 'Pot-2 zone', 'Pot-3 zone', 'Pot-4 zone',
    'Pot-1', 'Pot-2', 'Pot-3', 'Pot-4',
]
for var_name in priority_order:
    if var_name in behav:
        mask = behav[var_name] > 0.5
        zones[mask] = var_name

# =============================================================================
# DETECT RETREAT TRANSITIONS
# =============================================================================
print("\n[2] Detecting retreat transitions...")

arena_zones = {'Pot-1', 'Pot-2', 'Pot-3', 'Pot-4',
               'Pot-1 zone', 'Pot-2 zone', 'Pot-3 zone', 'Pot-4 zone',
               'Arna center', 'Foraging arena', 'Transition zone',
               'Right corner', 'Left corner'}
retreat_destinations = {'Home', 'Ladder'}

transitions = []
for i in range(1, n_bins_behav):
    if zones[i] != zones[i-1]:
        if zones[i-1] in arena_zones and zones[i] in retreat_destinations:
            transitions.append({
                'time': time_vals[i],
                'time_idx': i,
                'from_zone': zones[i-1],
                'to_zone': zones[i],
                'velocity': behav['Velocity'][i] if 'Velocity' in behav else np.nan,
            })

print(f"  Found {len(transitions)} retreat transitions")

# Categorize by source
for t in transitions:
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

cat_counts = {}
for t in transitions:
    cat_counts[t['source_category']] = cat_counts.get(t['source_category'], 0) + 1
for cat, cnt in sorted(cat_counts.items(), key=lambda x: -x[1]):
    print(f"    {cat}: {cnt}")

# =============================================================================
# LOAD NEURAL DATA
# =============================================================================
print("\n[3] Loading neural data...")

ci = sorted_path / "cluster_info.tsv"
df_ci = pd.read_csv(ci, sep='\t')
label_col = 'group' if ('group' in df_ci.columns and df_ci['group'].eq('good').any()) else 'KSLabel'
good = df_ci[(df_ci[label_col] == 'good') & (df_ci['fr'] > MIN_FR) & (df_ci['amp'] > MIN_AMP)]
lha_ids = good[good['depth'] < LHA_DEPTH_MAX]['cluster_id'].values
rsp_ids = good[good['depth'] >= LHA_DEPTH_MAX]['cluster_id'].values

sorting = se.read_kilosort(sorted_path)
avail = set(sorting.get_unit_ids())
lha_ids = np.array([u for u in lha_ids if u in avail])
rsp_ids = np.array([u for u in rsp_ids if u in avail])

print(f"  LHA: {len(lha_ids)} good units")
print(f"  RSP: {len(rsp_ids)} good units")

# Build binned spike trains aligned to behavior time
rec_duration = time_vals[-1] + BIN_SEC
bin_edges = np.arange(0, rec_duration + BIN_SEC, BIN_SEC)
n_neural_bins = len(bin_edges) - 1
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

def bin_spike_train(uid):
    st = sorting.get_unit_spike_train(uid) / FS
    counts, _ = np.histogram(st, bins=bin_edges)
    fr = counts / BIN_SEC  # firing rate in Hz
    return fr

print("  Binning spike trains...")
lha_fr = np.array([bin_spike_train(u) for u in lha_ids])
rsp_fr = np.array([bin_spike_train(u) for u in rsp_ids])

# Z-score each unit
def zscore_array(arr):
    mu = arr.mean(axis=1, keepdims=True)
    sd = arr.std(axis=1, keepdims=True)
    sd[sd < 1e-6] = 1
    return (arr - mu) / sd

lha_z = zscore_array(lha_fr)
rsp_z = zscore_array(rsp_fr)

print(f"  Neural bins: {n_neural_bins}, bin size: {BIN_SEC}s")

# PCA
def fit_pca(z_arr, n_components=3):
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(z_arr.T)  # (time, n_components)
    return pca, scores

lha_pca, lha_scores = fit_pca(lha_z)
rsp_pca, rsp_scores = fit_pca(rsp_z)
print(f"  LHA PCA var explained: {lha_pca.explained_variance_ratio_[:3].sum():.1%}")
print(f"  RSP PCA var explained: {rsp_pca.explained_variance_ratio_[:3].sum():.1%}")

# =============================================================================
# EXTRACT PERI-RETREAT WINDOWS
# =============================================================================
print("\n[4] Extracting peri-retreat neural windows...")

pre_bins = int(PRE_SEC / BIN_SEC)
post_bins = int(POST_SEC / BIN_SEC)
window_bins = pre_bins + post_bins
peri_time = np.arange(-pre_bins, post_bins) * BIN_SEC  # relative time

def get_neural_bin(event_time):
    """Convert event time to nearest neural bin index."""
    return int(np.searchsorted(bin_centers, event_time))

# Collect peri-event data
valid_transitions = []
lha_peri_fr = []  # (n_transitions, n_lha, window_bins)
rsp_peri_fr = []
lha_peri_pc = []  # (n_transitions, window_bins, 3)
rsp_peri_pc = []
lha_peri_pop = []  # population mean FR
rsp_peri_pop = []

for t in transitions:
    center_bin = get_neural_bin(t['time'])
    start = center_bin - pre_bins
    end = center_bin + post_bins

    if start < 0 or end > n_neural_bins:
        continue

    valid_transitions.append(t)
    lha_peri_fr.append(lha_z[:, start:end])
    rsp_peri_fr.append(rsp_z[:, start:end])
    lha_peri_pc.append(lha_scores[start:end, :3])
    rsp_peri_pc.append(rsp_scores[start:end, :3])
    lha_peri_pop.append(lha_z[:, start:end].mean(axis=0))
    rsp_peri_pop.append(rsp_z[:, start:end].mean(axis=0))

lha_peri_fr = np.array(lha_peri_fr)   # (n_valid, n_lha, window)
rsp_peri_fr = np.array(rsp_peri_fr)
lha_peri_pc = np.array(lha_peri_pc)   # (n_valid, window, 3)
rsp_peri_pc = np.array(rsp_peri_pc)
lha_peri_pop = np.array(lha_peri_pop)  # (n_valid, window)
rsp_peri_pop = np.array(rsp_peri_pop)

n_valid = len(valid_transitions)
print(f"  Valid transitions (with full window): {n_valid}/{len(transitions)}")

# =============================================================================
# FIGURE 1: Population peri-retreat FR + PC1 (all retreats, mean ± SEM)
# =============================================================================
print("\n[5] Plotting Figure 1: Population peri-retreat average...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)

for col, (region, pop_data, pc_data, color) in enumerate([
    ('LHA', lha_peri_pop, lha_peri_pc, '#e74c3c'),
    ('RSP', rsp_peri_pop, rsp_peri_pc, '#27ae60'),
]):
    # Population mean FR
    ax = axes[0, col]
    mean_fr = pop_data.mean(axis=0)
    sem_fr = pop_data.std(axis=0) / np.sqrt(n_valid)
    smooth_mean = gaussian_filter1d(mean_fr, SMOOTH_SIGMA)
    smooth_sem = gaussian_filter1d(sem_fr, SMOOTH_SIGMA)

    ax.fill_between(peri_time, smooth_mean - smooth_sem, smooth_mean + smooth_sem,
                     alpha=0.3, color=color)
    ax.plot(peri_time, smooth_mean, color=color, linewidth=1.5)
    ax.axvline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.set_ylabel('Population z-scored FR', fontsize=10)
    ax.set_title(f'{region} — Mean Population FR (n={n_valid} retreats)', fontsize=11)
    ax.tick_params(labelsize=9)

    # PC1
    ax = axes[1, col]
    pc1 = pc_data[:, :, 0]
    mean_pc1 = pc1.mean(axis=0)
    sem_pc1 = pc1.std(axis=0) / np.sqrt(n_valid)
    smooth_pc1 = gaussian_filter1d(mean_pc1, SMOOTH_SIGMA)
    smooth_sem_pc1 = gaussian_filter1d(sem_pc1, SMOOTH_SIGMA)

    ax.fill_between(peri_time, smooth_pc1 - smooth_sem_pc1, smooth_pc1 + smooth_sem_pc1,
                     alpha=0.3, color=color)
    ax.plot(peri_time, smooth_pc1, color=color, linewidth=1.5)
    ax.axvline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.set_ylabel('PC1 score', fontsize=10)
    ax.set_xlabel('Time from retreat (s)', fontsize=10)
    ax.set_title(f'{region} — PC1 (n={n_valid} retreats)', fontsize=11)
    ax.tick_params(labelsize=9)

fig.suptitle('Session 1 (Fed/Exploration) — Peri-Retreat Neural Activity\n'
             f'All retreats: arena zones → Home/Ladder (±{PRE_SEC:.0f}s window, {BIN_SEC*1000:.0f}ms bins)',
             fontsize=13, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig('figures/retreat_population_s1.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved figures/retreat_population_s1.png")

# =============================================================================
# FIGURE 2: Per-unit heatmaps (sorted by peak response time)
# =============================================================================
print("\n[6] Plotting Figure 2: Per-unit heatmaps...")

fig, axes = plt.subplots(1, 2, figsize=(16, 10))

for col, (region, ids, peri_data, color) in enumerate([
    ('LHA', lha_ids, lha_peri_fr, '#e74c3c'),
    ('RSP', rsp_ids, rsp_peri_fr, '#27ae60'),
]):
    ax = axes[col]
    # Average across transitions per unit: (n_units, window)
    mean_per_unit = peri_data.mean(axis=0)  # (n_units, window)

    # Sort by peak time
    peak_times = np.argmax(mean_per_unit, axis=1)
    sort_idx = np.argsort(peak_times)
    sorted_data = mean_per_unit[sort_idx]

    # Smooth each row
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

fig.suptitle('Session 1 — Per-Unit Peri-Retreat Heatmap\n'
             'Units sorted by peak response time',
             fontsize=13, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig('figures/retreat_unit_heatmap_s1.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved figures/retreat_unit_heatmap_s1.png")

# =============================================================================
# FIGURE 3: By source category (transition zone vs corners vs pot areas)
# =============================================================================
print("\n[7] Plotting Figure 3: Retreat by source category...")

categories = ['transition', 'corner', 'pot_area']
cat_labels = {'transition': 'Transition Zone', 'corner': 'Corners', 'pot_area': 'Pot Areas'}
cat_colors = {'transition': '#3498db', 'corner': '#e67e22', 'pot_area': '#9b59b6'}

# Group indices by category
cat_indices = {cat: [] for cat in categories}
for i, t in enumerate(valid_transitions):
    if t['source_category'] in cat_indices:
        cat_indices[t['source_category']].append(i)

fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)

for col, (region, pop_data, pc_data) in enumerate([
    ('LHA', lha_peri_pop, lha_peri_pc),
    ('RSP', rsp_peri_pop, rsp_peri_pc),
]):
    # Population FR by category
    ax = axes[0, col]
    for cat in categories:
        idx = cat_indices[cat]
        if len(idx) < 3:
            continue
        cat_data = pop_data[idx]
        mean_val = gaussian_filter1d(cat_data.mean(axis=0), SMOOTH_SIGMA)
        sem_val = gaussian_filter1d(cat_data.std(axis=0) / np.sqrt(len(idx)), SMOOTH_SIGMA)
        ax.fill_between(peri_time, mean_val - sem_val, mean_val + sem_val,
                         alpha=0.2, color=cat_colors[cat])
        ax.plot(peri_time, mean_val, color=cat_colors[cat], linewidth=1.5,
                label=f'{cat_labels[cat]} (n={len(idx)})')
    ax.axvline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.set_ylabel('Population z-scored FR', fontsize=10)
    ax.set_title(f'{region} — Population FR by Source', fontsize=11)
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=9)

    # PC1 by category
    ax = axes[1, col]
    for cat in categories:
        idx = cat_indices[cat]
        if len(idx) < 3:
            continue
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
    ax.tick_params(labelsize=9)

fig.suptitle('Session 1 — Peri-Retreat by Source Zone Category',
             fontsize=13, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('figures/retreat_by_source_s1.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved figures/retreat_by_source_s1.png")

# =============================================================================
# FIGURE 4: Pre vs Post retreat stats (paired comparison)
# =============================================================================
print("\n[8] Plotting Figure 4: Pre vs Post retreat statistics...")

from scipy.stats import wilcoxon

pre_window = slice(0, pre_bins)
post_window = slice(pre_bins, window_bins)

results = []
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

metrics = [
    ('LHA Pop FR', lha_peri_pop, '#e74c3c'),
    ('RSP Pop FR', rsp_peri_pop, '#27ae60'),
    ('LHA PC1', lha_peri_pc[:, :, 0], '#c0392b'),
    ('RSP PC1', rsp_peri_pc[:, :, 0], '#219a52'),
    ('LHA PC2', lha_peri_pc[:, :, 1], '#e67e22'),
    ('RSP PC2', rsp_peri_pc[:, :, 1], '#2980b9'),
]

for idx, (name, data, color) in enumerate(metrics):
    row, col = idx // 3, idx % 3
    ax = axes[row, col]

    pre_means = data[:, pre_window].mean(axis=1)
    post_means = data[:, post_window].mean(axis=1)

    stat, p_val = wilcoxon(pre_means, post_means)
    mean_diff = post_means.mean() - pre_means.mean()

    results.append({
        'metric': name,
        'pre_mean': pre_means.mean(),
        'post_mean': post_means.mean(),
        'diff': mean_diff,
        'wilcoxon_p': p_val,
        'n': len(pre_means),
    })

    ax.scatter(pre_means, post_means, alpha=0.3, s=15, color=color)
    lims = [min(pre_means.min(), post_means.min()), max(pre_means.max(), post_means.max())]
    margin = (lims[1] - lims[0]) * 0.1
    lims = [lims[0] - margin, lims[1] + margin]
    ax.plot(lims, lims, 'k--', linewidth=0.8, alpha=0.5)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel('Pre-retreat mean', fontsize=9)
    ax.set_ylabel('Post-retreat mean', fontsize=9)

    sig_str = f'p={p_val:.4f}' + (' *' if p_val < 0.05 else ' ns')
    ax.set_title(f'{name}\n{sig_str}', fontsize=10)
    ax.tick_params(labelsize=8)

fig.suptitle(f'Session 1 — Pre vs Post Retreat (n={n_valid} retreats)\n'
             f'Pre: {PRE_SEC:.0f}s before | Post: {POST_SEC:.0f}s after',
             fontsize=13, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.savefig('figures/retreat_pre_vs_post_s1.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved figures/retreat_pre_vs_post_s1.png")

# Print stats table
print("\n  Pre vs Post Retreat Statistics:")
print(f"  {'Metric':<15} {'Pre Mean':>10} {'Post Mean':>10} {'Diff':>10} {'p-value':>10} {'Sig':>5}")
for r in results:
    sig = '*' if r['wilcoxon_p'] < 0.05 else 'ns'
    print(f"  {r['metric']:<15} {r['pre_mean']:>10.4f} {r['post_mean']:>10.4f} "
          f"{r['diff']:>10.4f} {r['wilcoxon_p']:>10.4f} {sig:>5}")

# =============================================================================
# SAVE DATA
# =============================================================================
print("\n[9] Saving data...")

# Save transition events
trans_df = pd.DataFrame(valid_transitions)
trans_df.to_csv('data/retreat_transitions_s1.csv', index=False)
print(f"  Saved data/retreat_transitions_s1.csv ({len(trans_df)} events)")

# Save stats
stats_df = pd.DataFrame(results)
stats_df.to_csv('data/retreat_pre_vs_post_stats_s1.csv', index=False)
print("  Saved data/retreat_pre_vs_post_stats_s1.csv")

# Save peri-event traces
np.savez('data/retreat_peri_event_s1.npz',
         peri_time=peri_time,
         lha_peri_pop=lha_peri_pop,
         rsp_peri_pop=rsp_peri_pop,
         lha_peri_pc=lha_peri_pc,
         rsp_peri_pc=rsp_peri_pc)
print("  Saved data/retreat_peri_event_s1.npz")

print("\n[DONE]")
