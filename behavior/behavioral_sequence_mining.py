"""
Behavioral Sequence Mining — All 8 Sessions (M1, Coordinates 1)

Goal: Detect hidden behavioral patterns from positional data + feeding/digging
that a human observer would miss.

Analyses:
1. Zone transition probability matrices (per session + grouped)
2. N-gram motifs (common 2/3/4-zone sequences)
3. Dwell time distributions per zone
4. Temporal evolution of zone preferences (early vs mid vs late session)
5. Pre-feeding and pre-digging zone sequences
6. Return time distributions (time between revisits to same zone)
7. Behavioral entropy over time (predictability changes)
8. Cross-session consistency of transition structure
"""

import yaml
from pathlib import Path
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from scipy.stats import entropy as sp_entropy
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.distance import cosine as cosine_dist
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import warnings

warnings.filterwarnings('ignore')

with open("paths.yaml") as f:
    cfg = yaml.safe_load(f)

sessions_cfg = cfg["single_probe"]["coordinates_1"]["mouse01"]["sessions"]
session_meta = {
    1: ('fed', 'exploration'), 2: ('fed', 'foraging'),
    3: ('fed', 'exploration'), 4: ('fed', 'foraging'),
    5: ('fasted', 'exploration'), 6: ('fasted', 'foraging'),
    7: ('fasted', 'exploration'), 8: ('fasted', 'foraging'),
}

priority_order = [
    'Right corner', 'Left corner', 'Arna center', 'Foraging arena',
    'Home', 'Ladder', 'Transition zone',
    'Pot-1 zone', 'Pot-2 zone', 'Pot-3 zone', 'Pot-4 zone',
    'Pot-1', 'Pot-2', 'Pot-3', 'Pot-4',
]

# Simplified zone labels for sequence analysis
ZONE_SIMPLIFY = {
    'Pot-1': 'P1', 'Pot-2': 'P2', 'Pot-3': 'P3', 'Pot-4': 'P4',
    'Pot-1 zone': 'P1z', 'Pot-2 zone': 'P2z', 'Pot-3 zone': 'P3z', 'Pot-4 zone': 'P4z',
    'Home': 'H', 'Ladder': 'L', 'Transition zone': 'T',
    'Right corner': 'RC', 'Left corner': 'LC',
    'Arna center': 'AC', 'Foraging arena': 'FA',
    'other': 'O',
}

ZONE_COLORS = {
    'P1': '#ff9999', 'P2': '#ff6666', 'P3': '#ff3333', 'P4': '#cc0000',
    'P1z': '#ffcccc', 'P2z': '#ffaaaa', 'P3z': '#ff8888', 'P4z': '#ff5555',
    'H': '#6699cc', 'L': '#99cc66', 'T': '#cccc66',
    'RC': '#cc99cc', 'LC': '#bb88bb',
    'AC': '#dddd88', 'FA': '#88bb88', 'O': '#e0e0e0',
}


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


def get_zone_sequence(behav):
    """Get simplified zone label at each time bin."""
    n = len(behav['time'])
    zones = np.full(n, 'other', dtype=object)
    for var_name in priority_order:
        if var_name in behav:
            mask = behav[var_name] > 0.5
            zones[mask] = var_name
    simplified = np.array([ZONE_SIMPLIFY.get(z, 'O') for z in zones])
    return zones, simplified


def get_zone_bouts(simplified, time_vals):
    """Extract consecutive zone bouts: (zone, start_time, end_time, duration)."""
    bouts = []
    current_zone = simplified[0]
    start_idx = 0
    for i in range(1, len(simplified)):
        if simplified[i] != current_zone:
            bouts.append({
                'zone': current_zone,
                'start_time': time_vals[start_idx],
                'end_time': time_vals[i - 1],
                'duration': time_vals[i - 1] - time_vals[start_idx],
                'start_idx': start_idx,
                'end_idx': i - 1,
            })
            current_zone = simplified[i]
            start_idx = i
    # Last bout
    bouts.append({
        'zone': current_zone,
        'start_time': time_vals[start_idx],
        'end_time': time_vals[-1],
        'duration': time_vals[-1] - time_vals[start_idx],
        'start_idx': start_idx,
        'end_idx': len(simplified) - 1,
    })
    return bouts


def compute_transition_matrix(bouts, zones_list):
    """Compute transition probability matrix from zone bouts."""
    n_zones = len(zones_list)
    z2i = {z: i for i, z in enumerate(zones_list)}
    counts = np.zeros((n_zones, n_zones))
    for i in range(len(bouts) - 1):
        z_from = bouts[i]['zone']
        z_to = bouts[i + 1]['zone']
        if z_from in z2i and z_to in z2i:
            counts[z2i[z_from], z2i[z_to]] += 1
    # Normalize rows to probabilities
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    prob = counts / row_sums
    return counts, prob


def get_ngrams(bouts, n):
    """Extract n-gram sequences from bout zones."""
    zones = [b['zone'] for b in bouts]
    ngrams = []
    for i in range(len(zones) - n + 1):
        ngrams.append(tuple(zones[i:i + n]))
    return ngrams


def compute_behavioral_entropy(simplified, time_vals, window_sec=60, step_sec=10):
    """Sliding-window entropy of zone occupancy."""
    dt = np.median(np.diff(time_vals))
    window_bins = int(window_sec / dt)
    step_bins = int(step_sec / dt)

    all_zones = sorted(set(simplified))
    z2i = {z: i for i, z in enumerate(all_zones)}
    n_zones = len(all_zones)

    centers = []
    entropies = []

    for start in range(0, len(simplified) - window_bins, step_bins):
        end = start + window_bins
        window = simplified[start:end]
        counts = np.zeros(n_zones)
        for z in window:
            counts[z2i[z]] += 1
        counts /= counts.sum()
        h = sp_entropy(counts + 1e-10)
        centers.append(time_vals[start + window_bins // 2])
        entropies.append(h)

    return np.array(centers), np.array(entropies)


def get_event_times(behav, event_name):
    """Get onset times for a binary behavioral event."""
    if event_name not in behav:
        return np.array([])
    signal = behav[event_name]
    onsets = np.where(np.diff(signal > 0.5) & (signal[1:] > 0.5))[0] + 1
    return behav['time'][onsets]


# =============================================================================
# LOAD ALL SESSIONS
# =============================================================================
print("=" * 70)
print("BEHAVIORAL SEQUENCE MINING — All 8 Sessions")
print("=" * 70)

all_data = {}
all_zones_set = set()

for snum in range(1, 9):
    state, phase = session_meta[snum]
    sc = sessions_cfg[f"session_{snum}"]
    behav_path = sc.get('behavior')

    if not behav_path or not Path(behav_path).exists():
        print(f"S{snum}: no behavior data — SKIP")
        continue

    behav = load_behavior(behav_path)
    zones_raw, zones_simple = get_zone_sequence(behav)
    bouts = get_zone_bouts(zones_simple, behav['time'])

    feeding_onsets = get_event_times(behav, 'Feeding')
    digging_onsets = get_event_times(behav, 'Digging')

    all_data[snum] = {
        'state': state, 'phase': phase, 'behav': behav,
        'zones_simple': zones_simple, 'bouts': bouts,
        'feeding_onsets': feeding_onsets, 'digging_onsets': digging_onsets,
        'time': behav['time'],
    }
    all_zones_set.update(set(zones_simple))

    n_bouts = len(bouts)
    duration = behav['time'][-1]
    print(f"S{snum} ({state}/{phase}): {n_bouts} bouts, {duration:.0f}s, "
          f"{len(feeding_onsets)} feed onsets, {len(digging_onsets)} dig onsets")

zones_list = sorted(all_zones_set)
print(f"\nZones present: {zones_list}")


# =============================================================================
# 1. TRANSITION PROBABILITY MATRICES
# =============================================================================
print("\n\n" + "=" * 70)
print("1. TRANSITION PROBABILITY MATRICES")
print("=" * 70)

fig, axes = plt.subplots(2, 4, figsize=(28, 14))
session_trans = {}

for idx, snum in enumerate(sorted(all_data.keys())):
    info = all_data[snum]
    counts, prob = compute_transition_matrix(info['bouts'], zones_list)
    session_trans[snum] = {'counts': counts, 'prob': prob}

    row, col = idx // 4, idx % 4
    ax = axes[row, col]
    im = ax.imshow(prob, cmap='YlOrRd', vmin=0, vmax=0.5, aspect='auto')
    ax.set_xticks(range(len(zones_list)))
    ax.set_yticks(range(len(zones_list)))
    ax.set_xticklabels(zones_list, fontsize=6, rotation=45)
    ax.set_yticklabels(zones_list, fontsize=6)
    ax.set_title(f"S{snum} ({info['state'][:3]}/{info['phase'][:3]})", fontsize=10)

    # Annotate non-zero cells
    for i in range(len(zones_list)):
        for j in range(len(zones_list)):
            if prob[i, j] > 0.05:
                ax.text(j, i, f'{prob[i, j]:.2f}', ha='center', va='center', fontsize=5,
                        color='white' if prob[i, j] > 0.25 else 'black')

plt.colorbar(im, ax=axes, fraction=0.02, pad=0.02, label='P(next zone)')
fig.suptitle('Zone Transition Probabilities (from row → to column)', fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 0.95, 0.95])
plt.savefig('figures/behav_transition_matrices.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved figures/behav_transition_matrices.png")

# Compute mean transition matrices for exploration vs foraging
exp_probs = []
for_probs = []
fed_probs = []
fas_probs = []

for snum, trans in session_trans.items():
    info = all_data[snum]
    if info['phase'] == 'exploration':
        exp_probs.append(trans['prob'])
    else:
        for_probs.append(trans['prob'])
    if info['state'] == 'fed':
        fed_probs.append(trans['prob'])
    else:
        fas_probs.append(trans['prob'])

fig, axes = plt.subplots(2, 2, figsize=(16, 14))
for ax, (title, probs) in zip(axes.flat, [
    ('Exploration (S1,3,5,7)', exp_probs), ('Foraging (S2,4,6,8)', for_probs),
    ('Fed (S1-4)', fed_probs), ('Fasted (S5-8)', fas_probs),
]):
    mean_prob = np.mean(probs, axis=0)
    im = ax.imshow(mean_prob, cmap='YlOrRd', vmin=0, vmax=0.5, aspect='auto')
    ax.set_xticks(range(len(zones_list)))
    ax.set_yticks(range(len(zones_list)))
    ax.set_xticklabels(zones_list, fontsize=7, rotation=45)
    ax.set_yticklabels(zones_list, fontsize=7)
    ax.set_title(title, fontsize=11)
    for i in range(len(zones_list)):
        for j in range(len(zones_list)):
            if mean_prob[i, j] > 0.05:
                ax.text(j, i, f'{mean_prob[i, j]:.2f}', ha='center', va='center', fontsize=6,
                        color='white' if mean_prob[i, j] > 0.25 else 'black')

plt.colorbar(im, ax=axes, fraction=0.02, pad=0.02, label='P(next zone)')
fig.suptitle('Mean Transition Probabilities by Condition', fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 0.95, 0.95])
plt.savefig('figures/behav_transition_grouped.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved figures/behav_transition_grouped.png")

# Quantify transition matrix similarity across sessions
print("\nTransition matrix cosine similarity (1 = identical):")
sim_matrix = np.zeros((8, 8))
snums = sorted(all_data.keys())
for i, s1 in enumerate(snums):
    for j, s2 in enumerate(snums):
        p1 = session_trans[s1]['prob'].flatten()
        p2 = session_trans[s2]['prob'].flatten()
        sim_matrix[i, j] = 1 - cosine_dist(p1 + 1e-10, p2 + 1e-10)

print(f"  {'':>4}", end='')
for s in snums:
    print(f"  S{s}", end='')
print()
for i, s1 in enumerate(snums):
    print(f"  S{s1}", end='')
    for j in range(len(snums)):
        print(f" {sim_matrix[i, j]:.2f}", end='')
    print()

# Group similarities
exp_sessions = [s for s in snums if all_data[s]['phase'] == 'exploration']
for_sessions = [s for s in snums if all_data[s]['phase'] == 'foraging']
fed_sessions = [s for s in snums if all_data[s]['state'] == 'fed']
fas_sessions = [s for s in snums if all_data[s]['state'] == 'fasted']

def mean_sim(sessions):
    sims = []
    for i, s1 in enumerate(sessions):
        for j, s2 in enumerate(sessions):
            if i < j:
                idx1 = snums.index(s1)
                idx2 = snums.index(s2)
                sims.append(sim_matrix[idx1, idx2])
    return np.mean(sims) if sims else 0

print(f"\n  Within-exploration similarity: {mean_sim(exp_sessions):.3f}")
print(f"  Within-foraging similarity: {mean_sim(for_sessions):.3f}")
print(f"  Within-fed similarity: {mean_sim(fed_sessions):.3f}")
print(f"  Within-fasted similarity: {mean_sim(fas_sessions):.3f}")

# Cross-group
cross_exp_for = []
for s1 in exp_sessions:
    for s2 in for_sessions:
        cross_exp_for.append(sim_matrix[snums.index(s1), snums.index(s2)])
cross_fed_fas = []
for s1 in fed_sessions:
    for s2 in fas_sessions:
        cross_fed_fas.append(sim_matrix[snums.index(s1), snums.index(s2)])
print(f"  Cross exp-vs-for similarity: {np.mean(cross_exp_for):.3f}")
print(f"  Cross fed-vs-fas similarity: {np.mean(cross_fed_fas):.3f}")


# =============================================================================
# 2. N-GRAM MOTIFS
# =============================================================================
print("\n\n" + "=" * 70)
print("2. N-GRAM MOTIFS")
print("=" * 70)

for n in [2, 3, 4]:
    print(f"\n--- {n}-grams ---")

    # Per session
    ngram_counts_by_session = {}
    all_ngrams = Counter()

    for snum in sorted(all_data.keys()):
        info = all_data[snum]
        ngrams = get_ngrams(info['bouts'], n)
        counts = Counter(ngrams)
        ngram_counts_by_session[snum] = counts
        all_ngrams.update(ngrams)

    # Top n-grams overall
    print(f"\n  Top 15 {n}-grams (all sessions combined):")
    for gram, count in all_ngrams.most_common(15):
        label = ' -> '.join(gram)
        # Which sessions
        sessions_with = [f"S{s}" for s in sorted(all_data.keys())
                         if gram in ngram_counts_by_session[s]]
        print(f"    {label}: {count} total ({', '.join(sessions_with)})")

    # Motifs unique to exploration vs foraging
    exp_ngrams = Counter()
    for_ngrams = Counter()
    for snum in sorted(all_data.keys()):
        if all_data[snum]['phase'] == 'exploration':
            exp_ngrams.update(ngram_counts_by_session[snum])
        else:
            for_ngrams.update(ngram_counts_by_session[snum])

    # Enriched in exploration
    exp_enriched = []
    for gram in exp_ngrams:
        exp_c = exp_ngrams[gram]
        for_c = for_ngrams.get(gram, 0)
        if exp_c >= 5 and (for_c == 0 or exp_c / max(for_c, 1) > 3):
            exp_enriched.append((gram, exp_c, for_c))
    exp_enriched.sort(key=lambda x: -x[1])

    for_enriched = []
    for gram in for_ngrams:
        for_c = for_ngrams[gram]
        exp_c = exp_ngrams.get(gram, 0)
        if for_c >= 5 and (exp_c == 0 or for_c / max(exp_c, 1) > 3):
            for_enriched.append((gram, for_c, exp_c))
    for_enriched.sort(key=lambda x: -x[1])

    if exp_enriched:
        print(f"\n  Exploration-enriched {n}-grams (>3x ratio, count>=5):")
        for gram, ec, fc in exp_enriched[:10]:
            print(f"    {' -> '.join(gram)}: exp={ec}, for={fc}")

    if for_enriched:
        print(f"\n  Foraging-enriched {n}-grams (>3x ratio, count>=5):")
        for gram, fc, ec in for_enriched[:10]:
            print(f"    {' -> '.join(gram)}: for={fc}, exp={ec}")

    # Fed vs fasted enrichment
    fed_ngrams = Counter()
    fas_ngrams = Counter()
    for snum in sorted(all_data.keys()):
        if all_data[snum]['state'] == 'fed':
            fed_ngrams.update(ngram_counts_by_session[snum])
        else:
            fas_ngrams.update(ngram_counts_by_session[snum])

    fed_enriched = []
    for gram in fed_ngrams:
        fc = fed_ngrams[gram]
        fac = fas_ngrams.get(gram, 0)
        if fc >= 5 and (fac == 0 or fc / max(fac, 1) > 3):
            fed_enriched.append((gram, fc, fac))
    fed_enriched.sort(key=lambda x: -x[1])

    fas_enriched = []
    for gram in fas_ngrams:
        fac = fas_ngrams[gram]
        fc = fed_ngrams.get(gram, 0)
        if fac >= 5 and (fc == 0 or fac / max(fc, 1) > 3):
            fas_enriched.append((gram, fac, fc))
    fas_enriched.sort(key=lambda x: -x[1])

    if fed_enriched:
        print(f"\n  Fed-enriched {n}-grams (>3x ratio, count>=5):")
        for gram, fc, fac in fed_enriched[:10]:
            print(f"    {' -> '.join(gram)}: fed={fc}, fasted={fac}")

    if fas_enriched:
        print(f"\n  Fasted-enriched {n}-grams (>3x ratio, count>=5):")
        for gram, fac, fc in fas_enriched[:10]:
            print(f"    {' -> '.join(gram)}: fasted={fac}, fed={fc}")


# =============================================================================
# 3. DWELL TIME DISTRIBUTIONS
# =============================================================================
print("\n\n" + "=" * 70)
print("3. DWELL TIME DISTRIBUTIONS")
print("=" * 70)

# Collect dwell times per zone per condition
dwell_by_zone = defaultdict(lambda: defaultdict(list))

for snum in sorted(all_data.keys()):
    info = all_data[snum]
    condition = f"{info['state']}_{info['phase']}"
    for bout in info['bouts']:
        if bout['duration'] > 0:
            dwell_by_zone[bout['zone']][condition].append(bout['duration'])

# Plot
conditions = ['fed_exploration', 'fed_foraging', 'fasted_exploration', 'fasted_foraging']
cond_colors = {'fed_exploration': '#1f77b4', 'fed_foraging': '#aec7e8',
               'fasted_exploration': '#d62728', 'fasted_foraging': '#ff9896'}
cond_labels = {'fed_exploration': 'Fed/Exp', 'fed_foraging': 'Fed/For',
               'fasted_exploration': 'Fas/Exp', 'fasted_foraging': 'Fas/For'}

key_zones = ['H', 'L', 'T', 'P1', 'P2', 'P3', 'P4', 'AC', 'FA']
fig, axes = plt.subplots(3, 3, figsize=(16, 14))

for idx, zone in enumerate(key_zones):
    row, col = idx // 3, idx % 3
    ax = axes[row, col]

    for cond in conditions:
        dwells = dwell_by_zone[zone].get(cond, [])
        if len(dwells) >= 3:
            ax.hist(dwells, bins=30, alpha=0.4, color=cond_colors[cond],
                    label=f'{cond_labels[cond]} (n={len(dwells)}, med={np.median(dwells):.1f}s)',
                    density=True)

    ax.set_title(f'{zone}', fontsize=11, fontweight='bold')
    ax.set_xlabel('Dwell time (s)', fontsize=9)
    ax.set_ylabel('Density', fontsize=9)
    ax.legend(fontsize=6)
    ax.tick_params(labelsize=8)

fig.suptitle('Dwell Time Distributions by Zone and Condition', fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('figures/behav_dwell_times.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved figures/behav_dwell_times.png")

# Print summary stats
print("\nMedian dwell times (seconds):")
print(f"  {'Zone':<6}", end='')
for cond in conditions:
    print(f"  {cond_labels[cond]:>10}", end='')
print()
for zone in key_zones:
    print(f"  {zone:<6}", end='')
    for cond in conditions:
        dwells = dwell_by_zone[zone].get(cond, [])
        if dwells:
            print(f"  {np.median(dwells):>9.1f}s", end='')
        else:
            print(f"  {'—':>10}", end='')
    print()


# =============================================================================
# 4. TEMPORAL EVOLUTION OF ZONE PREFERENCES
# =============================================================================
print("\n\n" + "=" * 70)
print("4. TEMPORAL EVOLUTION — Zone Preference Shifts")
print("=" * 70)

fig, axes = plt.subplots(2, 4, figsize=(28, 10), sharex=False)

for idx, snum in enumerate(sorted(all_data.keys())):
    info = all_data[snum]
    time_vals = info['time']
    zones = info['zones_simple']
    duration = time_vals[-1]

    # Divide session into 5 equal epochs
    n_epochs = 5
    epoch_edges = np.linspace(0, duration, n_epochs + 1)

    # Zone fraction per epoch
    track_zones = ['H', 'L', 'T', 'P1', 'P2', 'P3', 'P4', 'AC', 'FA']
    epoch_fracs = np.zeros((n_epochs, len(track_zones)))

    for ep in range(n_epochs):
        mask = (time_vals >= epoch_edges[ep]) & (time_vals < epoch_edges[ep + 1])
        epoch_zones = zones[mask]
        n_total = len(epoch_zones)
        if n_total == 0:
            continue
        for zi, z in enumerate(track_zones):
            epoch_fracs[ep, zi] = np.sum(epoch_zones == z) / n_total

    row, col = idx // 4, idx % 4
    ax = axes[row, col]

    bottom = np.zeros(n_epochs)
    for zi, z in enumerate(track_zones):
        color = ZONE_COLORS.get(z, '#cccccc')
        ax.bar(range(n_epochs), epoch_fracs[:, zi], bottom=bottom, color=color,
               label=z, width=0.8)
        bottom += epoch_fracs[:, zi]

    ax.set_xticks(range(n_epochs))
    ax.set_xticklabels([f'{epoch_edges[i]:.0f}-{epoch_edges[i+1]:.0f}s'
                         for i in range(n_epochs)], fontsize=7, rotation=30)
    ax.set_ylabel('Fraction', fontsize=9)
    ax.set_title(f"S{snum} ({info['state'][:3]}/{info['phase'][:3]})", fontsize=10)
    ax.set_ylim(0, 1)
    if idx == 0:
        ax.legend(fontsize=6, loc='upper right', ncol=3)

fig.suptitle('Zone Occupancy Evolution (5 equal epochs per session)', fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('figures/behav_temporal_evolution.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved figures/behav_temporal_evolution.png")


# =============================================================================
# 5. PRE-FEEDING AND PRE-DIGGING ZONE SEQUENCES
# =============================================================================
print("\n\n" + "=" * 70)
print("5. PRE-EVENT ZONE SEQUENCES")
print("=" * 70)

N_BOUTS_BEFORE = 8  # look at last N bouts before event

for event_name, onset_key in [('Feeding', 'feeding_onsets'), ('Digging', 'digging_onsets')]:
    print(f"\n--- Pre-{event_name} sequences (last {N_BOUTS_BEFORE} zone bouts) ---")

    all_sequences = []

    for snum in sorted(all_data.keys()):
        info = all_data[snum]
        onsets = info[onset_key]
        bouts = info['bouts']

        if len(onsets) == 0:
            continue

        for onset_time in onsets:
            # Find bouts leading up to this event
            pre_bouts = [b for b in bouts if b['end_time'] < onset_time]
            if len(pre_bouts) < N_BOUTS_BEFORE:
                continue

            seq = [b['zone'] for b in pre_bouts[-N_BOUTS_BEFORE:]]
            # Also get the zone at onset
            onset_idx = np.searchsorted(info['time'], onset_time)
            onset_idx = min(onset_idx, len(info['zones_simple']) - 1)
            onset_zone = info['zones_simple'][onset_idx]

            all_sequences.append({
                'session': snum,
                'state': info['state'],
                'phase': info['phase'],
                'onset_time': onset_time,
                'sequence': seq,
                'onset_zone': onset_zone,
            })

    if not all_sequences:
        print(f"  No {event_name} events with sufficient preceding bouts")
        continue

    print(f"  {len(all_sequences)} events across sessions")

    # Zone frequency at each position before event
    pos_counts = defaultdict(Counter)
    for seq_info in all_sequences:
        for pos, z in enumerate(seq_info['sequence']):
            pos_counts[pos][z] += 1
        pos_counts['onset'][seq_info['onset_zone']] += 1

    print(f"\n  Most common zone at each position before {event_name}:")
    for pos in range(N_BOUTS_BEFORE):
        label = f"  bout-{N_BOUTS_BEFORE - pos}"
        top = pos_counts[pos].most_common(3)
        total = sum(pos_counts[pos].values())
        top_str = ', '.join([f"{z}({c}/{total}={100*c/total:.0f}%)" for z, c in top])
        print(f"    {label}: {top_str}")

    top_onset = pos_counts['onset'].most_common(3)
    total = sum(pos_counts['onset'].values())
    top_str = ', '.join([f"{z}({c}/{total}={100*c/total:.0f}%)" for z, c in top_onset])
    print(f"    onset:  {top_str}")

    # Common 3-bout patterns right before event
    pre3 = Counter()
    for seq_info in all_sequences:
        pre3[tuple(seq_info['sequence'][-3:])] += 1

    print(f"\n  Top 10 last-3-bout patterns before {event_name}:")
    for gram, count in pre3.most_common(10):
        pct = 100 * count / len(all_sequences)
        print(f"    {' -> '.join(gram)}: {count} ({pct:.1f}%)")


# =============================================================================
# 6. RETURN TIME DISTRIBUTIONS
# =============================================================================
print("\n\n" + "=" * 70)
print("6. RETURN TIME DISTRIBUTIONS")
print("=" * 70)

return_times = defaultdict(lambda: defaultdict(list))

for snum in sorted(all_data.keys()):
    info = all_data[snum]
    bouts = info['bouts']
    condition = f"{info['state']}_{info['phase']}"

    # For each zone, find time between consecutive visits
    last_visit_end = {}
    for bout in bouts:
        z = bout['zone']
        if z in last_visit_end:
            gap = bout['start_time'] - last_visit_end[z]
            if gap > 0:
                return_times[z][condition].append(gap)
        last_visit_end[z] = bout['end_time']

print("\nMedian return time (seconds between consecutive visits):")
print(f"  {'Zone':<6}", end='')
for cond in conditions:
    print(f"  {cond_labels[cond]:>10}", end='')
print()
for zone in key_zones:
    print(f"  {zone:<6}", end='')
    for cond in conditions:
        rts = return_times[zone].get(cond, [])
        if rts:
            print(f"  {np.median(rts):>9.1f}s", end='')
        else:
            print(f"  {'—':>10}", end='')
    print()


# =============================================================================
# 7. BEHAVIORAL ENTROPY OVER TIME
# =============================================================================
print("\n\n" + "=" * 70)
print("7. BEHAVIORAL ENTROPY OVER TIME")
print("=" * 70)

fig, axes = plt.subplots(2, 4, figsize=(28, 10), sharey=True)
state_colors_map = {'fed': '#1f77b4', 'fasted': '#d62728'}

for idx, snum in enumerate(sorted(all_data.keys())):
    info = all_data[snum]
    centers, entropies = compute_behavioral_entropy(info['zones_simple'], info['time'])
    smooth_ent = gaussian_filter1d(entropies, 3)

    row, col = idx // 4, idx % 4
    ax = axes[row, col]
    color = state_colors_map[info['state']]
    ax.plot(centers, smooth_ent, color=color, linewidth=1.2)
    ax.fill_between(centers, smooth_ent, alpha=0.2, color=color)

    # Mark feeding and digging
    for ft in info['feeding_onsets']:
        ax.axvline(ft, color='orange', linewidth=0.5, alpha=0.5)
    for dt in info['digging_onsets']:
        ax.axvline(dt, color='purple', linewidth=0.3, alpha=0.3)

    ax.set_title(f"S{snum} ({info['state'][:3]}/{info['phase'][:3]})", fontsize=10)
    ax.set_xlabel('Time (s)', fontsize=9)
    if col == 0:
        ax.set_ylabel('Entropy (bits)', fontsize=10)
    ax.tick_params(labelsize=8)

fig.suptitle('Behavioral Entropy Over Time (60s window, 10s step)\n'
             'Orange=Feeding, Purple=Digging',
             fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig('figures/behav_entropy_over_time.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved figures/behav_entropy_over_time.png")

# Mean entropy by condition
print("\nMean behavioral entropy by condition:")
for cond_label, sessions in [('Fed/Exp', [1, 3]), ('Fed/For', [2, 4]),
                               ('Fasted/Exp', [5, 7]), ('Fasted/For', [6, 8])]:
    ents = []
    for snum in sessions:
        if snum in all_data:
            _, e = compute_behavioral_entropy(all_data[snum]['zones_simple'], all_data[snum]['time'])
            ents.append(np.mean(e))
    if ents:
        print(f"  {cond_label}: {np.mean(ents):.3f} ± {np.std(ents):.3f}")


# =============================================================================
# 8. CROSS-SESSION CONSISTENCY OF BOUT ORDER
# =============================================================================
print("\n\n" + "=" * 70)
print("8. CONSISTENT SEQUENTIAL PATTERNS")
print("=" * 70)

# Find n-grams that appear in ALL exploration sessions or ALL foraging sessions
for n in [3, 4]:
    print(f"\n--- {n}-grams present in ALL sessions of a type ---")

    exp_per_session = {}
    for_per_session = {}

    for snum in sorted(all_data.keys()):
        info = all_data[snum]
        ngrams = set(get_ngrams(info['bouts'], n))
        if info['phase'] == 'exploration':
            exp_per_session[snum] = ngrams
        else:
            for_per_session[snum] = ngrams

    # Intersection across all exploration sessions
    if exp_per_session:
        common_exp = set.intersection(*exp_per_session.values())
        # Filter to non-trivial (not just H-L-H or T-L-T loops)
        trivial = set()
        for g in common_exp:
            if len(set(g)) <= 2 and all(z in ('H', 'L', 'T') for z in g):
                trivial.add(g)
        interesting_exp = common_exp - trivial

        print(f"\n  Common to ALL exploration sessions ({len(common_exp)} total, "
              f"{len(interesting_exp)} non-trivial):")
        for gram in sorted(interesting_exp, key=lambda g: ' '.join(g)):
            counts = [ngram_counts_by_session[s].get(gram, 0)
                      for s in sorted(exp_per_session.keys())
                      if s in ngram_counts_by_session]
            print(f"    {' -> '.join(gram)}: counts={counts}")

    if for_per_session:
        common_for = set.intersection(*for_per_session.values())
        trivial = set()
        for g in common_for:
            if len(set(g)) <= 2 and all(z in ('H', 'L', 'T') for z in g):
                trivial.add(g)
        interesting_for = common_for - trivial

        print(f"\n  Common to ALL foraging sessions ({len(common_for)} total, "
              f"{len(interesting_for)} non-trivial):")
        for gram in sorted(interesting_for, key=lambda g: ' '.join(g)):
            counts = [ngram_counts_by_session[s].get(gram, 0)
                      for s in sorted(for_per_session.keys())
                      if s in ngram_counts_by_session]
            print(f"    {' -> '.join(gram)}: counts={counts}")

    # N-grams in ALL 8 sessions
    all_per_session = {s: set(get_ngrams(all_data[s]['bouts'], n)) for s in sorted(all_data.keys())}
    if all_per_session:
        common_all = set.intersection(*all_per_session.values())
        trivial = set()
        for g in common_all:
            if len(set(g)) <= 2 and all(z in ('H', 'L', 'T') for z in g):
                trivial.add(g)
        interesting_all = common_all - trivial

        print(f"\n  Common to ALL 8 sessions ({len(common_all)} total, "
              f"{len(interesting_all)} non-trivial):")
        for gram in sorted(interesting_all, key=lambda g: ' '.join(g)):
            counts = [ngram_counts_by_session[s].get(gram, 0)
                      for s in sorted(all_data.keys())
                      if s in ngram_counts_by_session]
            print(f"    {' -> '.join(gram)}: counts={counts}")


# =============================================================================
# SAVE DATA
# =============================================================================
print("\n\n" + "=" * 70)
print("SAVING DATA")
print("=" * 70)

# Save bout data
all_bout_rows = []
for snum in sorted(all_data.keys()):
    info = all_data[snum]
    for bout in info['bouts']:
        bout_row = bout.copy()
        bout_row['session'] = snum
        bout_row['state'] = info['state']
        bout_row['phase'] = info['phase']
        all_bout_rows.append(bout_row)

bout_df = pd.DataFrame(all_bout_rows)
bout_df.to_csv('data/behavioral_bouts_all_sessions.csv', index=False)
print(f"Saved data/behavioral_bouts_all_sessions.csv ({len(bout_df)} bouts)")

# Save transition similarity
sim_df = pd.DataFrame(sim_matrix, index=[f'S{s}' for s in snums],
                       columns=[f'S{s}' for s in snums])
sim_df.to_csv('data/behavioral_transition_similarity.csv')
print("Saved data/behavioral_transition_similarity.csv")

print("\n[DONE]")
