"""
Dual-Probe: Digging Bout Cluster Transitions
=============================================
Identifies digging bout clusters — sequences of consecutive digs where the
mouse moves directly between pots WITHOUT returning to Home or Ladder zones
during the inter-bout gap.

Within each cluster, analyzes:
  - Whether transitions are to the same pot or a different pot
  - Transition diversity per cluster
  - Pot-to-pot transition matrices
  - Comparison across Fed / Fasted / HFD states
"""

import yaml
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu, chi2_contingency
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

with open("paths.yaml") as f:
    cfg = yaml.safe_load(f)

MIN_DIG_DURATION = 2.0
MIN_INTER_DIG = 10.0
SKIP_SESSIONS = {23, 24}

# Zones that BREAK a cluster (home/ladder excursion)
BREAK_ZONES = {'H', 'HCL', 'HCR', 'L'}

sessions_cfg = cfg["double_probe"]["coordinates_1"]["mouse01"]["sessions"]

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
    vel = pd.to_numeric(data['Velocity(Center-point)'], errors='coerce').values
    vel = np.nan_to_num(vel, nan=0.0)

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

    return time_vals, vel, zones, dig_vals


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


def get_pot_at_dig(zones, time_vals, dig_start):
    idx = np.searchsorted(time_vals, dig_start)
    dt = np.median(np.diff(time_vals))
    window = int(2.0 / dt)
    start = max(0, idx - window)
    end = min(len(zones), idx + window)
    segment = zones[start:end]
    pot_zones = [z for z in segment if z.startswith('P') and not z.endswith('z')]
    if pot_zones:
        return Counter(pot_zones).most_common(1)[0][0]
    return 'unknown'


def gap_has_home_excursion(zones, time_vals, gap_start, gap_end):
    """Check if the mouse enters Home or Ladder zones during a gap."""
    mask = (time_vals >= gap_start) & (time_vals <= gap_end)
    gap_zones = zones[mask]
    return any(z in BREAK_ZONES for z in gap_zones)


# ========================================================================
# Discover sessions
# ========================================================================
session_meta = {}
for skey, sval in sessions_cfg.items():
    snum = int(skey.split('_')[1])
    if snum in SKIP_SESSIONS:
        continue
    behav = sval.get('behavior')
    if not behav or not Path(behav).exists():
        continue
    session_meta[snum] = {
        'state': sval['state'], 'phase': sval['phase'],
        'behavior': behav,
    }

print(f"Found {len(session_meta)} sessions")

# ========================================================================
# EXTRACT CLUSTERS AND TRANSITIONS
# ========================================================================
all_clusters = []       # each cluster is a list of bouts with pots
all_transitions = []    # each transition: from_pot, to_pot, same/diff, session, state
session_summaries = []  # per-session summary stats

print("\n" + "=" * 100)
print("DIGGING BOUT CLUSTERS & POT-TO-POT TRANSITIONS")
print(f"Cluster break: gap contains Home (H/HCL/HCR) or Ladder (L) zone visit")
print("=" * 100)

for snum in sorted(session_meta.keys()):
    meta = session_meta[snum]
    state, phase = meta['state'], meta['phase']

    time_vals, vel, zones, dig_vals = load_behavior_xlsx(meta['behavior'])

    if np.sum(dig_vals > 0.5) == 0:
        continue

    bouts = extract_dig_bouts(dig_vals, time_vals, MIN_DIG_DURATION, MIN_INTER_DIG)
    if len(bouts) < 1:
        continue

    # Label each bout with its pot
    for b in bouts:
        b['pot'] = get_pot_at_dig(zones, time_vals, b['start_time'])

    # Build clusters: consecutive bouts where gap has no home/ladder visit
    clusters = [[bouts[0]]]
    for i in range(1, len(bouts)):
        gap_start = bouts[i-1]['end_time']
        gap_end = bouts[i]['start_time']

        if gap_has_home_excursion(zones, time_vals, gap_start, gap_end):
            clusters.append([bouts[i]])  # start new cluster
        else:
            clusters[-1].append(bouts[i])  # extend current cluster

    # Analyze transitions within clusters
    n_same = 0
    n_diff = 0
    n_total_transitions = 0
    cluster_sizes = []
    cluster_unique_pots = []

    print(f"\n  S{snum} ({state}/{phase}): {len(bouts)} bouts, {len(clusters)} clusters")

    for ci, cluster in enumerate(clusters):
        pots = [b['pot'] for b in cluster]
        cluster_sizes.append(len(cluster))
        unique = len(set(p for p in pots if p != 'unknown'))
        cluster_unique_pots.append(unique)

        cluster_info = {
            'session': snum, 'state': state, 'phase': phase,
            'cluster_idx': ci, 'cluster_size': len(cluster),
            'unique_pots': unique, 'pots_sequence': '>'.join(pots),
            'total_duration': cluster[-1]['end_time'] - cluster[0]['start_time'],
        }
        all_clusters.append(cluster_info)

        # Transitions within this cluster
        for j in range(len(cluster) - 1):
            from_pot = cluster[j]['pot']
            to_pot = cluster[j+1]['pot']
            is_same = from_pot == to_pot
            gap_dur = cluster[j+1]['start_time'] - cluster[j]['end_time']

            if is_same:
                n_same += 1
            else:
                n_diff += 1
            n_total_transitions += 1

            all_transitions.append({
                'session': snum, 'state': state, 'phase': phase,
                'from_pot': from_pot, 'to_pot': to_pot,
                'same_pot': is_same, 'gap_duration': gap_dur,
                'cluster_idx': ci, 'cluster_size': len(cluster),
            })

        if len(cluster) > 1:
            pot_seq = '>'.join(pots)
            print(f"    Cluster {ci+1}: {pot_seq} ({len(cluster)} bouts, "
                  f"{unique} unique pots)")

    # Session summary
    multi_bout_clusters = [c for c in clusters if len(c) > 1]
    if n_total_transitions > 0:
        same_frac = n_same / n_total_transitions
        diff_frac = n_diff / n_total_transitions
    else:
        same_frac = np.nan
        diff_frac = np.nan

    summary = {
        'session': snum, 'state': state, 'phase': phase,
        'n_bouts': len(bouts),
        'n_clusters': len(clusters),
        'n_multi_bout_clusters': len(multi_bout_clusters),
        'mean_cluster_size': np.mean(cluster_sizes),
        'max_cluster_size': np.max(cluster_sizes),
        'n_transitions': n_total_transitions,
        'n_same_pot': n_same,
        'n_diff_pot': n_diff,
        'frac_same_pot': same_frac,
        'frac_diff_pot': diff_frac,
        'mean_unique_pots_per_cluster': np.mean(cluster_unique_pots),
    }
    session_summaries.append(summary)

    print(f"    Summary: {n_total_transitions} transitions — "
          f"{n_same} same pot ({same_frac:.1%}), {n_diff} diff pot ({diff_frac:.1%})")

# ========================================================================
# SAVE DATA
# ========================================================================
df_clusters = pd.DataFrame(all_clusters)
df_transitions = pd.DataFrame(all_transitions)
df_summary = pd.DataFrame(session_summaries)

df_clusters.to_csv('data/dp_digging_clusters.csv', index=False)
df_transitions.to_csv('data/dp_digging_transitions.csv', index=False)
df_summary.to_csv('data/dp_digging_transition_summary.csv', index=False)
print(f"\nSaved {len(df_clusters)} clusters, {len(df_transitions)} transitions, "
      f"{len(df_summary)} session summaries")

# ========================================================================
# STATE COMPARISON
# ========================================================================
print("\n" + "=" * 100)
print("STATE COMPARISON")
print("=" * 100)

states = ['fed', 'fasted', 'fed-HFD']
state_labels = {'fed': 'Fed', 'fasted': 'Fasted', 'fed-HFD': 'HFD'}

compare_metrics = [
    ('n_clusters', 'Clusters per session'),
    ('n_multi_bout_clusters', 'Multi-bout clusters'),
    ('mean_cluster_size', 'Mean cluster size'),
    ('max_cluster_size', 'Max cluster size'),
    ('n_transitions', 'Transitions per session'),
    ('frac_same_pot', 'Fraction same-pot transitions'),
    ('frac_diff_pot', 'Fraction diff-pot transitions'),
    ('mean_unique_pots_per_cluster', 'Mean unique pots per cluster'),
]

for metric, label in compare_metrics:
    groups = {}
    for st in states:
        vals = df_summary.loc[df_summary['state'] == st, metric].dropna().values
        if len(vals) > 0:
            groups[st] = vals

    if len(groups) < 2:
        continue

    means_str = ', '.join([f"{state_labels[s]}={np.mean(v):.3f} (n={len(v)})"
                           for s, v in groups.items()])
    print(f"\n  {label}: {means_str}")

    # KW test
    if len(groups) == 3:
        vals_list = [groups[s] for s in states if s in groups]
        if all(len(v) >= 2 for v in vals_list):
            try:
                _, kw_p = kruskal(*vals_list)
                print(f"    KW p={kw_p:.4f}{'*' if kw_p < 0.05 else ''}")
            except Exception:
                pass

    # Pairwise
    for i in range(len(states)):
        for j in range(i+1, len(states)):
            s1, s2 = states[i], states[j]
            if s1 in groups and s2 in groups and len(groups[s1]) >= 2 and len(groups[s2]) >= 2:
                try:
                    _, p = mannwhitneyu(groups[s1], groups[s2], alternative='two-sided')
                    print(f"    {state_labels[s1]} vs {state_labels[s2]}: p={p:.4f}{'*' if p < 0.05 else ''}")
                except Exception:
                    pass

# ========================================================================
# POOLED TRANSITION ANALYSIS
# ========================================================================
print("\n" + "=" * 100)
print("POOLED TRANSITION COUNTS")
print("=" * 100)

if len(df_transitions) > 0:
    for st in states:
        t = df_transitions[df_transitions['state'] == st]
        if len(t) == 0:
            continue
        n_same = t['same_pot'].sum()
        n_diff = (~t['same_pot']).sum()
        total = len(t)
        print(f"\n  {state_labels[st]}: {total} transitions — "
              f"{n_same} same ({n_same/total:.1%}), {n_diff} diff ({n_diff/total:.1%})")

        # Gap duration: same vs diff
        same_gaps = t.loc[t['same_pot'], 'gap_duration'].values
        diff_gaps = t.loc[~t['same_pot'], 'gap_duration'].values
        if len(same_gaps) > 0:
            print(f"    Same-pot gap: {np.mean(same_gaps):.1f}s (median {np.median(same_gaps):.1f}s)")
        if len(diff_gaps) > 0:
            print(f"    Diff-pot gap: {np.mean(diff_gaps):.1f}s (median {np.median(diff_gaps):.1f}s)")

    # Chi-square: same/diff proportions across states
    print("\n  Chi-square test (same vs diff proportions across states):")
    contingency = []
    state_order = []
    for st in states:
        t = df_transitions[df_transitions['state'] == st]
        if len(t) > 0:
            n_same = t['same_pot'].sum()
            n_diff = (~t['same_pot']).sum()
            contingency.append([n_same, n_diff])
            state_order.append(st)
    if len(contingency) >= 2:
        contingency = np.array(contingency)
        try:
            chi2, chi_p, dof, expected = chi2_contingency(contingency)
            print(f"    chi2={chi2:.2f}, p={chi_p:.4f}{'*' if chi_p < 0.05 else ''}, dof={dof}")
        except Exception as e:
            print(f"    chi2 failed: {e}")

    # Pairwise chi-square
    for i in range(len(state_order)):
        for j in range(i+1, len(state_order)):
            s1, s2 = state_order[i], state_order[j]
            t1 = df_transitions[df_transitions['state'] == s1]
            t2 = df_transitions[df_transitions['state'] == s2]
            ct = np.array([
                [t1['same_pot'].sum(), (~t1['same_pot']).sum()],
                [t2['same_pot'].sum(), (~t2['same_pot']).sum()],
            ])
            try:
                chi2, p, _, _ = chi2_contingency(ct)
                print(f"    {state_labels[s1]} vs {state_labels[s2]}: "
                      f"chi2={chi2:.2f}, p={p:.4f}{'*' if p < 0.05 else ''}")
            except Exception:
                pass

    # Transition matrix per state
    print("\n" + "=" * 100)
    print("POT-TO-POT TRANSITION MATRICES (within clusters)")
    print("=" * 100)

    pot_list = ['P1', 'P2', 'P3', 'P4']
    for st in states:
        t = df_transitions[df_transitions['state'] == st]
        t_known = t[(t['from_pot'].isin(pot_list)) & (t['to_pot'].isin(pot_list))]
        if len(t_known) == 0:
            continue

        print(f"\n  {state_labels[st]} ({len(t_known)} transitions):")
        mat = np.zeros((4, 4), dtype=int)
        for _, row in t_known.iterrows():
            fi = pot_list.index(row['from_pot'])
            ti = pot_list.index(row['to_pot'])
            mat[fi, ti] += 1

        # Print matrix
        print(f"    {'':>6s} > {'P1':>5s} {'P2':>5s} {'P3':>5s} {'P4':>5s}  | total")
        for fi, fp in enumerate(pot_list):
            row_total = mat[fi].sum()
            row_str = ' '.join([f'{mat[fi,ti]:5d}' for ti in range(4)])
            diag_pct = mat[fi, fi] / row_total * 100 if row_total > 0 else 0
            print(f"    {fp:>6s}   {row_str}  | {row_total:3d}  ({diag_pct:.0f}% same)")

        # Diagonal fraction (same-pot)
        diag = np.trace(mat)
        total = mat.sum()
        print(f"    Overall same-pot: {diag}/{total} ({diag/total:.1%})")

# ========================================================================
# FIGURES
# ========================================================================
state_colors = {'fed': '#4e79a7', 'fasted': '#e15759', 'fed-HFD': '#f28e2b'}

fig, axes = plt.subplots(2, 4, figsize=(28, 12))

# Row 1: session-level metrics
plot_metrics = [
    ('n_clusters', 'Clusters per Session'),
    ('mean_cluster_size', 'Mean Cluster Size'),
    ('n_transitions', 'Transitions per Session'),
    ('frac_same_pot', 'Frac. Same-Pot Transitions'),
]

for idx, (metric, label) in enumerate(plot_metrics):
    ax = axes[0, idx]
    for si, st in enumerate(states):
        vals = df_summary.loc[df_summary['state'] == st, metric].dropna().values
        if len(vals) == 0:
            continue
        mean_v = np.mean(vals)
        sem_v = np.std(vals) / np.sqrt(len(vals)) if len(vals) > 1 else 0
        ax.bar(si, mean_v, color=state_colors[st], alpha=0.7, width=0.6,
               edgecolor='black', linewidth=0.5)
        ax.errorbar(si, mean_v, yerr=sem_v, color='black', capsize=5,
                    capthick=1.5, linewidth=1.5)
        jitter = np.random.uniform(-0.15, 0.15, len(vals))
        ax.scatter(si + jitter, vals, color=state_colors[st],
                   edgecolor='black', linewidth=0.5, s=50, zorder=5, alpha=0.8)

    ax.set_xticks(range(len(states)))
    ax.set_xticklabels([state_labels[s] for s in states], fontsize=14)
    ax.set_title(label, fontsize=15, fontweight='bold')
    ax.set_ylabel(label, fontsize=13)
    ax.tick_params(labelsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# Row 2: Transition matrices as heatmaps + stacked bar
pot_list = ['P1', 'P2', 'P3', 'P4']
for si, st in enumerate(states):
    ax = axes[1, si]
    t = df_transitions[df_transitions['state'] == st]
    t_known = t[(t['from_pot'].isin(pot_list)) & (t['to_pot'].isin(pot_list))]

    mat = np.zeros((4, 4))
    for _, row in t_known.iterrows():
        fi = pot_list.index(row['from_pot'])
        ti = pot_list.index(row['to_pot'])
        mat[fi, ti] += 1

    # Normalize rows to fractions
    row_sums = mat.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1)
    mat_norm = mat / row_sums

    im = ax.imshow(mat_norm, cmap='YlOrRd', vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(pot_list, fontsize=13)
    ax.set_yticklabels(pot_list, fontsize=13)
    ax.set_xlabel('To Pot', fontsize=13)
    ax.set_ylabel('From Pot', fontsize=13)

    # Annotate with counts
    for fi in range(4):
        for ti in range(4):
            count = int(mat[fi, ti])
            if count > 0:
                color = 'white' if mat_norm[fi, ti] > 0.5 else 'black'
                ax.text(ti, fi, f'{count}', ha='center', va='center',
                        fontsize=13, fontweight='bold', color=color)

    diag = np.trace(mat)
    total = mat.sum()
    same_pct = diag / total * 100 if total > 0 else 0
    ax.set_title(f'{state_labels[st]}\n{int(total)} trans, {same_pct:.0f}% same-pot',
                 fontsize=14, fontweight='bold')

# Stacked bar: same vs diff by state
ax = axes[1, 3]
for si, st in enumerate(states):
    t = df_transitions[df_transitions['state'] == st]
    if len(t) == 0:
        continue
    n_same = t['same_pot'].sum()
    n_diff = (~t['same_pot']).sum()
    total = len(t)
    ax.bar(si, n_same / total, color=state_colors[st], alpha=0.9, width=0.6,
           label=f'{state_labels[st]} same' if si == 0 else None)
    ax.bar(si, n_diff / total, bottom=n_same / total,
           color=state_colors[st], alpha=0.3, width=0.6, hatch='//')
    ax.text(si, 0.5, f'{n_same}/{total}\n({n_same/total:.0%})',
            ha='center', va='center', fontsize=12, fontweight='bold')

ax.set_xticks(range(len(states)))
ax.set_xticklabels([state_labels[s] for s in states], fontsize=14)
ax.set_ylabel('Fraction', fontsize=13)
ax.set_title('Same-Pot (solid) vs Diff-Pot (hatched)', fontsize=14, fontweight='bold')
ax.set_ylim(0, 1.05)
ax.tick_params(labelsize=12)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig.suptitle('Digging Bout Cluster Transitions: Fed vs Fasted vs HFD',
             fontsize=20, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('figures/dp_digging_transitions.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved figures/dp_digging_transitions.png")

print("\nDone.")
