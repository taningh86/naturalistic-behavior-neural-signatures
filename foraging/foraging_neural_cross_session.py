"""
Foraging neural signatures -- Cross-session comparison.

Reads per-session outputs from foraging_neural_all_sessions.py and produces
cross-session summary figures and fed-vs-fasted comparisons.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats as sp_stats
from scipy.ndimage import uniform_filter1d
import warnings, sys

warnings.filterwarnings('ignore')

SESSIONS = [2, 4, 6, 8]
SESSION_INFO = {
    2: {'state': 'fed', 'label': 'S2 (Fed)', 'color': 'royalblue'},
    4: {'state': 'fed', 'label': 'S4 (Fed)', 'color': 'cornflowerblue'},
    6: {'state': 'fasted', 'label': 'S6 (Fasted)', 'color': 'firebrick'},
    8: {'state': 'fasted', 'label': 'S8 (Fasted)', 'color': 'salmon'},
}


def main():
    print("=" * 70)
    print("  FORAGING NEURAL SIGNATURES -- Cross-Session Comparison")
    print("=" * 70)

    # Load per-session pot visit data
    pv_all = {}
    for s in SESSIONS:
        pv_all[s] = pd.read_csv(f'data/foraging_excursion_potvisits_s{s}.csv')

    # Load metrics
    metrics_df = pd.read_csv('data/foraging_neural_all_sessions_metrics.csv')
    print(f"  Metrics: {len(metrics_df)} sessions loaded")
    print(metrics_df[['session', 'state', 'discovery_time', 'first_dig_time',
                       'n_pre_disc', 'n_p2_pre', 'n_p4_pre']].to_string(index=False))

    # =========================================================================
    # 1. DISCOVERY SPEED & BEHAVIORAL SUMMARY
    # =========================================================================
    print(f"\n{'='*70}")
    print("  1. Discovery Speed & Behavioral Summary")
    print(f"{'='*70}")

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle('Cross-Session Behavioral Summary', fontsize=14, fontweight='bold')

    # Discovery time
    ax = axes[0]
    for s in SESSIONS:
        info = SESSION_INFO[s]
        row = metrics_df[metrics_df['session'] == s].iloc[0]
        ax.bar(info['label'], row['discovery_time'], color=info['color'],
               edgecolor='black')
    ax.set_ylabel('Time (s)')
    ax.set_title('Time to Discovery (First Feed)')

    # First dig
    ax = axes[1]
    for s in SESSIONS:
        info = SESSION_INFO[s]
        row = metrics_df[metrics_df['session'] == s].iloc[0]
        ax.bar(info['label'], row['first_dig_time'], color=info['color'],
               edgecolor='black')
    ax.set_ylabel('Time (s)')
    ax.set_title('Time to First Dig')

    # Pre-discovery Pot-2 visits
    ax = axes[2]
    for s in SESSIONS:
        info = SESSION_INFO[s]
        row = metrics_df[metrics_df['session'] == s].iloc[0]
        ax.bar(info['label'], row['n_p2_pre'], color=info['color'],
               edgecolor='black')
    ax.set_ylabel('Count')
    ax.set_title('Pre-Discovery Pot-2 Visits')

    # Pre-discovery Pot-4 visits
    ax = axes[3]
    for s in SESSIONS:
        info = SESSION_INFO[s]
        row = metrics_df[metrics_df['session'] == s].iloc[0]
        ax.bar(info['label'], row['n_p4_pre'], color=info['color'],
               edgecolor='black')
    ax.set_ylabel('Count')
    ax.set_title('Pre-Discovery Pot-4 Visits')

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig('figures/foraging_cross_session_behavioral.png',
                dpi=100, bbox_inches='tight')
    plt.close()
    print("  Saved: figures/foraging_cross_session_behavioral.png")

    # =========================================================================
    # 2. REWARD ONSET COMPARISON (Fed vs Fasted)
    # =========================================================================
    print(f"\n{'='*70}")
    print("  2. Reward Onset Comparison (Fed vs Fasted)")
    print(f"{'='*70}")

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Reward Onset: Fed vs Fasted\n'
                 'Peri-event FR and PC1 around first feeding',
                 fontsize=14, fontweight='bold')

    for ri, region in enumerate(['LHA', 'RSP']):
        # FR
        ax = axes[ri, 0]
        for s in SESSIONS:
            try:
                data = np.load(f'data/foraging_reward_traces_s{s}.npz',
                               allow_pickle=True)
                t_key = f'{region}_reward_fr_t'
                fr_key = f'{region}_reward_fr'
                if t_key in data and fr_key in data:
                    t = data[t_key]
                    fr = data[fr_key]
                    if len(fr) > 20:
                        fr_sm = uniform_filter1d(fr, 20)
                    else:
                        fr_sm = fr
                    info = SESSION_INFO[s]
                    ax.plot(t, fr_sm, color=info['color'], linewidth=2,
                            label=info['label'], alpha=0.8)
            except Exception as e:
                print(f"  Warning: Could not load reward traces for S{s}: {e}")

        ax.axvline(0, color='green', linewidth=2, linestyle='--', alpha=0.5)
        ax.set_xlabel('Time from first feed (s)')
        ax.set_ylabel(f'{region} Pop FR (Hz)')
        ax.set_title(f'{region} -- Firing Rate')
        ax.legend(fontsize=8)

        # PC1
        ax = axes[ri, 1]
        for s in SESSIONS:
            try:
                data = np.load(f'data/foraging_reward_traces_s{s}.npz',
                               allow_pickle=True)
                t_key = f'{region}_reward_pc1_t'
                pc1_key = f'{region}_reward_pc1'
                if t_key in data and pc1_key in data:
                    t = data[t_key]
                    pc1 = data[pc1_key]
                    # Subsample for readability
                    ss = max(1, len(t) // 500)
                    if len(pc1) > 30:
                        pc1_sm = uniform_filter1d(pc1, 30)
                    else:
                        pc1_sm = pc1
                    info = SESSION_INFO[s]
                    ax.plot(t[::ss], pc1_sm[::ss], color=info['color'],
                            linewidth=2, label=info['label'], alpha=0.8)
            except Exception as e:
                pass

        ax.axvline(0, color='green', linewidth=2, linestyle='--', alpha=0.5)
        ax.set_xlabel('Time from first feed (s)')
        ax.set_ylabel(f'{region} PC1')
        ax.set_title(f'{region} -- Latent PC1')
        ax.legend(fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.91])
    plt.savefig('figures/foraging_cross_session_reward.png',
                dpi=100, bbox_inches='tight')
    plt.close()
    print("  Saved: figures/foraging_cross_session_reward.png")

    # =========================================================================
    # 3. POST-DISCOVERY POT VISIT COMPARISON (Fed vs Fasted)
    # =========================================================================
    print(f"\n{'='*70}")
    print("  3. Post-Discovery Pot Visit Patterns")
    print(f"{'='*70}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Post-Discovery Pot Visit Patterns\n'
                 'All non-feeding visits after first feed',
                 fontsize=14, fontweight='bold')

    for pi, pot in enumerate(['Pot-2', 'Pot-4']):
        # Visit count over time (cumulative)
        ax = axes[0, pi]
        for s in SESSIONS:
            pv = pv_all[s]
            disc_t = metrics_df[metrics_df['session'] == s].iloc[0]['discovery_time']
            post = pv[(pv['pre_discovery'] == False) & (pv['pot'] == pot) &
                       (pv['feed_bins'] == 0)].sort_values('start_s')
            if len(post) > 0:
                times_rel = post['start_s'].values - disc_t
                cumcount = np.arange(1, len(post) + 1)
                info = SESSION_INFO[s]
                ax.step(times_rel, cumcount, color=info['color'], linewidth=2,
                        label=info['label'], where='post')
        ax.set_xlabel('Time after discovery (s)')
        ax.set_ylabel('Cumulative visits')
        ax.set_title(f'Post-discovery {pot} visits')
        ax.legend(fontsize=8)

        # Dwell time distribution
        ax = axes[1, pi]
        fed_dwells = []
        fasted_dwells = []
        for s in SESSIONS:
            pv = pv_all[s]
            post = pv[(pv['pre_discovery'] == False) & (pv['pot'] == pot) &
                       (pv['feed_bins'] == 0)]
            dwells = post['dwell_s'].values
            if SESSION_INFO[s]['state'] == 'fed':
                fed_dwells.extend(dwells)
            else:
                fasted_dwells.extend(dwells)

        data_to_plot = []
        labels = []
        if len(fed_dwells) > 0:
            data_to_plot.append(fed_dwells)
            labels.append(f'Fed (n={len(fed_dwells)})')
        if len(fasted_dwells) > 0:
            data_to_plot.append(fasted_dwells)
            labels.append(f'Fasted (n={len(fasted_dwells)})')

        if len(data_to_plot) > 0:
            bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
            colors = ['royalblue', 'firebrick']
            for patch, color in zip(bp['boxes'], colors[:len(data_to_plot)]):
                patch.set_facecolor(color)
                patch.set_alpha(0.5)

        if len(fed_dwells) > 1 and len(fasted_dwells) > 1:
            u, p = sp_stats.mannwhitneyu(fed_dwells, fasted_dwells,
                                         alternative='two-sided')
            ax.set_title(f'{pot} dwell time (p={p:.4f})')
        else:
            ax.set_title(f'{pot} dwell time')
        ax.set_ylabel('Dwell time (s)')

    plt.tight_layout(rect=[0, 0, 1, 0.91])
    plt.savefig('figures/foraging_cross_session_post_discovery.png',
                dpi=100, bbox_inches='tight')
    plt.close()
    print("  Saved: figures/foraging_cross_session_post_discovery.png")

    # =========================================================================
    # 4. TRANSITION DYNAMICS SUMMARY (Fed vs Fasted)
    # =========================================================================
    print(f"\n{'='*70}")
    print("  4. P2<->P4 Transition Summary")
    print(f"{'='*70}")

    # Collect all transition arrival metrics from console output
    # We'll re-compute from pot visit data
    transition_rows = []
    for s in SESSIONS:
        pv = pv_all[s]
        disc_t = metrics_df[metrics_df['session'] == s].iloc[0]['discovery_time']
        for exc_idx, grp in pv.groupby('excursion_idx'):
            pots = set(grp['pot'].values)
            if 'Pot-2' not in pots or 'Pot-4' not in pots:
                continue
            visits = list(grp.sort_values('start_s').itertuples())
            for i in range(len(visits) - 1):
                for j in range(i + 1, len(visits)):
                    v1, v2 = visits[i], visits[j]
                    if (v1.pot == 'Pot-2' and v2.pot == 'Pot-4') or \
                       (v1.pot == 'Pot-4' and v2.pot == 'Pot-2'):
                        direction = 'P2->P4' if v1.pot == 'Pot-2' else 'P4->P2'
                        transition_rows.append({
                            'session': s,
                            'state': SESSION_INFO[s]['state'],
                            'exc_idx': exc_idx,
                            'direction': direction,
                            'pre_discovery': v1.start_s < disc_t,
                            't1': v1.start_s, 't2': v2.start_s,
                            'gap_s': v2.start_s - v1.end_s,
                        })
                        break

    trans_df = pd.DataFrame(transition_rows)
    if len(trans_df) > 0:
        trans_df = trans_df.drop_duplicates(
            subset=['session', 'exc_idx', 'direction']).reset_index(drop=True)

    print(f"  Total transitions: {len(trans_df)}")
    for s in SESSIONS:
        st = trans_df[trans_df['session'] == s]
        n_pre = len(st[st['pre_discovery']])
        n_post = len(st[~st['pre_discovery']])
        print(f"    S{s}: {len(st)} transitions ({n_pre} pre, {n_post} post)")

    # Transition summary figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('P2<->P4 Transitions: Cross-Session Summary',
                 fontsize=14, fontweight='bold')

    # Count by session
    ax = axes[0]
    for s in SESSIONS:
        info = SESSION_INFO[s]
        st = trans_df[trans_df['session'] == s]
        n_pre = len(st[st['pre_discovery']])
        n_post = len(st[~st['pre_discovery']])
        ax.bar(info['label'], n_pre, color=info['color'], edgecolor='black',
               label='Pre' if s == 2 else '')
        ax.bar(info['label'], n_post, bottom=n_pre, color=info['color'],
               edgecolor='black', alpha=0.4, label='Post' if s == 2 else '')
    ax.set_ylabel('Count')
    ax.set_title('Transition Count')
    ax.legend(fontsize=8)

    # Direction split
    ax = axes[1]
    for di, direction in enumerate(['P2->P4', 'P4->P2']):
        counts = []
        labels = []
        for s in SESSIONS:
            st = trans_df[(trans_df['session'] == s) &
                          (trans_df['direction'] == direction)]
            counts.append(len(st))
            labels.append(SESSION_INFO[s]['label'])
        x = np.arange(len(SESSIONS))
        w = 0.35
        offset = -w/2 if di == 0 else w/2
        colors = [SESSION_INFO[s]['color'] for s in SESSIONS]
        ax.bar(x + offset, counts, w, color=colors if di == 0 else colors,
               edgecolor='black', alpha=1.0 if di == 0 else 0.5,
               label=direction)
    ax.set_xticks(x)
    ax.set_xticklabels([SESSION_INFO[s]['label'] for s in SESSIONS])
    ax.set_ylabel('Count')
    ax.set_title('Transitions by Direction')
    ax.legend(fontsize=8)

    # Gap time distribution
    ax = axes[2]
    fed_gaps = trans_df[trans_df['state'] == 'fed']['gap_s'].values
    fasted_gaps = trans_df[trans_df['state'] == 'fasted']['gap_s'].values
    data = []
    gap_labels = []
    if len(fed_gaps) > 0:
        data.append(fed_gaps)
        gap_labels.append(f'Fed (n={len(fed_gaps)})')
    if len(fasted_gaps) > 0:
        data.append(fasted_gaps)
        gap_labels.append(f'Fasted (n={len(fasted_gaps)})')
    if len(data) > 0:
        bp = ax.boxplot(data, labels=gap_labels, patch_artist=True)
        colors = ['royalblue', 'firebrick']
        for patch, color in zip(bp['boxes'], colors[:len(data)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)
    ax.set_ylabel('Gap time (s)')
    ax.set_title('Time Between Pots in Transition')

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig('figures/foraging_cross_session_transitions.png',
                dpi=100, bbox_inches='tight')
    plt.close()
    print("  Saved: figures/foraging_cross_session_transitions.png")

    # =========================================================================
    # 5. LEARNING CURVE COMPARISON (Fed sessions only: S2 vs S4)
    # =========================================================================
    print(f"\n{'='*70}")
    print("  5. Learning Curve Comparison (Fed Sessions)")
    print(f"{'='*70}")

    # Note: We can only compare S2 and S4 learning curves since S6/S8 have
    # insufficient pre-discovery data. The actual FR/PC1 values are session-
    # specific (different unit counts, PCA fits), so we normalize within session.

    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    fig.suptitle('Fed Sessions Learning Curves (Pre-Discovery)\n'
                 'Normalized visit number (0-1) | z-scored within session',
                 fontsize=14, fontweight='bold')

    for pi, pot in enumerate(['Pot-2', 'Pot-4']):
        for ri, region in enumerate(['LHA', 'RSP']):
            col = pi * 2 + ri

            for s in [2, 4]:
                pv = pv_all[s]
                pre = pv[(pv['pre_discovery'] == True) &
                          (pv['pot'] == pot)].sort_values('start_s')
                if len(pre) < 3:
                    continue

                # Use visit time as proxy (we don't have FR/PC1 stored in CSV)
                # Instead, normalize visit number
                visit_nums = np.linspace(0, 1, len(pre))
                visit_times = pre['start_s'].values
                dwell_times = pre['dwell_s'].values

                info = SESSION_INFO[s]

                # Visit timing (proxy for engagement)
                ax = axes[0, col]
                ax.plot(visit_nums, visit_times, color=info['color'],
                        linewidth=2, marker='o', markersize=4,
                        label=info['label'])
                ax.set_ylabel('Visit time (s)')
                ax.set_title(f'{pot} -- {region}')
                ax.legend(fontsize=7)

                # Dwell duration
                ax = axes[1, col]
                ax.plot(visit_nums, dwell_times, color=info['color'],
                        linewidth=2, marker='o', markersize=4,
                        label=info['label'])
                ax.set_xlabel('Normalized visit #')
                ax.set_ylabel('Dwell time (s)')

    plt.tight_layout(rect=[0, 0, 1, 0.91])
    plt.savefig('figures/foraging_cross_session_learning_fed.png',
                dpi=100, bbox_inches='tight')
    plt.close()
    print("  Saved: figures/foraging_cross_session_learning_fed.png")

    # =========================================================================
    # 6. COMPREHENSIVE SUMMARY TABLE
    # =========================================================================
    print(f"\n{'='*70}")
    print("  6. Comprehensive Summary")
    print(f"{'='*70}")

    summary_rows = []
    for s in SESSIONS:
        row = metrics_df[metrics_df['session'] == s].iloc[0]
        pv = pv_all[s]
        n_trans = len(trans_df[trans_df['session'] == s]) if len(trans_df) > 0 else 0

        # Post-discovery return to P4 time
        disc_t = row['discovery_time']
        post_p4 = pv[(pv['pre_discovery'] == False) & (pv['pot'] == 'Pot-4') &
                      (pv['feed_bins'] == 0)].sort_values('start_s')
        first_return_p4 = post_p4.iloc[0]['start_s'] - disc_t if len(post_p4) > 0 else np.nan

        summary_rows.append({
            'Session': f'S{s}',
            'State': SESSION_INFO[s]['state'].capitalize(),
            'Discovery (s)': f"{row['discovery_time']:.1f}",
            'First Dig (s)': f"{row['first_dig_time']:.1f}",
            'Pre-disc visits': int(row['n_pre_disc']),
            'Pre P2': int(row['n_p2_pre']),
            'Pre P4': int(row['n_p4_pre']),
            'Post P2': int(row['n_p2_post']),
            'Post P4': int(row['n_p4_post']),
            'P2-P4 transitions': n_trans,
            'First return to P4 (s)': f"{first_return_p4:.1f}" if not np.isnan(first_return_p4) else 'N/A',
        })

    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))

    # Save as figure
    fig, ax = plt.subplots(figsize=(16, 3))
    ax.axis('off')
    table = ax.table(cellText=summary_df.values,
                     colLabels=summary_df.columns,
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.auto_set_column_width(list(range(len(summary_df.columns))))

    # Color rows by state
    for i in range(len(summary_df)):
        state = summary_rows[i]['State']
        color = '#D6EAF8' if state == 'Fed' else '#FADBD8'
        for j in range(len(summary_df.columns)):
            table[i + 1, j].set_facecolor(color)

    plt.title('Cross-Session Summary', fontsize=14, fontweight='bold', pad=20)
    plt.savefig('figures/foraging_cross_session_summary.png',
                dpi=100, bbox_inches='tight')
    plt.close()
    print("  Saved: figures/foraging_cross_session_summary.png")

    # Save summary CSV
    summary_df.to_csv('data/foraging_cross_session_summary.csv', index=False)
    print("  Saved: data/foraging_cross_session_summary.csv")

    # =========================================================================
    # 7. FED VS FASTED KEY COMPARISONS
    # =========================================================================
    print(f"\n{'='*70}")
    print("  7. Fed vs Fasted Key Comparisons")
    print(f"{'='*70}")

    fed_sessions = [2, 4]
    fasted_sessions = [6, 8]

    comparisons = [
        ('discovery_time', 'Discovery Time (s)'),
        ('first_dig_time', 'First Dig Time (s)'),
        ('n_pre_disc', 'Pre-Discovery Visits'),
        ('n_p2_pre', 'Pre-Disc Pot-2 Visits'),
        ('n_p4_pre', 'Pre-Disc Pot-4 Visits'),
    ]

    fig, axes = plt.subplots(1, len(comparisons), figsize=(4 * len(comparisons), 5))
    fig.suptitle('Fed vs Fasted: Behavioral Metrics\nBars = mean, dots = individual sessions',
                 fontsize=14, fontweight='bold')

    for ci, (col, title) in enumerate(comparisons):
        ax = axes[ci]
        fed_vals = metrics_df[metrics_df['state'] == 'fed'][col].values
        fasted_vals = metrics_df[metrics_df['state'] == 'fasted'][col].values

        means = [np.mean(fed_vals), np.mean(fasted_vals)]
        sems = [sp_stats.sem(fed_vals) if len(fed_vals) > 1 else 0,
                sp_stats.sem(fasted_vals) if len(fasted_vals) > 1 else 0]

        ax.bar(['Fed', 'Fasted'], means, yerr=sems, capsize=5,
               color=['royalblue', 'firebrick'], edgecolor='black', alpha=0.6)
        ax.scatter(np.zeros(len(fed_vals)), fed_vals, color='navy', s=60, zorder=5)
        ax.scatter(np.ones(len(fasted_vals)), fasted_vals, color='darkred', s=60, zorder=5)

        if len(fed_vals) >= 2 and len(fasted_vals) >= 2:
            u, p = sp_stats.mannwhitneyu(fed_vals, fasted_vals,
                                         alternative='two-sided')
            sig = '*' if p < 0.05 else 'ns'
            ax.set_title(f'{title}\np={p:.3f} {sig}')
        else:
            ax.set_title(title)

    plt.tight_layout(rect=[0, 0, 1, 0.88])
    plt.savefig('figures/foraging_cross_session_fed_vs_fasted.png',
                dpi=100, bbox_inches='tight')
    plt.close()
    print("  Saved: figures/foraging_cross_session_fed_vs_fasted.png")

    print(f"\n{'='*70}")
    print("  CROSS-SESSION ANALYSIS COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
