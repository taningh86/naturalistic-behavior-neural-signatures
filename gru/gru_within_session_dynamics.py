"""
GRU Within-Session Dynamics Analysis
======================================
Sliding window analysis of GRU hidden states to detect temporal changes
in latent dimensionality, variance, and trajectory speed within each session.
Uses saved models from gru_by_region.py (500ms bins, 32 hidden, 1 layer).
"""

import yaml
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import torch
import torch.nn as nn
import spikeinterface.extractors as se
from sklearn.decomposition import PCA
from scipy import stats as sp_stats
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIG — must match gru_by_region.py
# =============================================================================

BIN_SIZE_MS = 500
FS = 30000
BIN_SAMPLES = int(BIN_SIZE_MS * FS / 1000)
HIDDEN_SIZE = 32
NUM_LAYERS = 1

# Sliding window parameters
WINDOW_BINS = 240       # 240 bins * 500ms = 120 seconds = 2 minutes
STEP_BINS = 30          # 30 bins * 500ms = 15 seconds step

LHA_DEPTH_MAX = 1300
RSP_DEPTH_MIN = 1300

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open("paths.yaml") as f:
    paths_config = yaml.safe_load(f)


# =============================================================================
# MODEL & HELPERS
# =============================================================================

class NeuralGRU(nn.Module):
    def __init__(self, n_neurons, hidden_size, num_layers, dropout):
        super().__init__()
        self.gru = nn.GRU(
            input_size=n_neurons, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_size, n_neurons)

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out


def get_good_units_by_region(sorted_path_obj):
    ci = sorted_path_obj / "cluster_info.tsv"
    if not ci.exists():
        return np.array([]), np.array([])
    df = pd.read_csv(ci, sep='\t')
    if 'depth' not in df.columns:
        return np.array([]), np.array([])
    label_col = None
    if 'group' in df.columns and df['group'].eq('good').any():
        label_col = 'group'
    elif 'KSLabel' in df.columns:
        label_col = 'KSLabel'
    if label_col is None:
        return np.array([]), np.array([])
    good = df[df[label_col] == 'good']
    lha_ids = good[good['depth'] < LHA_DEPTH_MAX]['cluster_id'].values
    rsp_ids = good[good['depth'] >= RSP_DEPTH_MIN]['cluster_id'].values
    return lha_ids, rsp_ids


def bin_spike_trains(sorting, unit_ids):
    spike_trains = {}
    all_min, all_max = np.inf, 0
    for uid in unit_ids:
        st = sorting.get_unit_spike_train(uid)
        spike_trains[uid] = st
        if len(st) > 0:
            all_min = min(all_min, np.min(st))
            all_max = max(all_max, np.max(st))

    n_bins = int((all_max - all_min) / BIN_SAMPLES) + 1
    data = np.zeros((n_bins, len(unit_ids)), dtype=np.float32)
    for i, uid in enumerate(unit_ids):
        st = spike_trains[uid]
        if len(st) > 0:
            b = ((st - all_min) // BIN_SAMPLES).astype(int)
            b = b[(b >= 0) & (b < n_bins)]
            np.add.at(data[:, i], b, 1)

    means = data.mean(axis=0, keepdims=True)
    stds = data.std(axis=0, keepdims=True)
    stds[stds < 1e-8] = 1.0
    data = (data - means) / stds
    return data, n_bins


def extract_hidden_states(model, data, device):
    model.eval()
    hidden_states = []
    with torch.no_grad():
        h = torch.zeros(NUM_LAYERS, 1, HIDDEN_SIZE).to(device)
        for t in range(len(data)):
            x = torch.tensor(data[t:t+1], dtype=torch.float32).unsqueeze(0).to(device)
            _, h = model.gru(x, h)
            hidden_states.append(h[-1, 0, :].cpu().numpy())
    return np.array(hidden_states)


def compute_window_metrics(hidden_states, window_bins, step_bins):
    """Compute PR, variance, speed in sliding windows."""
    n_bins = len(hidden_states)
    window_centers = []
    prs = []
    variances = []
    speeds = []

    for start in range(0, n_bins - window_bins + 1, step_bins):
        end = start + window_bins
        h_win = hidden_states[start:end]

        # Participation ratio
        pca = PCA(n_components=min(HIDDEN_SIZE, len(h_win)))
        pca.fit(h_win)
        evals = pca.explained_variance_
        pr = (np.sum(evals))**2 / np.sum(evals**2)
        prs.append(pr)

        # Mean variance across hidden dims
        var = np.mean(np.var(h_win, axis=0))
        variances.append(var)

        # Trajectory speed
        diffs = np.diff(h_win, axis=0)
        step_dist = np.sqrt(np.sum(diffs**2, axis=1))
        speeds.append(np.mean(step_dist))

        # Window center in seconds
        center_bin = start + window_bins // 2
        window_centers.append(center_bin * BIN_SIZE_MS / 1000)

    return np.array(window_centers), np.array(prs), np.array(variances), np.array(speeds)


def detect_significant_deviations(values, z_thresh=2.0):
    """Find windows that deviate > z_thresh from session mean."""
    mean = np.mean(values)
    std = np.std(values)
    if std < 1e-10:
        return np.array([]), np.array([])
    z_scores = (values - mean) / std
    sig_idx = np.where(np.abs(z_scores) > z_thresh)[0]
    return sig_idx, z_scores


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    print(f"Window: {WINDOW_BINS} bins ({WINDOW_BINS * BIN_SIZE_MS / 1000:.0f}s), "
          f"Step: {STEP_BINS} bins ({STEP_BINS * BIN_SIZE_MS / 1000:.0f}s)")

    sessions = paths_config["single_probe"]["coordinates_1"]["mouse01"]["sessions"]

    session_list = [
        ('session_1', 1, 'Fed',    'Exploration'),
        ('session_2', 2, 'Fed',    'Foraging'),
        ('session_3', 3, 'Fed',    'Exploration'),
        ('session_4', 4, 'Fed',    'Foraging'),
        ('session_5', 5, 'Fasted', 'Exploration'),
        ('session_6', 6, 'Fasted', 'Foraging'),
        ('session_7', 7, 'Fasted', 'Exploration'),
        ('session_8', 8, 'Fasted', 'Foraging'),
    ]

    all_window_data = []  # for CSV export

    # =========================================================================
    # FIGURE 1: Time series of PR, variance, speed per session (LHA & RSP)
    # =========================================================================

    for region in ['LHA', 'RSP']:
        fig, axes = plt.subplots(8, 3, figsize=(24, 32))

        for i, (session_name, session_num, state, phase) in enumerate(session_list):
            print(f"\nSession {session_num} {region}: {state}/{phase}...", end=" ")

            # Load model
            model_path = f'data/gru_session{session_num}_{region.lower()}_model.pt'
            try:
                checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
            except FileNotFoundError:
                print("[SKIP]")
                for j in range(3):
                    axes[i, j].text(0.5, 0.5, 'N/A', ha='center', va='center',
                                    transform=axes[i, j].transAxes)
                continue

            n_neurons = checkpoint['config']['n_neurons']
            unit_ids = np.array(checkpoint['unit_ids'])

            model = NeuralGRU(n_neurons, HIDDEN_SIZE, NUM_LAYERS, 0.0).to(DEVICE)
            model.load_state_dict(checkpoint['model_state_dict'])

            # Load and bin data
            sp = Path(sessions[session_name]['sorted'])
            sorting = se.read_kilosort(sp)
            avail = set(sorting.get_unit_ids())
            unit_ids_filtered = np.array([u for u in unit_ids if u in avail])
            data, n_bins = bin_spike_trains(sorting, unit_ids_filtered)

            # Extract hidden states
            hidden = extract_hidden_states(model, data, DEVICE)

            # Sliding window metrics
            centers, prs, variances, speeds = compute_window_metrics(
                hidden, WINDOW_BINS, STEP_BINS)

            # Detect significant deviations
            sig_pr, z_pr = detect_significant_deviations(prs)
            sig_var, z_var = detect_significant_deviations(variances)
            sig_spd, z_spd = detect_significant_deviations(speeds)

            n_sig = len(sig_pr) + len(sig_var) + len(sig_spd)
            print(f"{len(centers)} windows, {n_sig} significant deviations")

            # Store for CSV
            for w in range(len(centers)):
                all_window_data.append({
                    'session': session_num, 'state': state, 'phase': phase,
                    'region': region,
                    'window_center_s': centers[w],
                    'participation_ratio': prs[w],
                    'hidden_variance': variances[w],
                    'trajectory_speed': speeds[w],
                    'z_pr': z_pr[w] if len(z_pr) > 0 else 0,
                    'z_var': z_var[w] if len(z_var) > 0 else 0,
                    'z_speed': z_spd[w] if len(z_spd) > 0 else 0,
                })

            # --- Plot ---
            color = '#2196F3' if state == 'Fed' else '#F44336'
            minutes = centers / 60

            # PR
            ax = axes[i, 0]
            ax.plot(minutes, prs, color=color, linewidth=1.2)
            ax.axhline(y=np.mean(prs), color='gray', linestyle='--', alpha=0.5)
            if len(sig_pr) > 0:
                ax.scatter(minutes[sig_pr], prs[sig_pr], c='black', s=30, zorder=5, marker='x')
            ax.fill_between(minutes, np.mean(prs) - 2*np.std(prs),
                           np.mean(prs) + 2*np.std(prs), alpha=0.1, color='gray')
            ax.set_ylabel(f'S{session_num}\nPR', fontsize=9)
            if i == 0:
                ax.set_title(f'{region} — Participation Ratio', fontsize=11, fontweight='bold')
            if i == 7:
                ax.set_xlabel('Time (min)')

            # Variance
            ax = axes[i, 1]
            ax.plot(minutes, variances, color=color, linewidth=1.2)
            ax.axhline(y=np.mean(variances), color='gray', linestyle='--', alpha=0.5)
            if len(sig_var) > 0:
                ax.scatter(minutes[sig_var], variances[sig_var], c='black', s=30, zorder=5, marker='x')
            ax.fill_between(minutes, np.mean(variances) - 2*np.std(variances),
                           np.mean(variances) + 2*np.std(variances), alpha=0.1, color='gray')
            ax.set_ylabel('Variance', fontsize=9)
            if i == 0:
                ax.set_title(f'{region} — Hidden Variance', fontsize=11, fontweight='bold')
            if i == 7:
                ax.set_xlabel('Time (min)')

            # Speed
            ax = axes[i, 2]
            ax.plot(minutes, speeds, color=color, linewidth=1.2)
            ax.axhline(y=np.mean(speeds), color='gray', linestyle='--', alpha=0.5)
            if len(sig_spd) > 0:
                ax.scatter(minutes[sig_spd], speeds[sig_spd], c='black', s=30, zorder=5, marker='x')
            ax.fill_between(minutes, np.mean(speeds) - 2*np.std(speeds),
                           np.mean(speeds) + 2*np.std(speeds), alpha=0.1, color='gray')
            ax.set_ylabel('Speed', fontsize=9)
            if i == 0:
                ax.set_title(f'{region} — Trajectory Speed', fontsize=11, fontweight='bold')
            if i == 7:
                ax.set_xlabel('Time (min)')

            # Add state/phase label
            label_text = f'{state}/{phase}'
            axes[i, 0].text(-0.15, 0.5, label_text, transform=axes[i, 0].transAxes,
                           fontsize=9, fontweight='bold', color=color,
                           va='center', ha='right', rotation=90)

        plt.suptitle(f'{region} — Within-Session Latent Dynamics (2-min sliding window)',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'figures/gru_within_session_{region.lower()}.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n[OK] Saved figures/gru_within_session_{region.lower()}.png")

    # Save all window data
    window_df = pd.DataFrame(all_window_data)
    window_df.to_csv('data/gru_within_session_windows.csv', index=False)
    print(f"[OK] Saved data/gru_within_session_windows.csv ({len(window_df)} windows)")

    # =========================================================================
    # FIGURE 2: Segment comparison (split each session into thirds)
    # =========================================================================

    print("\n\n--- Segment Comparison (Thirds) ---")
    segment_results = []

    for region in ['LHA', 'RSP']:
        for session_name, session_num, state, phase in session_list:
            rdf = window_df[(window_df['session'] == session_num) &
                           (window_df['region'] == region)]
            if len(rdf) == 0:
                continue

            n_windows = len(rdf)
            third = n_windows // 3
            if third < 3:
                continue

            early = rdf.iloc[:third]
            middle = rdf.iloc[third:2*third]
            late = rdf.iloc[2*third:]

            for metric in ['participation_ratio', 'hidden_variance', 'trajectory_speed']:
                e_vals = early[metric].values
                m_vals = middle[metric].values
                l_vals = late[metric].values

                # Kruskal-Wallis across thirds
                kw_stat, kw_p = sp_stats.kruskal(e_vals, m_vals, l_vals)

                # Pairwise: early vs late
                _, p_el = sp_stats.mannwhitneyu(e_vals, l_vals, alternative='two-sided')

                segment_results.append({
                    'session': session_num, 'state': state, 'phase': phase,
                    'region': region, 'metric': metric,
                    'early_mean': np.mean(e_vals),
                    'middle_mean': np.mean(m_vals),
                    'late_mean': np.mean(l_vals),
                    'kw_p': kw_p,
                    'early_vs_late_p': p_el,
                })

    seg_df = pd.DataFrame(segment_results)
    seg_df.to_csv('data/gru_within_session_segments.csv', index=False)

    # Print significant segment changes
    sig_segments = seg_df[seg_df['kw_p'] < 0.05]
    print(f"\nSignificant within-session changes (KW p < 0.05): {len(sig_segments)} of {len(seg_df)}")
    if len(sig_segments) > 0:
        print(f"\n{'Sess':>5} {'State':>7} {'Phase':>13} {'Region':>7} {'Metric':>22} "
              f"{'Early':>8} {'Mid':>8} {'Late':>8} {'KW p':>8} {'E-L p':>8}")
        for _, r in sig_segments.iterrows():
            print(f"{int(r['session']):>5} {r['state']:>7} {r['phase']:>13} {r['region']:>7} "
                  f"{r['metric']:>22} {r['early_mean']:>8.3f} {r['middle_mean']:>8.3f} "
                  f"{r['late_mean']:>8.3f} {r['kw_p']:>8.4f} {r['early_vs_late_p']:>8.4f}")

    # =========================================================================
    # FIGURE 3: Summary — max PR range within each session
    # =========================================================================

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for idx, region in enumerate(['LHA', 'RSP']):
        ax = axes[idx]
        pr_ranges = []
        session_labels = []
        bar_colors = []

        for session_name, session_num, state, phase in session_list:
            rdf = window_df[(window_df['session'] == session_num) &
                           (window_df['region'] == region)]
            if len(rdf) == 0:
                continue

            pr_min = rdf['participation_ratio'].min()
            pr_max = rdf['participation_ratio'].max()
            pr_mean = rdf['participation_ratio'].mean()
            pr_ranges.append((pr_min, pr_mean, pr_max))
            session_labels.append(f'S{session_num}')
            bar_colors.append('#2196F3' if state == 'Fed' else '#F44336')

        x = np.arange(len(pr_ranges))
        for j, (pr_min, pr_mean, pr_max) in enumerate(pr_ranges):
            ax.bar(j, pr_mean, color=bar_colors[j], alpha=0.7, edgecolor='black', linewidth=0.5)
            ax.errorbar(j, pr_mean, yerr=[[pr_mean - pr_min], [pr_max - pr_mean]],
                       fmt='none', capsize=5, color='black', linewidth=1.5)

        ax.set_xticks(x)
        ax.set_xticklabels(session_labels)
        ax.set_ylabel('Participation Ratio')
        ax.set_title(f'{region} — PR Range Within Sessions\n(bars = mean, whiskers = min/max)')
        ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Within-Session Dimensionality Variation', fontsize=14)
    plt.tight_layout()
    plt.savefig('figures/gru_within_session_pr_range.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n[OK] Saved figures/gru_within_session_pr_range.png")

    # =========================================================================
    # FIGURE 4: Coefficient of variation across sessions
    # =========================================================================

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metrics = ['participation_ratio', 'hidden_variance', 'trajectory_speed']
    metric_labels = ['Participation Ratio', 'Hidden Variance', 'Trajectory Speed']

    for m_idx, (metric, mlabel) in enumerate(zip(metrics, metric_labels)):
        ax = axes[m_idx]
        cvs_lha = []
        cvs_rsp = []
        colors_lha = []
        colors_rsp = []

        for session_name, session_num, state, phase in session_list:
            for region, cv_list, color_list in [('LHA', cvs_lha, colors_lha),
                                                  ('RSP', cvs_rsp, colors_rsp)]:
                rdf = window_df[(window_df['session'] == session_num) &
                               (window_df['region'] == region)]
                if len(rdf) == 0:
                    cv_list.append(0)
                else:
                    vals = rdf[metric].values
                    cv = np.std(vals) / np.mean(vals) if np.mean(vals) > 0 else 0
                    cv_list.append(cv)
                color_list.append('#2196F3' if state == 'Fed' else '#F44336')

        x = np.arange(len(cvs_lha))
        bar_width = 0.35
        ax.bar(x - bar_width/2, cvs_lha, bar_width, color=colors_lha, alpha=0.7,
               edgecolor='black', linewidth=0.5, label='LHA')
        ax.bar(x + bar_width/2, cvs_rsp, bar_width, color=colors_rsp, alpha=0.5,
               edgecolor='black', linewidth=0.5, hatch='///', label='RSP')

        ax.set_xticks(x)
        ax.set_xticklabels([f'S{s+1}' for s in range(len(x))])
        ax.set_ylabel('Coefficient of Variation')
        ax.set_title(f'{mlabel}\nWithin-Session Variability')
        ax.grid(True, alpha=0.3, axis='y')
        if m_idx == 0:
            ax.legend(fontsize=9)

    plt.suptitle('Within-Session Stability of Latent Dynamics', fontsize=14)
    plt.tight_layout()
    plt.savefig('figures/gru_within_session_cv.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved figures/gru_within_session_cv.png")

    # --- Print CV summary ---
    print("\n--- Coefficient of Variation Summary ---")
    for region in ['LHA', 'RSP']:
        for metric in metrics:
            fed_cvs = []
            fas_cvs = []
            for session_name, session_num, state, phase in session_list:
                rdf = window_df[(window_df['session'] == session_num) &
                               (window_df['region'] == region)]
                if len(rdf) == 0:
                    continue
                vals = rdf[metric].values
                cv = np.std(vals) / np.mean(vals) if np.mean(vals) > 0 else 0
                if state == 'Fed':
                    fed_cvs.append(cv)
                else:
                    fas_cvs.append(cv)
            if len(fed_cvs) >= 2 and len(fas_cvs) >= 2:
                _, p = sp_stats.mannwhitneyu(fed_cvs, fas_cvs, alternative='two-sided')
                print(f"  {region} {metric}: Fed CV={np.mean(fed_cvs):.4f} vs "
                      f"Fasted CV={np.mean(fas_cvs):.4f}, p={p:.4f}")

    print(f"\n[DONE]")
