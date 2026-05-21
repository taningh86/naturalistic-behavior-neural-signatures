"""
GRU Within-Session Dynamics — Dual-Probe (ACA & LHA)
=====================================================
Sliding window analysis of GRU hidden states for dual-probe sessions.
Uses saved models from gru_dual_probe.py (500ms bins, 32 hidden, 1 layer).
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
# CONFIG
# =============================================================================

BIN_SIZE_MS = 500
FS = 30000
BIN_SAMPLES = int(BIN_SIZE_MS * FS / 1000)
HIDDEN_SIZE = 32
NUM_LAYERS = 1

WINDOW_BINS = 240       # 2 minutes
STEP_BINS = 30          # 15 seconds

LHA_DEPTH_MIN = 0
LHA_DEPTH_MAX = 345

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


def compute_cluster_depths(sorted_path_obj):
    templates_path = sorted_path_obj / "templates.npy"
    chan_pos_path = sorted_path_obj / "channel_positions.npy"
    if not templates_path.exists() or not chan_pos_path.exists():
        return {}
    templates = np.load(templates_path)
    channel_positions = np.load(chan_pos_path)
    peak_channels = np.argmax(np.max(np.abs(templates), axis=1), axis=1)
    depths = channel_positions[peak_channels, 1]
    spike_clusters_path = sorted_path_obj / "spike_clusters.npy"
    spike_templates_path = sorted_path_obj / "spike_templates.npy"
    if spike_clusters_path.exists() and spike_templates_path.exists():
        spike_clusters = np.load(spike_clusters_path).flatten()
        spike_templates = np.load(spike_templates_path).flatten()
        unique_clusters = np.unique(spike_clusters)
        cluster_depths = {}
        for cid in unique_clusters:
            mask = spike_clusters == cid
            templates_for_cluster = spike_templates[mask]
            if len(templates_for_cluster) > 0:
                most_common_template = np.bincount(templates_for_cluster).argmax()
                if most_common_template < len(depths):
                    cluster_depths[cid] = depths[most_common_template]
        return cluster_depths
    else:
        return {i: depths[i] for i in range(len(depths))}


def get_good_unit_ids(sorted_path_obj):
    ci = sorted_path_obj / "cluster_info.tsv"
    if ci.exists():
        df = pd.read_csv(ci, sep='\t')
        if 'group' in df.columns and df['group'].eq('good').any():
            return df[df['group'] == 'good']['cluster_id'].values
        if 'KSLabel' in df.columns:
            return df[df['KSLabel'] == 'good']['cluster_id'].values
    cg = sorted_path_obj / "cluster_group.tsv"
    if cg.exists():
        df = pd.read_csv(cg, sep='\t')
        col = df.columns[1]
        return df[df[col].str.strip() == 'good'].iloc[:, 0].values
    return np.array([])


def get_good_lha_unit_ids(sorted_path_obj):
    ci = sorted_path_obj / "cluster_info.tsv"
    if ci.exists():
        df = pd.read_csv(ci, sep='\t')
        if 'depth' in df.columns:
            label_col = None
            if 'group' in df.columns and df['group'].eq('good').any():
                label_col = 'group'
            elif 'KSLabel' in df.columns:
                label_col = 'KSLabel'
            if label_col is not None:
                good_lha = df[(df[label_col] == 'good') &
                              (df['depth'] >= LHA_DEPTH_MIN) &
                              (df['depth'] <= LHA_DEPTH_MAX)]
                return good_lha['cluster_id'].values
            return np.array([])
    cluster_depths = compute_cluster_depths(sorted_path_obj)
    if not cluster_depths:
        return np.array([])
    cg = sorted_path_obj / "cluster_group.tsv"
    if not cg.exists():
        cg = sorted_path_obj / "cluster_KSLabel.tsv"
    if not cg.exists():
        return np.array([])
    df = pd.read_csv(cg, sep='\t')
    col = df.columns[1]
    good_ids = df[df[col].str.strip() == 'good'].iloc[:, 0].values
    lha_ids = [cid for cid in good_ids
               if cid in cluster_depths and LHA_DEPTH_MIN <= cluster_depths[cid] <= LHA_DEPTH_MAX]
    return np.array(lha_ids)


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
    n_bins = len(hidden_states)
    window_centers = []
    prs = []
    variances = []
    speeds = []

    for start in range(0, n_bins - window_bins + 1, step_bins):
        end = start + window_bins
        h_win = hidden_states[start:end]

        pca = PCA(n_components=min(HIDDEN_SIZE, len(h_win)))
        pca.fit(h_win)
        evals = pca.explained_variance_
        pr = (np.sum(evals))**2 / np.sum(evals**2)
        prs.append(pr)

        var = np.mean(np.var(h_win, axis=0))
        variances.append(var)

        diffs = np.diff(h_win, axis=0)
        step_dist = np.sqrt(np.sum(diffs**2, axis=1))
        speeds.append(np.mean(step_dist))

        center_bin = start + window_bins // 2
        window_centers.append(center_bin * BIN_SIZE_MS / 1000)

    return np.array(window_centers), np.array(prs), np.array(variances), np.array(speeds)


# =============================================================================
# SESSION DEFINITIONS
# =============================================================================

dp_config = paths_config["double_probe"]["coordinates_1"]["mouse01"]["sessions"]
STATE_MAP = {'fed': 'Fed', 'fasted': 'Fasted', 'fed-HFD': 'HFD'}

all_sessions = []
for i in range(1, 25):
    key = f"session_{i}"
    if key not in dp_config:
        continue
    sc = dp_config[key]
    p0_data = sc.get("probe_0_aca", {})
    p1_data = sc.get("probe_1_lha_rsp", {})
    p0_path = p0_data.get("sorted") if isinstance(p0_data, dict) else None
    p1_path = p1_data.get("sorted") if isinstance(p1_data, dict) else None
    if p0_path is None or p1_path is None:
        continue
    yaml_state = sc.get("state", "")
    state = STATE_MAP.get(yaml_state, yaml_state)
    phase = 'Exploration' if i % 2 == 1 else 'Foraging'
    all_sessions.append((key, i, state, phase, p0_path, p1_path))


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    print(f"Window: {WINDOW_BINS} bins ({WINDOW_BINS * BIN_SIZE_MS / 1000:.0f}s), "
          f"Step: {STEP_BINS} bins ({STEP_BINS * BIN_SIZE_MS / 1000:.0f}s)")
    print(f"Sessions: {len(all_sessions)}")

    all_window_data = []

    for region in ['ACA', 'LHA']:
        # Determine how many sessions per condition for subplot layout
        state_sessions = {}
        for _, snum, state, phase, _, _ in all_sessions:
            model_path = f'data/gru_dp_session{snum}_{region.lower()}_model.pt'
            try:
                torch.load(model_path, map_location='cpu', weights_only=False)
                state_sessions.setdefault(state, []).append(snum)
            except FileNotFoundError:
                pass

        n_sessions_total = sum(len(v) for v in state_sessions.values())

        # Create figure: one row per session, 3 columns (PR, variance, speed)
        fig, axes = plt.subplots(n_sessions_total, 3, figsize=(24, 3 * n_sessions_total))
        if n_sessions_total == 1:
            axes = axes.reshape(1, -1)

        row_idx = 0

        for session_key, session_num, state, phase, p0_path, p1_path in all_sessions:
            probe_path = p0_path if region == 'ACA' else p1_path

            model_path = f'data/gru_dp_session{session_num}_{region.lower()}_model.pt'
            try:
                checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
            except FileNotFoundError:
                continue

            print(f"Session {session_num} {region}: {state}/{phase}...", end=" ")

            n_neurons = checkpoint['config']['n_neurons']
            unit_ids = np.array(checkpoint['unit_ids'])

            model = NeuralGRU(n_neurons, HIDDEN_SIZE, NUM_LAYERS, 0.0).to(DEVICE)
            model.load_state_dict(checkpoint['model_state_dict'])

            # Load and bin data
            sp = Path(probe_path)
            sorting = se.read_kilosort(sp)
            avail = set(sorting.get_unit_ids())
            unit_ids_filtered = np.array([u for u in unit_ids if u in avail])
            data, n_bins = bin_spike_trains(sorting, unit_ids_filtered)

            # Check if enough bins for window
            if n_bins < WINDOW_BINS + STEP_BINS:
                print(f"[SKIP] only {n_bins} bins, need {WINDOW_BINS}")
                for j in range(3):
                    axes[row_idx, j].text(0.5, 0.5, 'Too short', ha='center',
                                          va='center', transform=axes[row_idx, j].transAxes)
                row_idx += 1
                continue

            # Extract hidden states
            hidden = extract_hidden_states(model, data, DEVICE)

            # Sliding window
            centers, prs, variances, speeds = compute_window_metrics(
                hidden, WINDOW_BINS, STEP_BINS)

            print(f"{len(centers)} windows")

            # Store for CSV
            for w in range(len(centers)):
                z_pr = (prs[w] - np.mean(prs)) / (np.std(prs) + 1e-10)
                z_var = (variances[w] - np.mean(variances)) / (np.std(variances) + 1e-10)
                z_spd = (speeds[w] - np.mean(speeds)) / (np.std(speeds) + 1e-10)
                all_window_data.append({
                    'session': session_num, 'state': state, 'phase': phase,
                    'region': region,
                    'window_center_s': centers[w],
                    'participation_ratio': prs[w],
                    'hidden_variance': variances[w],
                    'trajectory_speed': speeds[w],
                    'z_pr': z_pr, 'z_var': z_var, 'z_speed': z_spd,
                })

            # --- Plot ---
            state_colors = {'Fed': '#2196F3', 'Fasted': '#F44336', 'HFD': '#FF9800'}
            color = state_colors.get(state, 'gray')
            minutes = centers / 60

            for col, (vals, ylabel, title) in enumerate([
                (prs, 'PR', 'Participation Ratio'),
                (variances, 'Variance', 'Hidden Variance'),
                (speeds, 'Speed', 'Trajectory Speed'),
            ]):
                ax = axes[row_idx, col]
                ax.plot(minutes, vals, color=color, linewidth=1.2)
                ax.axhline(y=np.mean(vals), color='gray', linestyle='--', alpha=0.5)
                ax.fill_between(minutes, np.mean(vals) - 2*np.std(vals),
                               np.mean(vals) + 2*np.std(vals), alpha=0.1, color='gray')
                ax.set_ylabel(ylabel, fontsize=8)
                if row_idx == 0:
                    ax.set_title(f'{region} -- {title}', fontsize=11, fontweight='bold')
                if row_idx == n_sessions_total - 1:
                    ax.set_xlabel('Time (min)')
                ax.tick_params(labelsize=7)
                ax.grid(True, alpha=0.2)

            # Label on left
            axes[row_idx, 0].text(-0.18, 0.5,
                                   f'S{session_num}\n{state}\n{phase[:3]}',
                                   transform=axes[row_idx, 0].transAxes,
                                   fontsize=8, fontweight='bold', color=color,
                                   va='center', ha='right')
            row_idx += 1

        plt.suptitle(f'{region} -- Within-Session Latent Dynamics (2-min sliding window)',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'figures/gru_dp_within_session_{region.lower()}.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n[OK] Saved figures/gru_dp_within_session_{region.lower()}.png")

    # Save window data
    window_df = pd.DataFrame(all_window_data)
    window_df.to_csv('data/gru_dp_within_session_windows.csv', index=False)
    print(f"[OK] Saved CSV ({len(window_df)} windows)")

    # =========================================================================
    # Segment comparison (thirds)
    # =========================================================================

    print("\n--- Segment Comparison (Thirds) ---")
    segment_results = []

    for region in ['ACA', 'LHA']:
        for _, session_num, state, phase, _, _ in all_sessions:
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
                kw_stat, kw_p = sp_stats.kruskal(e_vals, m_vals, l_vals)
                _, p_el = sp_stats.mannwhitneyu(e_vals, l_vals, alternative='two-sided')

                segment_results.append({
                    'session': session_num, 'state': state, 'phase': phase,
                    'region': region, 'metric': metric,
                    'early_mean': np.mean(e_vals),
                    'middle_mean': np.mean(m_vals),
                    'late_mean': np.mean(l_vals),
                    'kw_p': kw_p, 'early_vs_late_p': p_el,
                })

    seg_df = pd.DataFrame(segment_results)
    seg_df.to_csv('data/gru_dp_within_session_segments.csv', index=False)

    sig_segments = seg_df[seg_df['kw_p'] < 0.05]
    total_tests = len(seg_df)
    print(f"\nSignificant within-session changes (KW p < 0.05): {len(sig_segments)} of {total_tests}")

    if len(sig_segments) > 0:
        print(f"\n{'Sess':>5} {'State':>7} {'Phase':>5} {'Region':>7} {'Metric':>22} "
              f"{'Early':>8} {'Mid':>8} {'Late':>8} {'KW p':>8} {'E-L p':>8}")
        for _, r in sig_segments.iterrows():
            print(f"{int(r['session']):>5} {r['state']:>7} {r['phase'][:5]:>5} {r['region']:>7} "
                  f"{r['metric']:>22} {r['early_mean']:>8.3f} {r['middle_mean']:>8.3f} "
                  f"{r['late_mean']:>8.3f} {r['kw_p']:>8.4f} {r['early_vs_late_p']:>8.4f}")

    # =========================================================================
    # Summary: CV by condition
    # =========================================================================

    print("\n--- Coefficient of Variation Summary ---")
    metrics = ['participation_ratio', 'hidden_variance', 'trajectory_speed']

    for region in ['ACA', 'LHA']:
        for metric in metrics:
            cv_by_state = {}
            for _, session_num, state, phase, _, _ in all_sessions:
                rdf = window_df[(window_df['session'] == session_num) &
                               (window_df['region'] == region)]
                if len(rdf) == 0:
                    continue
                vals = rdf[metric].values
                cv = np.std(vals) / np.mean(vals) if np.mean(vals) > 0 else 0
                cv_by_state.setdefault(state, []).append(cv)

            states = list(cv_by_state.keys())
            if len(states) >= 2:
                vals_str = ", ".join([f"{s}={np.mean(cv_by_state[s]):.4f}" for s in states])
                # 3-way KW if 3 groups
                if len(states) == 3 and all(len(cv_by_state[s]) >= 2 for s in states):
                    kw_stat, kw_p = sp_stats.kruskal(*[cv_by_state[s] for s in states])
                    print(f"  {region} {metric}: {vals_str}, KW p={kw_p:.4f}")
                else:
                    print(f"  {region} {metric}: {vals_str}")

    print(f"\n[DONE]")
