"""
GRU Neural Dynamics — Dual-Probe (ACA & LHA)
==============================================
Trains separate per-session GRU models for ACA (probe-0) and LHA (probe-1).
Covers Fed (sessions 1-10), Fasted (11-16), and HFD (17-22).
Same config as single-probe: 500ms bins, 32 hidden, 1 layer.
"""

import yaml
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.decomposition import PCA
from scipy import stats as sp_stats
import spikeinterface.extractors as se
import warnings
import time

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

BIN_SIZE_MS = 500
FS = 30000
BIN_SAMPLES = int(BIN_SIZE_MS * FS / 1000)

SEQ_LEN = 10
HIDDEN_SIZE = 32
NUM_LAYERS = 1
DROPOUT = 0.0
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
NUM_EPOCHS = 100
PATIENCE = 10
TRAIN_FRAC = 0.8

# Dual-probe LHA depth range (probe-1)
LHA_DEPTH_MIN = 0
LHA_DEPTH_MAX = 345  # um

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open("paths.yaml") as f:
    paths_config = yaml.safe_load(f)


# =============================================================================
# UNIT SELECTION
# =============================================================================

def get_good_unit_ids(sorted_path_obj):
    """Get all good unit IDs (no depth filtering) — for ACA probe-0."""
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


def compute_cluster_depths(sorted_path_obj):
    """Compute depth per cluster from templates.npy + channel_positions.npy.
    Fallback for sessions without cluster_info.tsv depth column.
    """
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


def get_good_lha_unit_ids(sorted_path_obj):
    """Get good unit IDs in LHA depth range (0-345um) — for probe-1.
    Falls back to computing depth from templates if cluster_info.tsv missing.
    """
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

    # Fallback: compute depths from templates
    print(f"      [INFO] No cluster_info.tsv, computing depths from templates...")
    cluster_depths = compute_cluster_depths(sorted_path_obj)
    if not cluster_depths:
        return np.array([])

    # Get good labels from cluster_group.tsv
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


# =============================================================================
# DATA PIPELINE
# =============================================================================

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
    rec_duration_s = (all_max - all_min) / FS

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

    return data, n_bins, rec_duration_s


# =============================================================================
# DATASET & MODEL
# =============================================================================

class NeuralSequenceDataset(Dataset):
    def __init__(self, data, seq_len):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.seq_len = seq_len
        self.n_samples = len(data) - seq_len

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + self.seq_len]
        return x, y


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


# =============================================================================
# TRAINING & EVALUATION
# =============================================================================

def train_model(model, train_loader, val_loader, n_epochs, patience, lr, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val_loss = np.inf
    best_epoch = 0
    best_state = None
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(n_epochs):
        model.train()
        train_losses = []
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            pred = model(x_batch)
            loss = criterion(pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                pred = model(x_batch)
                loss = criterion(pred, y_batch)
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= patience:
            break

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"        Epoch {epoch+1:>3}: train={train_loss:.6f}, val={val_loss:.6f}")

    model.load_state_dict(best_state)
    return model, history, best_epoch + 1, best_val_loss


def evaluate_model(model, data_loader, device):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            x_batch = x_batch.to(device)
            pred = model(x_batch)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y_batch.numpy())

    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    mse = np.mean((preds - targets) ** 2)
    ss_res = np.sum((targets - preds) ** 2, axis=0)
    ss_tot = np.sum((targets - targets.mean(axis=0, keepdims=True)) ** 2, axis=0)
    r2_per_neuron = 1 - ss_res / (ss_tot + 1e-10)
    r2_overall = 1 - np.sum(ss_res) / (np.sum(ss_tot) + 1e-10)

    return mse, r2_overall, r2_per_neuron


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


def compute_dimensionality(hidden_states):
    pca = PCA(n_components=min(HIDDEN_SIZE, hidden_states.shape[0]))
    pca.fit(hidden_states)
    eigenvalues = pca.explained_variance_
    pr = (np.sum(eigenvalues))**2 / np.sum(eigenvalues**2)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n90 = np.searchsorted(cumvar, 0.90) + 1
    n95 = np.searchsorted(cumvar, 0.95) + 1
    return pr, n90, n95


# =============================================================================
# SESSION DEFINITIONS
# =============================================================================

dp_config = paths_config["double_probe"]["coordinates_1"]["mouse01"]["sessions"]

# State mapping from YAML values to labels
STATE_MAP = {'fed': 'Fed', 'fasted': 'Fasted', 'fed-HFD': 'HFD'}

all_sessions = []

for i in range(1, 25):
    key = f"session_{i}"
    if key not in dp_config:
        continue
    sc = dp_config[key]

    # Get probe paths from correct YAML structure
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

fed_sessions = [s for s in all_sessions if s[2] == 'Fed']
fasted_sessions_list = [s for s in all_sessions if s[2] == 'Fasted']
hfd_sessions = [s for s in all_sessions if s[2] == 'HFD']


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    print(f"Config: {BIN_SIZE_MS}ms bins, {HIDDEN_SIZE} hidden, {NUM_LAYERS} layer")
    print(f"Sessions: {len(fed_sessions)} Fed, {len(fasted_sessions_list)} Fasted, {len(hfd_sessions)} HFD")
    print(f"Running separate ACA and LHA models per session\n")

    results = []

    for session_key, session_num, state, phase, p0_path, p1_path in all_sessions:
        print(f"\n{'='*70}")
        print(f"Session {session_num}: {state} / {phase}")
        print(f"{'='*70}")

        for region, probe_path in [('ACA', p0_path), ('LHA', p1_path)]:
            sp = Path(probe_path)

            # Get units
            if region == 'ACA':
                unit_ids = get_good_unit_ids(sp)
            else:
                unit_ids = get_good_lha_unit_ids(sp)

            n_neurons = len(unit_ids)
            if n_neurons < 3:
                print(f"  [{region}] SKIP -- only {n_neurons} units")
                continue

            print(f"\n  --- {region} ({n_neurons} neurons) ---")

            # Load sorting and filter
            sorting = se.read_kilosort(sp)
            avail = set(sorting.get_unit_ids())
            unit_ids = np.array([u for u in unit_ids if u in avail])
            n_neurons = len(unit_ids)

            if n_neurons < 3:
                print(f"    SKIP -- only {n_neurons} units available in sorting")
                continue

            print(f"    Available: {n_neurons} units")

            # Bin
            t0 = time.time()
            data, n_bins, rec_dur = bin_spike_trains(sorting, unit_ids)
            print(f"    Binned: {n_bins} bins ({rec_dur:.0f}s) in {time.time()-t0:.1f}s")

            # Split
            split_idx = int(n_bins * TRAIN_FRAC)
            train_data = data[:split_idx]
            test_data = data[split_idx:]

            train_dataset = NeuralSequenceDataset(train_data, SEQ_LEN)
            test_dataset = NeuralSequenceDataset(test_data, SEQ_LEN)

            if len(train_dataset) < BATCH_SIZE or len(test_dataset) < 10:
                print(f"    SKIP -- not enough data")
                continue

            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

            # Train
            model = NeuralGRU(n_neurons, HIDDEN_SIZE, NUM_LAYERS, DROPOUT).to(DEVICE)
            n_params = sum(p.numel() for p in model.parameters())
            print(f"    Model: {n_params:,} parameters")
            print(f"    Training...")

            t0 = time.time()
            model, history, best_epoch, best_val_loss = train_model(
                model, train_loader, test_loader, NUM_EPOCHS, PATIENCE, LEARNING_RATE, DEVICE
            )
            train_time = time.time() - t0
            print(f"    Done in {train_time:.1f}s (best epoch: {best_epoch})")

            # Evaluate
            train_mse, train_r2, train_r2_neurons = evaluate_model(model, train_loader, DEVICE)
            test_mse, test_r2, test_r2_neurons = evaluate_model(model, test_loader, DEVICE)
            print(f"    Train -- MSE: {train_mse:.6f}, R2: {train_r2:.4f}")
            print(f"    Test  -- MSE: {test_mse:.6f}, R2: {test_r2:.4f}")

            # Hidden states & dimensionality
            hidden = extract_hidden_states(model, data, DEVICE)
            pr, n90, n95 = compute_dimensionality(hidden)
            print(f"    PR: {pr:.2f}, PCs@90%: {n90}, PCs@95%: {n95}")

            results.append({
                'session': session_num, 'state': state, 'phase': phase,
                'region': region, 'n_neurons': n_neurons, 'n_bins': n_bins,
                'train_mse': train_mse, 'test_mse': test_mse,
                'train_r2': train_r2, 'test_r2': test_r2,
                'test_r2_median': np.median(test_r2_neurons),
                'best_epoch': best_epoch, 'n_params': n_params,
                'participation_ratio': pr, 'pcs_90': n90, 'pcs_95': n95,
                'hidden_var': np.mean(np.var(hidden, axis=0)),
                'traj_speed': np.mean(np.sqrt(np.sum(np.diff(hidden, axis=0)**2, axis=1))),
            })

            # Save model
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': {
                    'n_neurons': n_neurons, 'hidden_size': HIDDEN_SIZE,
                    'num_layers': NUM_LAYERS, 'seq_len': SEQ_LEN,
                    'bin_size_ms': BIN_SIZE_MS,
                },
                'unit_ids': unit_ids.tolist(),
                'session': session_num, 'state': state, 'phase': phase, 'region': region,
            }, f'data/gru_dp_session{session_num}_{region.lower()}_model.pt')

            # Save training history
            pd.DataFrame(history).to_csv(
                f'data/gru_dp_session{session_num}_{region.lower()}_history.csv', index=False)

    # =============================================================================
    # SUMMARY
    # =============================================================================

    results_df = pd.DataFrame(results)
    results_df.to_csv('data/gru_dual_probe_results.csv', index=False)

    print(f"\n\n{'='*95}")
    print("SUMMARY -- GRU Dual-Probe by Region")
    print(f"{'='*95}")
    print(f"{'Sess':>5} {'State':>7} {'Phase':>13} {'Region':>7} {'Neurons':>8} "
          f"{'Test MSE':>10} {'Test R2':>9} {'PR':>8} {'PCs@90':>7} {'HidVar':>8} {'Speed':>8}")
    print("-" * 95)

    for _, r in results_df.iterrows():
        print(f"{int(r['session']):>5} {r['state']:>7} {r['phase']:>13} {r['region']:>7} "
              f"{int(r['n_neurons']):>8} {r['test_mse']:>10.6f} {r['test_r2']:>9.4f} "
              f"{r['participation_ratio']:>8.2f} {int(r['pcs_90']):>7} "
              f"{r['hidden_var']:>8.4f} {r['traj_speed']:>8.4f}")

    # --- Statistical comparisons per region ---
    for region in ['ACA', 'LHA']:
        rdf = results_df[results_df['region'] == region]

        # Fed vs Fasted
        fed = rdf[rdf['state'] == 'Fed']
        fas = rdf[rdf['state'] == 'Fasted']
        if len(fed) >= 2 and len(fas) >= 2:
            print(f"\n--- {region}: Fed vs Fasted ---")
            for metric in ['test_r2', 'participation_ratio', 'pcs_90', 'hidden_var', 'traj_speed']:
                f_vals = fed[metric].values
                fa_vals = fas[metric].values
                _, p = sp_stats.mannwhitneyu(f_vals, fa_vals, alternative='two-sided')
                print(f"  {metric}: Fed={np.mean(f_vals):.4f} vs Fasted={np.mean(fa_vals):.4f}, p={p:.4f}")

        # Fed vs HFD
        hfd = rdf[rdf['state'] == 'HFD']
        if len(fed) >= 2 and len(hfd) >= 2:
            print(f"\n--- {region}: Fed vs HFD ---")
            for metric in ['test_r2', 'participation_ratio', 'pcs_90', 'hidden_var', 'traj_speed']:
                f_vals = fed[metric].values
                h_vals = hfd[metric].values
                _, p = sp_stats.mannwhitneyu(f_vals, h_vals, alternative='two-sided')
                print(f"  {metric}: Fed={np.mean(f_vals):.4f} vs HFD={np.mean(h_vals):.4f}, p={p:.4f}")

        # Fasted vs HFD
        if len(fas) >= 2 and len(hfd) >= 2:
            print(f"\n--- {region}: Fasted vs HFD ---")
            for metric in ['test_r2', 'participation_ratio', 'pcs_90', 'hidden_var', 'traj_speed']:
                fa_vals = fas[metric].values
                h_vals = hfd[metric].values
                _, p = sp_stats.mannwhitneyu(fa_vals, h_vals, alternative='two-sided')
                print(f"  {metric}: Fasted={np.mean(fa_vals):.4f} vs HFD={np.mean(h_vals):.4f}, p={p:.4f}")

        # 3-way Kruskal-Wallis
        if len(fed) >= 2 and len(fas) >= 2 and len(hfd) >= 2:
            print(f"\n--- {region}: 3-Way KW (Fed vs Fasted vs HFD) ---")
            for metric in ['test_r2', 'participation_ratio', 'pcs_90', 'hidden_var', 'traj_speed']:
                f_vals = fed[metric].values
                fa_vals = fas[metric].values
                h_vals = hfd[metric].values
                kw_stat, kw_p = sp_stats.kruskal(f_vals, fa_vals, h_vals)
                print(f"  {metric}: KW p={kw_p:.4f}")

    # ACA vs LHA
    print(f"\n--- ACA vs LHA (all sessions) ---")
    for metric in ['test_r2', 'participation_ratio', 'pcs_90', 'hidden_var', 'traj_speed']:
        aca_vals = results_df[results_df['region'] == 'ACA'][metric].values
        lha_vals = results_df[results_df['region'] == 'LHA'][metric].values
        if len(aca_vals) >= 2 and len(lha_vals) >= 2:
            _, p = sp_stats.mannwhitneyu(aca_vals, lha_vals, alternative='two-sided')
            print(f"  {metric}: ACA={np.mean(aca_vals):.4f} vs LHA={np.mean(lha_vals):.4f}, p={p:.4f}")

    # =============================================================================
    # FIGURES
    # =============================================================================

    state_colors = {'Fed': '#2196F3', 'Fasted': '#F44336', 'HFD': '#FF9800'}

    # --- Figure 1: Test R2 and PR by session ---
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))

    for col, region in enumerate(['ACA', 'LHA']):
        rdf = results_df[results_df['region'] == region]

        # R2
        ax = axes[0, col]
        for _, r in rdf.iterrows():
            marker = 'o' if r['phase'] == 'Exploration' else 's'
            ax.scatter(r['session'], r['test_r2'],
                       c=state_colors[r['state']], marker=marker, s=100,
                       edgecolors='black', linewidth=0.5, zorder=5)
        ax.set_xlabel('Session')
        ax.set_ylabel('Test R2')
        ax.set_title(f'{region} -- Test R2 by Session')
        ax.grid(True, alpha=0.3)

        # PR
        ax = axes[1, col]
        for _, r in rdf.iterrows():
            marker = 'o' if r['phase'] == 'Exploration' else 's'
            ax.scatter(r['session'], r['participation_ratio'],
                       c=state_colors[r['state']], marker=marker, s=100,
                       edgecolors='black', linewidth=0.5, zorder=5)
        ax.set_xlabel('Session')
        ax.set_ylabel('Participation Ratio')
        ax.set_title(f'{region} -- Dimensionality by Session')
        ax.grid(True, alpha=0.3)

    # Shared legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2196F3', markersize=10, label='Fed Exp'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#2196F3', markersize=10, label='Fed For'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#F44336', markersize=10, label='Fasted Exp'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#F44336', markersize=10, label='Fasted For'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF9800', markersize=10, label='HFD Exp'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#FF9800', markersize=10, label='HFD For'),
    ]
    axes[0, 1].legend(handles=legend_elements, fontsize=8, loc='upper right')

    plt.suptitle('GRU Dual-Probe: Performance & Dimensionality', fontsize=14)
    plt.tight_layout()
    plt.savefig('figures/gru_dual_probe_overview.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n[OK] Saved figures/gru_dual_probe_overview.png")

    # --- Figure 2: 3-way comparison bars ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    metrics_to_plot = ['test_r2', 'participation_ratio', 'traj_speed']
    metric_labels = ['Test R2', 'Participation Ratio', 'Trajectory Speed']

    for row, region in enumerate(['ACA', 'LHA']):
        rdf = results_df[results_df['region'] == region]

        for col, (metric, mlabel) in enumerate(zip(metrics_to_plot, metric_labels)):
            ax = axes[row, col]

            for j, st in enumerate(['Fed', 'Fasted', 'HFD']):
                vals = rdf[rdf['state'] == st][metric].values
                if len(vals) == 0:
                    continue
                ax.bar(j, np.mean(vals),
                       yerr=np.std(vals)/np.sqrt(len(vals)),
                       color=state_colors[st], capsize=5, alpha=0.7, width=0.6)
                for v in vals:
                    ax.scatter(j, v, c=state_colors[st], edgecolors='black',
                               linewidth=0.5, s=40, zorder=5)

            ax.set_xticks([0, 1, 2])
            ax.set_xticklabels(['Fed', 'Fasted', 'HFD'])
            ax.set_ylabel(mlabel)
            ax.set_title(f'{region} -- {mlabel}')
            ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('GRU Dual-Probe: Fed vs Fasted vs HFD', fontsize=14)
    plt.tight_layout()
    plt.savefig('figures/gru_dual_probe_3way.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved figures/gru_dual_probe_3way.png")

    print(f"\n[DONE] GRU dual-probe analysis complete!")
