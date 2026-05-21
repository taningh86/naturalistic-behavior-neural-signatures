"""
GRU Neural Dynamics — Separate LHA and RSP Models
===================================================
Trains separate per-session GRU models for LHA-only and RSP-only neurons.
Same config as combined model: 500ms bins, 32 hidden, 1 layer.
Compares prediction performance and latent dimensionality across regions,
metabolic states, and behavioral phases.
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

LHA_DEPTH_MAX = 1300
RSP_DEPTH_MIN = 1300

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open("paths.yaml") as f:
    paths_config = yaml.safe_load(f)


# =============================================================================
# DATA LOADING
# =============================================================================

def get_good_units_by_region(sorted_path_obj):
    """Get good LHA and RSP unit IDs from cluster_info.tsv, split by depth."""
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
    """Bin spike trains and z-score each neuron."""
    import spikeinterface.extractors as se

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
    """Run data through GRU step-by-step and extract hidden states."""
    model.eval()
    n_bins = len(data)
    hidden_states = []

    with torch.no_grad():
        h = torch.zeros(NUM_LAYERS, 1, HIDDEN_SIZE).to(device)
        for t in range(n_bins):
            x = torch.tensor(data[t:t+1], dtype=torch.float32).unsqueeze(0).to(device)
            _, h = model.gru(x, h)
            hidden_states.append(h[-1, 0, :].cpu().numpy())

    return np.array(hidden_states)


def compute_dimensionality(hidden_states):
    """Compute participation ratio and PCs needed for 90%/95% variance."""
    pca = PCA(n_components=min(HIDDEN_SIZE, hidden_states.shape[0]))
    pca.fit(hidden_states)
    eigenvalues = pca.explained_variance_

    pr = (np.sum(eigenvalues))**2 / np.sum(eigenvalues**2)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n90 = np.searchsorted(cumvar, 0.90) + 1
    n95 = np.searchsorted(cumvar, 0.95) + 1

    return pr, n90, n95, cumvar


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import spikeinterface.extractors as se

    print(f"Device: {DEVICE}")
    print(f"Config: {BIN_SIZE_MS}ms bins, {HIDDEN_SIZE} hidden, {NUM_LAYERS} layer")
    print(f"Running SEPARATE LHA and RSP models per session\n")

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

    results = []

    for session_name, session_num, state, phase in session_list:
        print(f"{'='*70}")
        print(f"Session {session_num}: {state} / {phase}")
        print(f"{'='*70}")

        session_config = sessions[session_name]
        sorted_path = session_config.get("sorted")
        if sorted_path is None:
            print("  [SKIP] No sorted path")
            continue

        sp = Path(sorted_path)
        sorting = se.read_kilosort(sp)
        avail = set(sorting.get_unit_ids())

        lha_ids, rsp_ids = get_good_units_by_region(sp)
        lha_ids = np.array([u for u in lha_ids if u in avail])
        rsp_ids = np.array([u for u in rsp_ids if u in avail])

        print(f"  Units: {len(lha_ids)} LHA, {len(rsp_ids)} RSP")

        for region, unit_ids in [('LHA', lha_ids), ('RSP', rsp_ids)]:
            n_neurons = len(unit_ids)
            if n_neurons < 3:
                print(f"  [{region}] SKIP — only {n_neurons} units")
                continue

            print(f"\n  --- {region} ({n_neurons} neurons) ---")

            # Bin spike trains
            t0 = time.time()
            data, n_bins, rec_dur = bin_spike_trains(sorting, unit_ids)
            print(f"    Binned: {n_bins} bins ({rec_dur:.0f}s) in {time.time()-t0:.1f}s")

            # Temporal split
            split_idx = int(n_bins * TRAIN_FRAC)
            train_data = data[:split_idx]
            test_data = data[split_idx:]

            train_dataset = NeuralSequenceDataset(train_data, SEQ_LEN)
            test_dataset = NeuralSequenceDataset(test_data, SEQ_LEN)

            if len(train_dataset) < BATCH_SIZE or len(test_dataset) < 10:
                print(f"    [{region}] SKIP — not enough data")
                continue

            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

            # Build and train
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

            print(f"    Train — MSE: {train_mse:.6f}, R2: {train_r2:.4f}")
            print(f"    Test  — MSE: {test_mse:.6f}, R2: {test_r2:.4f}")

            # Hidden states & dimensionality
            hidden = extract_hidden_states(model, data, DEVICE)
            pr, n90, n95, cumvar = compute_dimensionality(hidden)
            print(f"    Dimensionality — PR: {pr:.2f}, PCs@90%: {n90}, PCs@95%: {n95}")

            results.append({
                'session': session_num,
                'state': state,
                'phase': phase,
                'region': region,
                'n_neurons': n_neurons,
                'n_bins': n_bins,
                'train_mse': train_mse,
                'test_mse': test_mse,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'test_r2_median': np.median(test_r2_neurons),
                'best_epoch': best_epoch,
                'n_params': n_params,
                'participation_ratio': pr,
                'pcs_90': n90,
                'pcs_95': n95,
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
            }, f'data/gru_session{session_num}_{region.lower()}_model.pt')

            # Save training history
            pd.DataFrame(history).to_csv(
                f'data/gru_session{session_num}_{region.lower()}_training_history.csv', index=False)

    # =============================================================================
    # SUMMARY
    # =============================================================================

    results_df = pd.DataFrame(results)
    results_df.to_csv('data/gru_by_region_results.csv', index=False)

    print(f"\n\n{'='*90}")
    print("SUMMARY -- GRU by Region")
    print(f"{'='*90}")
    print(f"{'Sess':>5} {'State':>7} {'Phase':>13} {'Region':>7} {'Neurons':>8} "
          f"{'Test MSE':>10} {'Test R2':>9} {'PR':>8} {'PCs@90':>7} {'Epoch':>6}")
    print(f"{'-'*5} {'-'*7} {'-'*13} {'-'*7} {'-'*8} "
          f"{'-'*10} {'-'*9} {'-'*8} {'-'*7} {'-'*6}")

    for _, r in results_df.iterrows():
        print(f"{int(r['session']):>5} {r['state']:>7} {r['phase']:>13} {r['region']:>7} "
              f"{int(r['n_neurons']):>8} {r['test_mse']:>10.6f} {r['test_r2']:>9.4f} "
              f"{r['participation_ratio']:>8.2f} {int(r['pcs_90']):>7} {int(r['best_epoch']):>6}")

    # --- Statistical comparisons ---
    for region in ['LHA', 'RSP']:
        rdf = results_df[results_df['region'] == region]
        print(f"\n--- {region}: Fed vs Fasted ---")
        for metric in ['test_r2', 'participation_ratio', 'pcs_90', 'hidden_var', 'traj_speed']:
            fed = rdf[rdf['state'] == 'Fed'][metric].values
            fas = rdf[rdf['state'] == 'Fasted'][metric].values
            if len(fed) >= 2 and len(fas) >= 2:
                _, p = sp_stats.mannwhitneyu(fed, fas, alternative='two-sided')
                print(f"  {metric}: Fed={np.mean(fed):.4f} vs Fasted={np.mean(fas):.4f}, p={p:.4f}")

    # LHA vs RSP comparison (pooled across sessions)
    print(f"\n--- LHA vs RSP (all sessions) ---")
    for metric in ['test_r2', 'participation_ratio', 'pcs_90', 'hidden_var', 'traj_speed']:
        lha_vals = results_df[results_df['region'] == 'LHA'][metric].values
        rsp_vals = results_df[results_df['region'] == 'RSP'][metric].values
        if len(lha_vals) >= 2 and len(rsp_vals) >= 2:
            _, p = sp_stats.mannwhitneyu(lha_vals, rsp_vals, alternative='two-sided')
            print(f"  {metric}: LHA={np.mean(lha_vals):.4f} vs RSP={np.mean(rsp_vals):.4f}, p={p:.4f}")

    # =============================================================================
    # FIGURES
    # =============================================================================

    # --- Figure 1: Test R2 by session, region side-by-side ---
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    ax = axes[0]
    bar_width = 0.35
    sessions_nums = sorted(results_df['session'].unique())
    x_pos = np.arange(len(sessions_nums))

    for i, snum in enumerate(sessions_nums):
        for j, region in enumerate(['LHA', 'RSP']):
            row = results_df[(results_df['session'] == snum) & (results_df['region'] == region)]
            if len(row) == 0:
                continue
            r = row.iloc[0]
            color = '#2196F3' if r['state'] == 'Fed' else '#F44336'
            hatch = '' if region == 'LHA' else '///'
            offset = -bar_width/2 if region == 'LHA' else bar_width/2
            ax.bar(i + offset, r['test_r2'], bar_width, color=color, alpha=0.7,
                   edgecolor='black', linewidth=0.5, hatch=hatch)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'S{s}' for s in sessions_nums])
    ax.set_ylabel('Test R2')
    ax.set_title('Test R2 by Session & Region')
    ax.grid(True, alpha=0.3, axis='y')
    legend_elements = [
        Line2D([0], [0], color='#2196F3', linewidth=8, alpha=0.7, label='Fed LHA'),
        plt.Rectangle((0,0), 1, 1, fc='#2196F3', alpha=0.7, hatch='///', label='Fed RSP'),
        Line2D([0], [0], color='#F44336', linewidth=8, alpha=0.7, label='Fasted LHA'),
        plt.Rectangle((0,0), 1, 1, fc='#F44336', alpha=0.7, hatch='///', label='Fasted RSP'),
    ]
    ax.legend(handles=legend_elements, fontsize=8)

    # --- Panel 2: R2 Fed vs Fasted by region ---
    ax = axes[1]
    positions = [0, 1, 2.5, 3.5]
    bar_colors = ['#2196F3', '#F44336', '#2196F3', '#F44336']
    bar_labels = ['Fed', 'Fasted', 'Fed', 'Fasted']

    for idx, (region, pos_start) in enumerate([('LHA', 0), ('RSP', 2.5)]):
        rdf = results_df[results_df['region'] == region]
        fed_r2 = rdf[rdf['state'] == 'Fed']['test_r2'].values
        fas_r2 = rdf[rdf['state'] == 'Fasted']['test_r2'].values
        _, p = sp_stats.mannwhitneyu(fed_r2, fas_r2, alternative='two-sided')

        ax.bar(pos_start, np.mean(fed_r2),
               yerr=np.std(fed_r2)/np.sqrt(len(fed_r2)),
               color='#2196F3', capsize=5, alpha=0.7, width=0.7)
        ax.bar(pos_start + 1, np.mean(fas_r2),
               yerr=np.std(fas_r2)/np.sqrt(len(fas_r2)),
               color='#F44336', capsize=5, alpha=0.7, width=0.7)
        for v in fed_r2:
            ax.scatter(pos_start, v, c='#2196F3', edgecolors='black', linewidth=0.5, s=50, zorder=5)
        for v in fas_r2:
            ax.scatter(pos_start + 1, v, c='#F44336', edgecolors='black', linewidth=0.5, s=50, zorder=5)

        # p-value annotation
        ymax = max(np.max(fed_r2), np.max(fas_r2))
        ax.text(pos_start + 0.5, ymax + 0.01, f'p={p:.3f}', ha='center', fontsize=9)

    ax.set_xticks([0.5, 3.0])
    ax.set_xticklabels(['LHA', 'RSP'])
    ax.set_ylabel('Test R2')
    ax.set_title('Test R2: Fed vs Fasted by Region')
    ax.grid(True, alpha=0.3, axis='y')

    # --- Panel 3: LHA vs RSP ---
    ax = axes[2]
    lha_r2 = results_df[results_df['region'] == 'LHA']['test_r2'].values
    rsp_r2 = results_df[results_df['region'] == 'RSP']['test_r2'].values
    _, p_region = sp_stats.mannwhitneyu(lha_r2, rsp_r2, alternative='two-sided')

    ax.bar([0, 1], [np.mean(lha_r2), np.mean(rsp_r2)],
           yerr=[np.std(lha_r2)/np.sqrt(len(lha_r2)), np.std(rsp_r2)/np.sqrt(len(rsp_r2))],
           color=['#9C27B0', '#4CAF50'], capsize=5, alpha=0.7, width=0.5)
    for v in lha_r2:
        ax.scatter(0, v, c='#9C27B0', edgecolors='black', linewidth=0.5, s=60, zorder=5)
    for v in rsp_r2:
        ax.scatter(1, v, c='#4CAF50', edgecolors='black', linewidth=0.5, s=60, zorder=5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['LHA', 'RSP'])
    ax.set_ylabel('Test R2')
    ax.set_title(f'Test R2: LHA vs RSP (p={p_region:.4f})')
    ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('GRU Performance by Region', fontsize=14)
    plt.tight_layout()
    plt.savefig('figures/gru_by_region_performance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n[OK] Saved figures/gru_by_region_performance.png")

    # --- Figure 2: Dimensionality comparison ---
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Panel 1: PR by session & region
    ax = axes[0]
    for i, snum in enumerate(sessions_nums):
        for j, region in enumerate(['LHA', 'RSP']):
            row = results_df[(results_df['session'] == snum) & (results_df['region'] == region)]
            if len(row) == 0:
                continue
            r = row.iloc[0]
            color = '#2196F3' if r['state'] == 'Fed' else '#F44336'
            hatch = '' if region == 'LHA' else '///'
            offset = -bar_width/2 if region == 'LHA' else bar_width/2
            ax.bar(i + offset, r['participation_ratio'], bar_width, color=color, alpha=0.7,
                   edgecolor='black', linewidth=0.5, hatch=hatch)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'S{s}' for s in sessions_nums])
    ax.set_ylabel('Participation Ratio')
    ax.set_title('Effective Dimensionality by Session & Region')
    ax.grid(True, alpha=0.3, axis='y')

    # Panel 2: PR Fed vs Fasted by region
    ax = axes[1]
    for idx, (region, pos_start) in enumerate([('LHA', 0), ('RSP', 2.5)]):
        rdf = results_df[results_df['region'] == region]
        fed_pr = rdf[rdf['state'] == 'Fed']['participation_ratio'].values
        fas_pr = rdf[rdf['state'] == 'Fasted']['participation_ratio'].values
        _, p = sp_stats.mannwhitneyu(fed_pr, fas_pr, alternative='two-sided')

        ax.bar(pos_start, np.mean(fed_pr),
               yerr=np.std(fed_pr)/np.sqrt(len(fed_pr)),
               color='#2196F3', capsize=5, alpha=0.7, width=0.7)
        ax.bar(pos_start + 1, np.mean(fas_pr),
               yerr=np.std(fas_pr)/np.sqrt(len(fas_pr)),
               color='#F44336', capsize=5, alpha=0.7, width=0.7)
        for v in fed_pr:
            ax.scatter(pos_start, v, c='#2196F3', edgecolors='black', linewidth=0.5, s=50, zorder=5)
        for v in fas_pr:
            ax.scatter(pos_start + 1, v, c='#F44336', edgecolors='black', linewidth=0.5, s=50, zorder=5)

        ymax = max(np.max(fed_pr), np.max(fas_pr))
        ax.text(pos_start + 0.5, ymax + 0.5, f'p={p:.3f}', ha='center', fontsize=9)

    ax.set_xticks([0.5, 3.0])
    ax.set_xticklabels(['LHA', 'RSP'])
    ax.set_ylabel('Participation Ratio')
    ax.set_title('Dimensionality: Fed vs Fasted by Region')
    ax.grid(True, alpha=0.3, axis='y')

    # Panel 3: LHA vs RSP
    ax = axes[2]
    lha_pr = results_df[results_df['region'] == 'LHA']['participation_ratio'].values
    rsp_pr = results_df[results_df['region'] == 'RSP']['participation_ratio'].values
    _, p_pr_region = sp_stats.mannwhitneyu(lha_pr, rsp_pr, alternative='two-sided')

    ax.bar([0, 1], [np.mean(lha_pr), np.mean(rsp_pr)],
           yerr=[np.std(lha_pr)/np.sqrt(len(lha_pr)), np.std(rsp_pr)/np.sqrt(len(rsp_pr))],
           color=['#9C27B0', '#4CAF50'], capsize=5, alpha=0.7, width=0.5)
    for v in lha_pr:
        ax.scatter(0, v, c='#9C27B0', edgecolors='black', linewidth=0.5, s=60, zorder=5)
    for v in rsp_pr:
        ax.scatter(1, v, c='#4CAF50', edgecolors='black', linewidth=0.5, s=60, zorder=5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['LHA', 'RSP'])
    ax.set_ylabel('Participation Ratio')
    ax.set_title(f'Dimensionality: LHA vs RSP (p={p_pr_region:.4f})')
    ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('GRU Latent Dimensionality by Region', fontsize=14)
    plt.tight_layout()
    plt.savefig('figures/gru_by_region_dimensionality.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved figures/gru_by_region_dimensionality.png")

    # --- Figure 3: Training curves (2x8 grid: LHA top, RSP bottom) ---
    fig, axes = plt.subplots(2, 8, figsize=(28, 8))

    for i, (session_name, session_num, state, phase) in enumerate(session_list):
        for j, region in enumerate(['LHA', 'RSP']):
            ax = axes[j, i]
            hist_path = f'data/gru_session{session_num}_{region.lower()}_training_history.csv'
            try:
                hist = pd.read_csv(hist_path)
                ax.plot(hist['train_loss'], label='Train', color='#2196F3', linewidth=1)
                ax.plot(hist['val_loss'], label='Val', color='#F44336', linewidth=1)
            except FileNotFoundError:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax.transAxes)

            color = '#2196F3' if state == 'Fed' else '#F44336'
            ax.set_title(f'S{session_num} {region}', fontsize=9, color=color, fontweight='bold')
            if j == 1:
                ax.set_xlabel('Epoch', fontsize=8)
            if i == 0:
                ax.set_ylabel(f'{region}\nMSE Loss', fontsize=8)
            ax.legend(fontsize=6)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=7)

    plt.suptitle('GRU Training Curves by Region', fontsize=14)
    plt.tight_layout()
    plt.savefig('figures/gru_by_region_training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved figures/gru_by_region_training_curves.png")

    print(f"\n[DONE] GRU by-region analysis complete!")
