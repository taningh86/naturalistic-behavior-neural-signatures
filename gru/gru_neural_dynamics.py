"""
GRU Neural Dynamics — Single-Probe Next-Step Prediction
========================================================
Per-session GRU trained to predict population activity at t+1 from activity up to t.
100ms bins, all good units (LHA + RSP) per session.
Fed sessions: 1-4 | Fasted sessions: 5-8
Odd = Exploration | Even = Foraging

Compares prediction performance (MSE, R²) across metabolic states and behavioral phases.
"""

import yaml
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import spikeinterface.extractors as se
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import warnings
import time

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

BIN_SIZE_MS = 500          # 500ms bins
FS = 30000                 # Neuropixels sampling rate
BIN_SAMPLES = int(BIN_SIZE_MS * FS / 1000)  # 15000 samples per bin

SEQ_LEN = 10              # 10 time steps = 5 seconds of context
HIDDEN_SIZE = 32           # GRU hidden state size
NUM_LAYERS = 1             # GRU layers
DROPOUT = 0.0              # No dropout with single layer
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
NUM_EPOCHS = 100
PATIENCE = 10              # Early stopping patience
TRAIN_FRAC = 0.8           # 80% train, 20% test (temporal split)

LHA_DEPTH_MAX = 1300       # µm — LHA is below this
RSP_DEPTH_MIN = 1300       # µm — RSP is at or above this

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


def bin_spike_trains_100ms(sorting, unit_ids):
    """Bin spike trains at 100ms resolution and z-score each neuron."""
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

    # Build (n_bins, n_neurons) matrix
    data = np.zeros((n_bins, len(unit_ids)), dtype=np.float32)
    for i, uid in enumerate(unit_ids):
        st = spike_trains[uid]
        if len(st) > 0:
            b = ((st - all_min) // BIN_SAMPLES).astype(int)
            b = b[(b >= 0) & (b < n_bins)]
            np.add.at(data[:, i], b, 1)

    # Z-score each neuron independently
    means = data.mean(axis=0, keepdims=True)
    stds = data.std(axis=0, keepdims=True)
    stds[stds < 1e-8] = 1.0
    data = (data - means) / stds

    return data, n_bins, rec_duration_s


# =============================================================================
# DATASET
# =============================================================================

class NeuralSequenceDataset(Dataset):
    """Sliding window dataset for next-step prediction."""

    def __init__(self, data, seq_len):
        """
        data: (T, N_neurons) numpy array
        Creates pairs: X = data[t : t+seq_len], Y = data[t+seq_len]
        """
        self.data = torch.tensor(data, dtype=torch.float32)
        self.seq_len = seq_len
        self.n_samples = len(data) - seq_len

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]     # (seq_len, n_neurons)
        y = self.data[idx + self.seq_len]            # (n_neurons,)
        return x, y


# =============================================================================
# MODEL
# =============================================================================

class NeuralGRU(nn.Module):
    """GRU for next-step neural activity prediction."""

    def __init__(self, n_neurons, hidden_size, num_layers, dropout):
        super().__init__()
        self.gru = nn.GRU(
            input_size=n_neurons,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_size, n_neurons)

    def forward(self, x):
        # x: (batch, seq_len, n_neurons)
        out, _ = self.gru(x)            # (batch, seq_len, hidden_size)
        out = out[:, -1, :]             # last time step: (batch, hidden_size)
        out = self.fc(out)              # (batch, n_neurons)
        return out


# =============================================================================
# TRAINING
# =============================================================================

def train_model(model, train_loader, val_loader, n_epochs, patience, lr, device):
    """Train GRU with early stopping. Returns training history."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val_loss = np.inf
    best_epoch = 0
    best_state = None
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(n_epochs):
        # --- Train ---
        model.train()
        train_losses = []
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            pred = model(x_batch)
            loss = criterion(pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # --- Validate ---
        model.eval()
        val_losses = []
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                pred = model(x_batch)
                loss = criterion(pred, y_batch)
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        # --- Early stopping ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= patience:
            print(f"      Early stopping at epoch {epoch+1} (best: {best_epoch+1})")
            break

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"      Epoch {epoch+1:>3}: train={train_loss:.6f}, val={val_loss:.6f}")

    # Restore best model
    model.load_state_dict(best_state)
    return model, history, best_epoch + 1, best_val_loss


def evaluate_model(model, data_loader, device):
    """Evaluate model and compute MSE, R² per neuron and overall."""
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            x_batch = x_batch.to(device)
            pred = model(x_batch)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y_batch.numpy())

    preds = np.concatenate(all_preds, axis=0)     # (N_samples, N_neurons)
    targets = np.concatenate(all_targets, axis=0)

    # Overall MSE
    mse = np.mean((preds - targets) ** 2)

    # Per-neuron R²
    ss_res = np.sum((targets - preds) ** 2, axis=0)
    ss_tot = np.sum((targets - targets.mean(axis=0, keepdims=True)) ** 2, axis=0)
    r2_per_neuron = 1 - ss_res / (ss_tot + 1e-10)

    # Overall R²
    ss_res_total = np.sum(ss_res)
    ss_tot_total = np.sum(ss_tot)
    r2_overall = 1 - ss_res_total / (ss_tot_total + 1e-10)

    return mse, r2_overall, r2_per_neuron, preds, targets


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    print(f"Bin size: {BIN_SIZE_MS}ms | Seq length: {SEQ_LEN} steps ({SEQ_LEN * BIN_SIZE_MS / 1000:.1f}s)")
    print(f"GRU: hidden={HIDDEN_SIZE}, layers={NUM_LAYERS}, dropout={DROPOUT}")
    print(f"Training: epochs={NUM_EPOCHS}, patience={PATIENCE}, lr={LEARNING_RATE}, batch={BATCH_SIZE}")

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
        print(f"\n{'='*70}")
        print(f"Session {session_num}: {state} / {phase}")
        print(f"{'='*70}")

        session_config = sessions[session_name]
        sorted_path = session_config.get("sorted")
        if sorted_path is None:
            print("  [SKIP] No sorted path")
            continue

        sp = Path(sorted_path)

        # --- Load units ---
        lha_ids, rsp_ids = get_good_units_by_region(sp)
        all_ids = np.concatenate([lha_ids, rsp_ids])
        n_lha = len(lha_ids)
        n_rsp = len(rsp_ids)
        print(f"  Units: {n_lha} LHA + {n_rsp} RSP = {len(all_ids)} total")

        if len(all_ids) < 3:
            print("  [SKIP] Too few units")
            continue

        # --- Load sorting and bin ---
        sorting = se.read_kilosort(sp)
        avail = set(sorting.get_unit_ids())
        all_ids = np.array([u for u in all_ids if u in avail])
        n_neurons = len(all_ids)
        print(f"  Available in sorting: {n_neurons} units")

        t0 = time.time()
        data, n_bins, rec_dur = bin_spike_trains_100ms(sorting, all_ids)
        print(f"  Binned: {n_bins} bins ({rec_dur:.0f}s recording) in {time.time()-t0:.1f}s")

        # --- Temporal train/test split ---
        split_idx = int(n_bins * TRAIN_FRAC)
        train_data = data[:split_idx]
        test_data = data[split_idx:]
        print(f"  Train: {split_idx} bins ({split_idx * BIN_SIZE_MS / 1000:.0f}s) | "
              f"Test: {n_bins - split_idx} bins ({(n_bins - split_idx) * BIN_SIZE_MS / 1000:.0f}s)")

        # --- Create datasets ---
        train_dataset = NeuralSequenceDataset(train_data, SEQ_LEN)
        test_dataset = NeuralSequenceDataset(test_data, SEQ_LEN)

        if len(train_dataset) < BATCH_SIZE or len(test_dataset) < 10:
            print("  [SKIP] Not enough data for training")
            continue

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        print(f"  Train samples: {len(train_dataset)} | Test samples: {len(test_dataset)}")

        # --- Build and train model ---
        model = NeuralGRU(n_neurons, HIDDEN_SIZE, NUM_LAYERS, DROPOUT).to(DEVICE)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Model: {n_params:,} parameters")
        print(f"  Training...")

        t0 = time.time()
        model, history, best_epoch, best_val_loss = train_model(
            model, train_loader, test_loader, NUM_EPOCHS, PATIENCE, LEARNING_RATE, DEVICE
        )
        train_time = time.time() - t0
        print(f"  Trained in {train_time:.1f}s (best epoch: {best_epoch})")

        # --- Evaluate ---
        train_mse, train_r2, train_r2_neurons, _, _ = evaluate_model(model, train_loader, DEVICE)
        test_mse, test_r2, test_r2_neurons, preds, targets = evaluate_model(model, test_loader, DEVICE)

        print(f"  Train — MSE: {train_mse:.6f}, R²: {train_r2:.4f}")
        print(f"  Test  — MSE: {test_mse:.6f}, R²: {test_r2:.4f}")
        print(f"  Test R² per neuron: min={test_r2_neurons.min():.4f}, "
              f"median={np.median(test_r2_neurons):.4f}, max={test_r2_neurons.max():.4f}")

        # --- Store results ---
        results.append({
            'session': session_num,
            'state': state,
            'phase': phase,
            'n_lha': n_lha,
            'n_rsp': n_rsp,
            'n_neurons': n_neurons,
            'n_bins': n_bins,
            'rec_duration_s': rec_dur,
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'test_r2_median_neuron': np.median(test_r2_neurons),
            'test_r2_min_neuron': test_r2_neurons.min(),
            'test_r2_max_neuron': test_r2_neurons.max(),
            'best_epoch': best_epoch,
            'train_time_s': train_time,
            'n_params': n_params,
        })

        # --- Save per-neuron R² ---
        neuron_df = pd.DataFrame({
            'unit_id': all_ids,
            'region': ['LHA'] * n_lha + ['RSP'] * n_rsp,
            'train_r2': train_r2_neurons,
            'test_r2': test_r2_neurons,
        })
        neuron_df.to_csv(f'data/gru_session{session_num}_neuron_r2.csv', index=False)

        # --- Save training history ---
        hist_df = pd.DataFrame(history)
        hist_df.to_csv(f'data/gru_session{session_num}_training_history.csv', index=False)

        # --- Save model ---
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': {
                'n_neurons': n_neurons,
                'hidden_size': HIDDEN_SIZE,
                'num_layers': NUM_LAYERS,
                'dropout': DROPOUT,
                'seq_len': SEQ_LEN,
                'bin_size_ms': BIN_SIZE_MS,
            },
            'unit_ids': all_ids.tolist(),
            'session': session_num,
            'state': state,
            'phase': phase,
        }, f'data/gru_session{session_num}_model.pt')

    # =============================================================================
    # SUMMARY
    # =============================================================================

    from scipy import stats as sp_stats

    results_df = pd.DataFrame(results)
    results_df.to_csv('data/gru_session_results.csv', index=False)

    print(f"\n\n{'='*70}")
    print("SUMMARY — GRU Next-Step Prediction Performance")
    print(f"{'='*70}")
    print(f"{'Session':>8} {'State':>7} {'Phase':>13} {'Neurons':>8} "
          f"{'Train MSE':>10} {'Test MSE':>10} {'Train R²':>9} {'Test R²':>9} "
          f"{'Med R²':>7} {'Epochs':>7}")
    print(f"{'-'*8} {'-'*7} {'-'*13} {'-'*8} "
          f"{'-'*10} {'-'*10} {'-'*9} {'-'*9} {'-'*7} {'-'*7}")

    for _, r in results_df.iterrows():
        print(f"{int(r['session']):>8} {r['state']:>7} {r['phase']:>13} {int(r['n_neurons']):>8} "
              f"{r['train_mse']:>10.6f} {r['test_mse']:>10.6f} {r['train_r2']:>9.4f} {r['test_r2']:>9.4f} "
              f"{r['test_r2_median_neuron']:>7.4f} {int(r['best_epoch']):>7}")

    # --- Compare by state ---
    print(f"\n--- State Comparison (Fed vs Fasted) ---")
    for metric in ['test_mse', 'test_r2', 'test_r2_median_neuron']:
        fed = results_df[results_df['state'] == 'Fed'][metric].values
        fasted = results_df[results_df['state'] == 'Fasted'][metric].values
        if len(fed) >= 2 and len(fasted) >= 2:
            stat, p = sp_stats.mannwhitneyu(fed, fasted, alternative='two-sided') if len(fed) >= 2 else (0, 1)
            print(f"  {metric}: Fed={np.mean(fed):.4f}±{np.std(fed):.4f}, "
                  f"Fasted={np.mean(fasted):.4f}±{np.std(fasted):.4f}, p={p:.4f}")

    # --- Compare by phase ---
    print(f"\n--- Phase Comparison (Exploration vs Foraging) ---")
    for metric in ['test_mse', 'test_r2', 'test_r2_median_neuron']:
        exp = results_df[results_df['phase'] == 'Exploration'][metric].values
        fora = results_df[results_df['phase'] == 'Foraging'][metric].values
        if len(exp) >= 2 and len(fora) >= 2:
            stat, p = sp_stats.mannwhitneyu(exp, fora, alternative='two-sided') if len(exp) >= 2 else (0, 1)
            print(f"  {metric}: Exp={np.mean(exp):.4f}±{np.std(exp):.4f}, "
                  f"For={np.mean(fora):.4f}±{np.std(fora):.4f}, p={p:.4f}")

    # =============================================================================
    # FIGURES
    # =============================================================================

    from scipy import stats as sp_stats

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- Panel 1: Test R² by session ---
    ax = axes[0]
    colors = {'Fed': '#2196F3', 'Fasted': '#F44336'}
    markers = {'Exploration': 'o', 'Foraging': 's'}
    for _, r in results_df.iterrows():
        ax.scatter(r['session'], r['test_r2'],
                   c=colors[r['state']], marker=markers[r['phase']], s=100, zorder=5,
                   edgecolors='black', linewidth=0.5)
    ax.set_xlabel('Session')
    ax.set_ylabel('Test R²')
    ax.set_title('GRU Test R² by Session')
    ax.set_xticks(results_df['session'].values)
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2196F3', markersize=10, label='Fed Exp'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#2196F3', markersize=10, label='Fed For'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#F44336', markersize=10, label='Fasted Exp'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#F44336', markersize=10, label='Fasted For'),
    ]
    ax.legend(handles=legend_elements, fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Panel 2: Test R² by state ---
    ax = axes[1]
    fed_r2 = results_df[results_df['state'] == 'Fed']['test_r2'].values
    fasted_r2 = results_df[results_df['state'] == 'Fasted']['test_r2'].values
    bp = ax.bar([0, 1], [np.mean(fed_r2), np.mean(fasted_r2)],
                yerr=[np.std(fed_r2)/np.sqrt(len(fed_r2)), np.std(fasted_r2)/np.sqrt(len(fasted_r2))],
                color=['#2196F3', '#F44336'], capsize=5, alpha=0.7, width=0.5)
    for v in fed_r2:
        ax.scatter(0, v, c='#2196F3', edgecolors='black', linewidth=0.5, s=60, zorder=5)
    for v in fasted_r2:
        ax.scatter(1, v, c='#F44336', edgecolors='black', linewidth=0.5, s=60, zorder=5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Fed', 'Fasted'])
    ax.set_ylabel('Test R²')
    ax.set_title('Test R² by Metabolic State')
    ax.grid(True, alpha=0.3, axis='y')

    # --- Panel 3: Test R² by phase ---
    ax = axes[2]
    exp_r2 = results_df[results_df['phase'] == 'Exploration']['test_r2'].values
    for_r2 = results_df[results_df['phase'] == 'Foraging']['test_r2'].values
    ax.bar([0, 1], [np.mean(exp_r2), np.mean(for_r2)],
           yerr=[np.std(exp_r2)/np.sqrt(len(exp_r2)), np.std(for_r2)/np.sqrt(len(for_r2))],
           color=['#4CAF50', '#FF9800'], capsize=5, alpha=0.7, width=0.5)
    for v in exp_r2:
        ax.scatter(0, v, c='#4CAF50', edgecolors='black', linewidth=0.5, s=60, zorder=5)
    for v in for_r2:
        ax.scatter(1, v, c='#FF9800', edgecolors='black', linewidth=0.5, s=60, zorder=5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Exploration', 'Foraging'])
    ax.set_ylabel('Test R²')
    ax.set_title('Test R² by Behavioral Phase')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('figures/gru_performance_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n[OK] Saved figures/gru_performance_summary.png")

    # --- Training curves ---
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    for i, (_, r) in enumerate(results_df.iterrows()):
        ax = axes[i // 4, i % 4]
        hist = pd.read_csv(f'data/gru_session{int(r["session"])}_training_history.csv')
        ax.plot(hist['train_loss'], label='Train', color='#2196F3')
        ax.plot(hist['val_loss'], label='Val', color='#F44336')
        ax.set_title(f'S{int(r["session"])} {r["state"]}/{r["phase"]}', fontsize=10)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MSE Loss')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.suptitle('GRU Training Curves', fontsize=14)
    plt.tight_layout()
    plt.savefig('figures/gru_training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved figures/gru_training_curves.png")

    print(f"\n[DONE] GRU neural dynamics analysis complete!")
