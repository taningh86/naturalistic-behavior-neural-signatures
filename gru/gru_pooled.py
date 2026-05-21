"""
Pooled GRU with Session-Specific Input Layers — Single-Probe
=============================================================
Instead of training one GRU per session, this pools all sessions within a
condition (Fed / Fasted) into a single GRU model.

Architecture:
  Session i: (batch, seq_len, N_neurons_i)
    → Session-specific Linear(N_neurons_i → D_shared)
    → Shared GRU(D_shared → hidden_size)
    → Shared Linear(hidden_size → D_shared)
    → Session-specific Linear(D_shared → N_neurons_i)

Each session has its own input/output projection layers to handle different
neuron counts, but the GRU dynamics are shared across all sessions.

This asks: is there a common dynamical structure shared across sessions
within a metabolic state?
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

BIN_SIZE_MS = 500
FS = 30000
BIN_SAMPLES = int(BIN_SIZE_MS * FS / 1000)

SEQ_LEN = 10               # 10 steps = 5s context
D_SHARED = 32               # Shared latent space dimensionality
HIDDEN_SIZE = 32             # GRU hidden state size
NUM_LAYERS = 1
DROPOUT = 0.0
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
NUM_EPOCHS = 150
PATIENCE = 15
TRAIN_FRAC = 0.8

LHA_DEPTH_MAX = 1300
RSP_DEPTH_MIN = 1300

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open("paths.yaml") as f:
    paths_config = yaml.safe_load(f)

# Session definitions
SESSION_INFO = {
    1: {'state': 'Fed', 'phase': 'Exploration'},
    2: {'state': 'Fed', 'phase': 'Foraging'},
    3: {'state': 'Fed', 'phase': 'Exploration'},
    4: {'state': 'Fed', 'phase': 'Foraging'},
    5: {'state': 'Fasted', 'phase': 'Exploration'},
    6: {'state': 'Fasted', 'phase': 'Foraging'},
    7: {'state': 'Fasted', 'phase': 'Exploration'},
    8: {'state': 'Fasted', 'phase': 'Foraging'},
}


# =============================================================================
# DATA LOADING (reused from gru_neural_dynamics.py)
# =============================================================================

def get_good_unit_ids(sorted_path_obj):
    """Get all good unit IDs from cluster_info.tsv."""
    ci = sorted_path_obj / "cluster_info.tsv"
    if not ci.exists():
        return np.array([])
    df = pd.read_csv(ci, sep='\t')
    label_col = None
    if 'group' in df.columns and df['group'].eq('good').any():
        label_col = 'group'
    elif 'KSLabel' in df.columns:
        label_col = 'KSLabel'
    if label_col is None:
        return np.array([])
    return df[df[label_col] == 'good']['cluster_id'].values


def bin_spike_trains(sorting, unit_ids):
    """Bin spike trains at BIN_SIZE_MS resolution and z-score each neuron."""
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

    # Z-score each neuron
    means = data.mean(axis=0, keepdims=True)
    stds = data.std(axis=0, keepdims=True)
    stds[stds < 1e-8] = 1.0
    data = (data - means) / stds
    return data, n_bins


def load_all_sessions():
    """Load binned spike data for all 8 single-probe sessions."""
    sessions_data = {}
    sp = paths_config['single_probe']['coordinates_1']['mouse01']['sessions']

    for sess_num, info in SESSION_INFO.items():
        key = f"session_{sess_num}"
        sc = sp[key]
        sorted_path = Path(sc['sorted'])

        if not sorted_path.exists():
            print(f"  Session {sess_num}: sorted path not found, skipping")
            continue

        sorting = se.read_kilosort(sorted_path)
        unit_ids = get_good_unit_ids(sorted_path)

        if len(unit_ids) < 5:
            print(f"  Session {sess_num}: only {len(unit_ids)} good units, skipping")
            continue

        data, n_bins = bin_spike_trains(sorting, unit_ids)
        sessions_data[sess_num] = {
            'data': data,
            'n_neurons': len(unit_ids),
            'n_bins': n_bins,
            'state': info['state'],
            'phase': info['phase'],
            'unit_ids': unit_ids,
        }
        print(f"  Session {sess_num}: {info['state']} {info['phase']}, "
              f"{len(unit_ids)} neurons, {n_bins} bins ({n_bins * BIN_SIZE_MS / 1000:.0f}s)")

    return sessions_data


# =============================================================================
# DATASET — Multi-session
# =============================================================================

class MultiSessionDataset(Dataset):
    """Dataset that returns sequences tagged with session ID."""

    def __init__(self, sessions_data, session_nums, seq_len, split='train', train_frac=0.8):
        self.seq_len = seq_len
        self.samples = []  # list of (session_num, start_idx)

        for sn in session_nums:
            data = sessions_data[sn]['data']
            T = len(data)
            split_idx = int(T * train_frac)

            if split == 'train':
                start, end = 0, split_idx
            else:
                start, end = split_idx, T

            n_samples = end - start - seq_len
            for i in range(n_samples):
                self.samples.append((sn, start + i))

        self.sessions_data = sessions_data

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sn, start = self.samples[idx]
        data = self.sessions_data[sn]['data']
        x = torch.tensor(data[start:start + self.seq_len], dtype=torch.float32)
        y = torch.tensor(data[start + self.seq_len], dtype=torch.float32)
        return x, y, sn


# =============================================================================
# MODEL — Pooled GRU with Session-Specific Layers
# =============================================================================

class PooledGRU(nn.Module):
    """GRU with session-specific input/output projection layers."""

    def __init__(self, session_neuron_counts, d_shared, hidden_size, num_layers, dropout):
        """
        session_neuron_counts: dict {session_num: n_neurons}
        """
        super().__init__()
        self.d_shared = d_shared
        self.hidden_size = hidden_size

        # Session-specific input projections: N_neurons_i → D_shared
        self.input_projections = nn.ModuleDict()
        for sn, n_neurons in session_neuron_counts.items():
            self.input_projections[str(sn)] = nn.Linear(n_neurons, d_shared)

        # Shared GRU
        self.gru = nn.GRU(
            input_size=d_shared,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Shared hidden → shared latent decoder
        self.fc_shared = nn.Linear(hidden_size, d_shared)

        # Session-specific output projections: D_shared → N_neurons_i
        self.output_projections = nn.ModuleDict()
        for sn, n_neurons in session_neuron_counts.items():
            self.output_projections[str(sn)] = nn.Linear(d_shared, n_neurons)

    def forward(self, x, session_num):
        """
        x: (batch, seq_len, n_neurons_session)
        session_num: int — which session this batch comes from
        """
        sn_key = str(session_num)
        # Project to shared space
        projected = self.input_projections[sn_key](x)  # (batch, seq_len, d_shared)

        # Shared GRU
        out, _ = self.gru(projected)  # (batch, seq_len, hidden_size)
        out = out[:, -1, :]           # (batch, hidden_size)

        # Decode to shared latent, then to session-specific output
        shared_out = self.fc_shared(out)  # (batch, d_shared)
        pred = self.output_projections[sn_key](shared_out)  # (batch, n_neurons)
        return pred

    def extract_hidden_states(self, x, session_num):
        """Extract step-by-step hidden states for latent analysis."""
        sn_key = str(session_num)
        with torch.no_grad():
            projected = self.input_projections[sn_key](x)
            out, _ = self.gru(projected)
        return out  # (batch, seq_len, hidden_size)


# =============================================================================
# CUSTOM COLLATE — group by session within batch
# =============================================================================

def collate_by_session(batch):
    """Group batch samples by session number."""
    by_session = {}
    for x, y, sn in batch:
        if sn not in by_session:
            by_session[sn] = {'x': [], 'y': []}
        by_session[sn]['x'].append(x)
        by_session[sn]['y'].append(y)

    result = {}
    for sn, data in by_session.items():
        result[sn] = {
            'x': torch.stack(data['x']),
            'y': torch.stack(data['y']),
        }
    return result


# =============================================================================
# TRAINING
# =============================================================================

def train_pooled_model(model, train_loader, val_loader, n_epochs, patience, lr, device):
    """Train pooled GRU with early stopping."""
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
        for batch_dict in train_loader:
            total_loss = 0.0
            total_count = 0
            for sn, data in batch_dict.items():
                x = data['x'].to(device)
                y = data['y'].to(device)
                pred = model(x, sn)
                loss = criterion(pred, y)
                total_loss += loss * len(x)
                total_count += len(x)

            avg_loss = total_loss / total_count
            optimizer.zero_grad()
            avg_loss.backward()
            optimizer.step()
            train_losses.append(avg_loss.item())

        # --- Validate ---
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_dict in val_loader:
                total_loss = 0.0
                total_count = 0
                for sn, data in batch_dict.items():
                    x = data['x'].to(device)
                    y = data['y'].to(device)
                    pred = model(x, sn)
                    loss = criterion(pred, y)
                    total_loss += loss * len(x)
                    total_count += len(x)
                if total_count > 0:
                    val_losses.append((total_loss / total_count).item())

        mean_train = np.mean(train_losses)
        mean_val = np.mean(val_losses) if val_losses else np.inf
        history['train_loss'].append(mean_train)
        history['val_loss'].append(mean_val)

        if mean_val < best_val_loss:
            best_val_loss = mean_val
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch - best_epoch >= patience:
            break

        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1}: train={mean_train:.6f} val={mean_val:.6f}")

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    return history, best_epoch, best_val_loss


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_per_session(model, sessions_data, session_nums, device):
    """Evaluate pooled model on each session's test set independently."""
    results = {}
    criterion = nn.MSELoss()

    for sn in session_nums:
        data = sessions_data[sn]['data']
        T = len(data)
        split_idx = int(T * TRAIN_FRAC)

        # Build test sequences
        test_x, test_y = [], []
        for i in range(split_idx, T - SEQ_LEN):
            test_x.append(data[i:i + SEQ_LEN])
            test_y.append(data[i + SEQ_LEN])

        if len(test_x) == 0:
            continue

        test_x = torch.tensor(np.array(test_x), dtype=torch.float32).to(device)
        test_y = torch.tensor(np.array(test_y), dtype=torch.float32).to(device)

        model.eval()
        with torch.no_grad():
            pred = model(test_x, sn)
            mse = criterion(pred, test_y).item()

            # R2
            ss_res = ((test_y - pred) ** 2).sum().item()
            ss_tot = ((test_y - test_y.mean(dim=0)) ** 2).sum().item()
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        results[sn] = {'test_mse': mse, 'test_r2': r2}

    return results


def extract_all_hidden_states(model, sessions_data, session_nums, device):
    """Extract full hidden state trajectory for each session."""
    hidden_states = {}

    for sn in session_nums:
        data = sessions_data[sn]['data']
        T = len(data)

        # Build sequences covering whole session
        seqs = []
        for i in range(T - SEQ_LEN):
            seqs.append(data[i:i + SEQ_LEN])

        seqs_t = torch.tensor(np.array(seqs), dtype=torch.float32).to(device)

        model.eval()
        # Process in chunks to avoid OOM
        chunk_size = 512
        all_hidden = []
        for start in range(0, len(seqs_t), chunk_size):
            chunk = seqs_t[start:start + chunk_size]
            h = model.extract_hidden_states(chunk, sn)
            all_hidden.append(h[:, -1, :].cpu().numpy())  # last hidden state

        hidden_states[sn] = np.concatenate(all_hidden, axis=0)

    return hidden_states


def compute_latent_metrics(hidden_states):
    """Compute PR, variance, speed from hidden states."""
    cov = np.cov(hidden_states.T)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = eigenvalues[eigenvalues > 0]

    pr = (np.sum(eigenvalues)) ** 2 / np.sum(eigenvalues ** 2)
    variance = np.mean(np.var(hidden_states, axis=0))

    diffs = np.diff(hidden_states, axis=0)
    speed = np.mean(np.sqrt(np.sum(diffs ** 2, axis=1)))

    # PCs for 90%
    from sklearn.decomposition import PCA
    pca = PCA().fit(hidden_states)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    pcs90 = np.searchsorted(cumvar, 0.90) + 1
    pcs95 = np.searchsorted(cumvar, 0.95) + 1

    return {
        'pr': pr, 'variance': variance, 'speed': speed,
        'pcs_90': pcs90, 'pcs_95': pcs95,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print(f"Device: {DEVICE}")
    print(f"Config: {BIN_SIZE_MS}ms bins, D_shared={D_SHARED}, hidden={HIDDEN_SIZE}, "
          f"layers={NUM_LAYERS}, seq_len={SEQ_LEN}")
    print()

    # --- Load all sessions ---
    print("Loading session data...")
    sessions_data = load_all_sessions()
    print(f"Loaded {len(sessions_data)} sessions\n")

    fed_sessions = [sn for sn in sessions_data if sessions_data[sn]['state'] == 'Fed']
    fasted_sessions = [sn for sn in sessions_data if sessions_data[sn]['state'] == 'Fasted']

    print(f"Fed sessions: {sorted(fed_sessions)}")
    print(f"Fasted sessions: {sorted(fasted_sessions)}")

    results_all = []

    for condition, session_nums in [('Fed', fed_sessions), ('Fasted', fasted_sessions)]:
        print(f"\n{'='*60}")
        print(f"Training POOLED GRU — {condition} ({len(session_nums)} sessions)")
        print(f"{'='*60}")

        if len(session_nums) == 0:
            print("  No sessions, skipping")
            continue

        # Neuron counts per session
        neuron_counts = {sn: sessions_data[sn]['n_neurons'] for sn in session_nums}
        print(f"  Neuron counts: {neuron_counts}")

        # Create datasets
        train_ds = MultiSessionDataset(sessions_data, session_nums, SEQ_LEN, 'train', TRAIN_FRAC)
        test_ds = MultiSessionDataset(sessions_data, session_nums, SEQ_LEN, 'test', TRAIN_FRAC)
        print(f"  Train samples: {len(train_ds)}, Test samples: {len(test_ds)}")

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                  collate_fn=collate_by_session, drop_last=False)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                                 collate_fn=collate_by_session, drop_last=False)

        # Build model
        model = PooledGRU(neuron_counts, D_SHARED, HIDDEN_SIZE, NUM_LAYERS, DROPOUT).to(DEVICE)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Model parameters: {n_params:,}")

        # Train
        t0 = time.time()
        history, best_epoch, best_val = train_pooled_model(
            model, train_loader, test_loader, NUM_EPOCHS, PATIENCE, LEARNING_RATE, DEVICE
        )
        elapsed = time.time() - t0
        print(f"  Training done in {elapsed:.1f}s, best epoch: {best_epoch+1}, "
              f"best val loss: {best_val:.6f}")

        # Evaluate per session
        per_session_results = evaluate_per_session(model, sessions_data, session_nums, DEVICE)
        print(f"\n  Per-session test results:")
        for sn in sorted(session_nums):
            r = per_session_results[sn]
            print(f"    Session {sn} ({sessions_data[sn]['phase']}): "
                  f"R2={r['test_r2']:.4f}, MSE={r['test_mse']:.6f}")

        # Overall R2
        overall_r2 = np.mean([r['test_r2'] for r in per_session_results.values()])
        print(f"  Overall mean R2: {overall_r2:.4f}")

        # Extract hidden states
        hidden_states = extract_all_hidden_states(model, sessions_data, session_nums, DEVICE)

        # Compute latent metrics per session
        print(f"\n  Latent metrics from shared GRU:")
        for sn in sorted(session_nums):
            hs = hidden_states[sn]
            metrics = compute_latent_metrics(hs)
            info = sessions_data[sn]
            print(f"    Session {sn} ({info['phase']}): PR={metrics['pr']:.2f}, "
                  f"Var={metrics['variance']:.4f}, Speed={metrics['speed']:.4f}, "
                  f"PCs@90%={metrics['pcs_90']}")

            results_all.append({
                'condition': condition,
                'session': sn,
                'state': info['state'],
                'phase': info['phase'],
                'n_neurons': info['n_neurons'],
                'test_r2': per_session_results[sn]['test_r2'],
                'test_mse': per_session_results[sn]['test_mse'],
                'pr': metrics['pr'],
                'variance': metrics['variance'],
                'speed': metrics['speed'],
                'pcs_90': metrics['pcs_90'],
                'pcs_95': metrics['pcs_95'],
            })

        # Save model
        model_path = Path("data") / f"gru_pooled_{condition.lower()}_model.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'neuron_counts': neuron_counts,
            'config': {
                'd_shared': D_SHARED, 'hidden_size': HIDDEN_SIZE,
                'num_layers': NUM_LAYERS, 'seq_len': SEQ_LEN,
                'bin_size_ms': BIN_SIZE_MS,
            },
            'history': history,
            'best_epoch': best_epoch,
        }, model_path)
        print(f"  Model saved: {model_path}")

        # Save training history
        hist_df = pd.DataFrame(history)
        hist_df.to_csv(Path("data") / f"gru_pooled_{condition.lower()}_history.csv", index=False)

    # --- Results table ---
    if not results_all:
        print("\nNo results to save.")
        return

    df = pd.DataFrame(results_all)
    df.to_csv(Path("data") / "gru_pooled_results.csv", index=False)
    print(f"\nResults saved: data/gru_pooled_results.csv")

    # --- Comparison: pooled vs per-session ---
    print("\n" + "=" * 60)
    print("COMPARISON: Pooled GRU vs Per-Session GRU")
    print("=" * 60)

    # Load per-session results for comparison
    per_session_csv = Path("data") / "gru_session_results.csv"
    if per_session_csv.exists():
        df_ps = pd.read_csv(per_session_csv)
        print(f"\nPer-session model R2 (from gru_session_results.csv):")
        for _, row in df_ps.iterrows():
            print(f"  Session {int(row['session'])}: R2={row['test_r2']:.4f}")

    print(f"\nPooled model R2:")
    for _, row in df.iterrows():
        print(f"  Session {int(row['session'])}: R2={row['test_r2']:.4f}")

    # --- Statistical comparison ---
    from scipy import stats

    fed_df = df[df['condition'] == 'Fed']
    fas_df = df[df['condition'] == 'Fasted']

    if len(fed_df) > 0 and len(fas_df) > 0:
        print(f"\n{'='*60}")
        print("FED vs FASTED — Pooled GRU Metrics")
        print(f"{'='*60}")

        for metric in ['test_r2', 'pr', 'variance', 'speed', 'pcs_90']:
            fed_vals = fed_df[metric].values
            fas_vals = fas_df[metric].values
            u_stat, p_val = stats.mannwhitneyu(fed_vals, fas_vals, alternative='two-sided')
            print(f"  {metric:12s}: Fed={np.mean(fed_vals):.4f} vs Fasted={np.mean(fas_vals):.4f}  "
                  f"p={p_val:.4f} {'*' if p_val < 0.05 else 'ns'}")

    # --- Figures ---
    print("\nGenerating figures...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Pooled GRU — Session-Specific Input Layers, Shared Dynamics", fontsize=14)

    metrics_to_plot = [
        ('test_r2', 'Test R²', axes[0, 0]),
        ('pr', 'Participation Ratio', axes[0, 1]),
        ('pcs_90', 'PCs for 90% Variance', axes[0, 2]),
        ('variance', 'Hidden Variance', axes[1, 0]),
        ('speed', 'Trajectory Speed', axes[1, 1]),
    ]

    colors = {'Fed': '#2196F3', 'Fasted': '#F44336'}

    for metric, label, ax in metrics_to_plot:
        for cond in ['Fed', 'Fasted']:
            cdf = df[df['condition'] == cond]
            vals = cdf[metric].values
            x_pos = 0 if cond == 'Fed' else 1
            ax.bar(x_pos, np.mean(vals), width=0.5, color=colors[cond],
                   alpha=0.6, label=cond)
            ax.errorbar(x_pos, np.mean(vals), yerr=np.std(vals) / np.sqrt(len(vals)),
                        color='black', capsize=5)
            # Individual points
            jitter = np.random.uniform(-0.1, 0.1, len(vals))
            ax.scatter([x_pos + j for j in jitter], vals, color=colors[cond],
                       edgecolors='black', s=60, zorder=5)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Fed', 'Fasted'])
        ax.set_ylabel(label)
        ax.set_title(label)

        # Add p-value
        if len(fed_df) > 0 and len(fas_df) > 0:
            fed_v = fed_df[metric].values
            fas_v = fas_df[metric].values
            _, p = stats.mannwhitneyu(fed_v, fas_v, alternative='two-sided')
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
            ax.set_title(f"{label}\np={p:.3f} {sig}")

    # Training curves
    ax_tc = axes[1, 2]
    for cond in ['Fed', 'Fasted']:
        hist_path = Path("data") / f"gru_pooled_{cond.lower()}_history.csv"
        if hist_path.exists():
            h = pd.read_csv(hist_path)
            ax_tc.plot(h['train_loss'], label=f'{cond} train', linestyle='-',
                       color=colors[cond], alpha=0.5)
            ax_tc.plot(h['val_loss'], label=f'{cond} val', linestyle='--',
                       color=colors[cond])
    ax_tc.set_xlabel('Epoch')
    ax_tc.set_ylabel('MSE Loss')
    ax_tc.set_title('Training Curves')
    ax_tc.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(Path("figures") / "gru_pooled_overview.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: figures/gru_pooled_overview.png")

    # --- Figure 2: Latent trajectories from shared GRU ---
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))
    fig2.suptitle("Pooled GRU — Shared Latent Trajectories (PCA of Hidden States)", fontsize=13)

    from sklearn.decomposition import PCA

    for idx, (cond, session_nums) in enumerate([('Fed', sorted(fed_sessions)),
                                                  ('Fasted', sorted(fasted_sessions))]):
        ax = axes2[idx]
        # Reload hidden states from the saved model
        model_path = Path("data") / f"gru_pooled_{cond.lower()}_model.pt"
        if not model_path.exists():
            continue

        checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
        neuron_counts = checkpoint['neuron_counts']
        model = PooledGRU(neuron_counts, D_SHARED, HIDDEN_SIZE, NUM_LAYERS, DROPOUT).to(DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])

        hidden_states = extract_all_hidden_states(model, sessions_data, session_nums, DEVICE)

        # Concatenate for joint PCA
        all_hs = np.concatenate([hidden_states[sn] for sn in session_nums], axis=0)
        pca = PCA(n_components=3).fit(all_hs)

        cmap = plt.cm.tab10
        for i, sn in enumerate(session_nums):
            hs_pca = pca.transform(hidden_states[sn])
            phase = sessions_data[sn]['phase']
            ax.plot(hs_pca[:, 0], hs_pca[:, 1], alpha=0.5, linewidth=0.5,
                    color=cmap(i), label=f"S{sn} ({phase})")
            ax.scatter(hs_pca[0, 0], hs_pca[0, 1], marker='o', s=80,
                       color=cmap(i), edgecolors='black', zorder=5)

        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
        ax.set_title(f'{cond} — Shared GRU Latent Trajectories')
        ax.legend(fontsize=8)

    plt.tight_layout()
    fig2.savefig(Path("figures") / "gru_pooled_latent_trajectories.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: figures/gru_pooled_latent_trajectories.png")

    # --- Figure 3: Overlay fed vs fasted in shared space ---
    # Train an "all sessions" model for this comparison
    print(f"\n{'='*60}")
    print("Training ALL-SESSIONS pooled GRU (Fed + Fasted together)")
    print(f"{'='*60}")

    all_sessions = sorted(fed_sessions + fasted_sessions)
    all_neuron_counts = {sn: sessions_data[sn]['n_neurons'] for sn in all_sessions}
    print(f"  All sessions: {all_sessions}")
    print(f"  Neuron counts: {all_neuron_counts}")

    train_ds_all = MultiSessionDataset(sessions_data, all_sessions, SEQ_LEN, 'train', TRAIN_FRAC)
    test_ds_all = MultiSessionDataset(sessions_data, all_sessions, SEQ_LEN, 'test', TRAIN_FRAC)
    print(f"  Train samples: {len(train_ds_all)}, Test samples: {len(test_ds_all)}")

    train_loader_all = DataLoader(train_ds_all, batch_size=BATCH_SIZE, shuffle=True,
                                  collate_fn=collate_by_session, drop_last=False)
    test_loader_all = DataLoader(test_ds_all, batch_size=BATCH_SIZE, shuffle=False,
                                 collate_fn=collate_by_session, drop_last=False)

    model_all = PooledGRU(all_neuron_counts, D_SHARED, HIDDEN_SIZE, NUM_LAYERS, DROPOUT).to(DEVICE)
    n_params_all = sum(p.numel() for p in model_all.parameters())
    print(f"  Model parameters: {n_params_all:,}")

    t0 = time.time()
    history_all, best_epoch_all, best_val_all = train_pooled_model(
        model_all, train_loader_all, test_loader_all, NUM_EPOCHS, PATIENCE, LEARNING_RATE, DEVICE
    )
    elapsed = time.time() - t0
    print(f"  Training done in {elapsed:.1f}s, best epoch: {best_epoch_all+1}")

    # Per-session R2 from combined model
    all_results = evaluate_per_session(model_all, sessions_data, all_sessions, DEVICE)
    print(f"\n  Per-session R2 (combined model):")
    for sn in all_sessions:
        r = all_results[sn]
        state = sessions_data[sn]['state']
        phase = sessions_data[sn]['phase']
        print(f"    Session {sn} ({state} {phase}): R2={r['test_r2']:.4f}")

    # Hidden states from combined model
    hidden_all = extract_all_hidden_states(model_all, sessions_data, all_sessions, DEVICE)

    # PCA on all hidden states together
    all_hs_combined = np.concatenate([hidden_all[sn] for sn in all_sessions], axis=0)
    pca_all = PCA(n_components=3).fit(all_hs_combined)

    fig3, axes3 = plt.subplots(1, 3, figsize=(20, 6))
    fig3.suptitle("All-Sessions Pooled GRU — Shared Latent Space", fontsize=14)

    # Panel 1: All trajectories colored by state
    ax = axes3[0]
    for sn in all_sessions:
        hs_pca = pca_all.transform(hidden_all[sn])
        state = sessions_data[sn]['state']
        color = colors[state]
        ax.plot(hs_pca[:, 0], hs_pca[:, 1], alpha=0.4, linewidth=0.5, color=color)
        ax.scatter(hs_pca[0, 0], hs_pca[0, 1], marker='o', s=60,
                   color=color, edgecolors='black', zorder=5)
    ax.set_xlabel(f'PC1 ({pca_all.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca_all.explained_variance_ratio_[1]*100:.1f}%)')
    ax.set_title('All Sessions (Blue=Fed, Red=Fasted)')

    # Panel 2: Fed only
    ax = axes3[1]
    for i, sn in enumerate(sorted(fed_sessions)):
        hs_pca = pca_all.transform(hidden_all[sn])
        phase = sessions_data[sn]['phase']
        ax.plot(hs_pca[:, 0], hs_pca[:, 1], alpha=0.5, linewidth=0.5,
                color=plt.cm.Blues(0.4 + i * 0.15), label=f"S{sn} ({phase})")
    ax.set_xlabel(f'PC1')
    ax.set_ylabel(f'PC2')
    ax.set_title('Fed Sessions')
    ax.legend(fontsize=8)

    # Panel 3: Fasted only
    ax = axes3[2]
    for i, sn in enumerate(sorted(fasted_sessions)):
        hs_pca = pca_all.transform(hidden_all[sn])
        phase = sessions_data[sn]['phase']
        ax.plot(hs_pca[:, 0], hs_pca[:, 1], alpha=0.5, linewidth=0.5,
                color=plt.cm.Reds(0.4 + i * 0.15), label=f"S{sn} ({phase})")
    ax.set_xlabel(f'PC1')
    ax.set_ylabel(f'PC2')
    ax.set_title('Fasted Sessions')
    ax.legend(fontsize=8)

    plt.tight_layout()
    fig3.savefig(Path("figures") / "gru_pooled_all_sessions_latent.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: figures/gru_pooled_all_sessions_latent.png")

    # Save combined model
    torch.save({
        'model_state_dict': model_all.state_dict(),
        'neuron_counts': all_neuron_counts,
        'config': {
            'd_shared': D_SHARED, 'hidden_size': HIDDEN_SIZE,
            'num_layers': NUM_LAYERS, 'seq_len': SEQ_LEN,
            'bin_size_ms': BIN_SIZE_MS,
        },
        'history': history_all,
        'best_epoch': best_epoch_all,
    }, Path("data") / "gru_pooled_all_model.pt")

    # Latent metrics from combined model
    combined_results = []
    print(f"\n  Latent metrics from combined model:")
    for sn in all_sessions:
        hs = hidden_all[sn]
        metrics = compute_latent_metrics(hs)
        state = sessions_data[sn]['state']
        phase = sessions_data[sn]['phase']
        r2 = all_results[sn]['test_r2']
        print(f"    S{sn} ({state} {phase}): R2={r2:.4f}, PR={metrics['pr']:.2f}, "
              f"Var={metrics['variance']:.4f}, Speed={metrics['speed']:.4f}")
        combined_results.append({
            'model': 'combined', 'session': sn, 'state': state, 'phase': phase,
            'n_neurons': sessions_data[sn]['n_neurons'],
            'test_r2': r2, 'pr': metrics['pr'], 'variance': metrics['variance'],
            'speed': metrics['speed'], 'pcs_90': metrics['pcs_90'],
        })

    # Stats on combined model
    comb_df = pd.DataFrame(combined_results)
    fed_c = comb_df[comb_df['state'] == 'Fed']
    fas_c = comb_df[comb_df['state'] == 'Fasted']

    print(f"\n  Combined model — Fed vs Fasted:")
    for metric in ['test_r2', 'pr', 'variance', 'speed', 'pcs_90']:
        fed_v = fed_c[metric].values
        fas_v = fas_c[metric].values
        _, p = stats.mannwhitneyu(fed_v, fas_v, alternative='two-sided')
        print(f"    {metric:12s}: Fed={np.mean(fed_v):.4f} vs Fasted={np.mean(fas_v):.4f}  "
              f"p={p:.4f} {'*' if p < 0.05 else 'ns'}")

    # Save combined results
    comb_df.to_csv(Path("data") / "gru_pooled_combined_results.csv", index=False)

    print("\nAll done!")


if __name__ == "__main__":
    main()
