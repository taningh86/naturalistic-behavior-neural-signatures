"""
Pooled GRU with Session-Specific Input Layers — By Region (LHA / RSP)
======================================================================
Same pooled architecture as gru_pooled.py but trains separate models for
LHA-only and RSP-only neurons.

For each region x condition:
  - Condition-specific pooled model (Fed-only or Fasted-only)
  - Combined model (all 8 sessions)

Asks: is the shared dynamical structure region-specific?
Does the fasted dimensionality collapse hold when regions are separated?
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
D_SHARED = 32
HIDDEN_SIZE = 32
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
    """Bin spike trains at BIN_SIZE_MS resolution and z-score."""
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


def load_all_sessions():
    """Load binned spike data for all 8 sessions, separated by region."""
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
        lha_ids, rsp_ids = get_good_units_by_region(sorted_path)

        if len(lha_ids) < 3 or len(rsp_ids) < 3:
            print(f"  Session {sess_num}: too few units (LHA={len(lha_ids)}, RSP={len(rsp_ids)}), skipping")
            continue

        lha_data, lha_bins = bin_spike_trains(sorting, lha_ids)
        rsp_data, rsp_bins = bin_spike_trains(sorting, rsp_ids)

        sessions_data[sess_num] = {
            'lha': {'data': lha_data, 'n_neurons': len(lha_ids), 'n_bins': lha_bins},
            'rsp': {'data': rsp_data, 'n_neurons': len(rsp_ids), 'n_bins': rsp_bins},
            'state': info['state'],
            'phase': info['phase'],
        }
        print(f"  Session {sess_num}: {info['state']} {info['phase']}, "
              f"LHA={len(lha_ids)} neurons ({lha_bins} bins), "
              f"RSP={len(rsp_ids)} neurons ({rsp_bins} bins)")

    return sessions_data


# =============================================================================
# DATASET
# =============================================================================

class MultiSessionDataset(Dataset):
    """Dataset that returns sequences tagged with session ID."""

    def __init__(self, sessions_data, session_nums, region, seq_len, split='train', train_frac=0.8):
        self.seq_len = seq_len
        self.samples = []
        self.region = region

        for sn in session_nums:
            data = sessions_data[sn][region]['data']
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
        data = self.sessions_data[sn][self.region]['data']
        x = torch.tensor(data[start:start + self.seq_len], dtype=torch.float32)
        y = torch.tensor(data[start + self.seq_len], dtype=torch.float32)
        return x, y, sn


# =============================================================================
# MODEL
# =============================================================================

class PooledGRU(nn.Module):
    """GRU with session-specific input/output projection layers."""

    def __init__(self, session_neuron_counts, d_shared, hidden_size, num_layers, dropout):
        super().__init__()
        self.d_shared = d_shared
        self.hidden_size = hidden_size

        self.input_projections = nn.ModuleDict()
        for sn, n_neurons in session_neuron_counts.items():
            self.input_projections[str(sn)] = nn.Linear(n_neurons, d_shared)

        self.gru = nn.GRU(
            input_size=d_shared, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.fc_shared = nn.Linear(hidden_size, d_shared)

        self.output_projections = nn.ModuleDict()
        for sn, n_neurons in session_neuron_counts.items():
            self.output_projections[str(sn)] = nn.Linear(d_shared, n_neurons)

    def forward(self, x, session_num):
        sn_key = str(session_num)
        projected = self.input_projections[sn_key](x)
        out, _ = self.gru(projected)
        out = out[:, -1, :]
        shared_out = self.fc_shared(out)
        pred = self.output_projections[sn_key](shared_out)
        return pred

    def extract_hidden_states(self, x, session_num):
        sn_key = str(session_num)
        with torch.no_grad():
            projected = self.input_projections[sn_key](x)
            out, _ = self.gru(projected)
        return out


# =============================================================================
# COLLATE & TRAINING
# =============================================================================

def collate_by_session(batch):
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


def train_pooled_model(model, train_loader, val_loader, n_epochs, patience, lr, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    best_val_loss = np.inf
    best_epoch = 0
    best_state = None
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(n_epochs):
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

        if (epoch + 1) % 25 == 0:
            print(f"      Epoch {epoch+1}: train={mean_train:.6f} val={mean_val:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return history, best_epoch, best_val_loss


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_per_session(model, sessions_data, session_nums, region, device):
    results = {}
    criterion = nn.MSELoss()
    for sn in session_nums:
        data = sessions_data[sn][region]['data']
        T = len(data)
        split_idx = int(T * TRAIN_FRAC)
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
            ss_res = ((test_y - pred) ** 2).sum().item()
            ss_tot = ((test_y - test_y.mean(dim=0)) ** 2).sum().item()
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        results[sn] = {'test_mse': mse, 'test_r2': r2}
    return results


def extract_all_hidden_states(model, sessions_data, session_nums, region, device):
    hidden_states = {}
    for sn in session_nums:
        data = sessions_data[sn][region]['data']
        T = len(data)
        seqs = []
        for i in range(T - SEQ_LEN):
            seqs.append(data[i:i + SEQ_LEN])
        seqs_t = torch.tensor(np.array(seqs), dtype=torch.float32).to(device)
        model.eval()
        chunk_size = 512
        all_hidden = []
        for start in range(0, len(seqs_t), chunk_size):
            chunk = seqs_t[start:start + chunk_size]
            h = model.extract_hidden_states(chunk, sn)
            all_hidden.append(h[:, -1, :].cpu().numpy())
        hidden_states[sn] = np.concatenate(all_hidden, axis=0)
    return hidden_states


def compute_latent_metrics(hidden_states):
    cov = np.cov(hidden_states.T)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = eigenvalues[eigenvalues > 0]
    pr = (np.sum(eigenvalues)) ** 2 / np.sum(eigenvalues ** 2)
    variance = np.mean(np.var(hidden_states, axis=0))
    diffs = np.diff(hidden_states, axis=0)
    speed = np.mean(np.sqrt(np.sum(diffs ** 2, axis=1)))
    pca = PCA().fit(hidden_states)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    pcs90 = np.searchsorted(cumvar, 0.90) + 1
    pcs95 = np.searchsorted(cumvar, 0.95) + 1
    return {'pr': pr, 'variance': variance, 'speed': speed, 'pcs_90': pcs90, 'pcs_95': pcs95}


# =============================================================================
# MAIN
# =============================================================================

def run_pooled_for_region(sessions_data, region, fed_sessions, fasted_sessions):
    """Train pooled models for a single region."""
    results_all = []
    models = {}

    for condition, session_nums in [('Fed', fed_sessions), ('Fasted', fasted_sessions)]:
        print(f"\n  --- {region.upper()} Pooled GRU: {condition} ({len(session_nums)} sessions) ---")

        neuron_counts = {sn: sessions_data[sn][region]['n_neurons'] for sn in session_nums}
        print(f"    Neuron counts: {neuron_counts}")

        train_ds = MultiSessionDataset(sessions_data, session_nums, region, SEQ_LEN, 'train', TRAIN_FRAC)
        test_ds = MultiSessionDataset(sessions_data, session_nums, region, SEQ_LEN, 'test', TRAIN_FRAC)
        print(f"    Train: {len(train_ds)}, Test: {len(test_ds)} samples")

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                  collate_fn=collate_by_session)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                                 collate_fn=collate_by_session)

        model = PooledGRU(neuron_counts, D_SHARED, HIDDEN_SIZE, NUM_LAYERS, DROPOUT).to(DEVICE)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"    Parameters: {n_params:,}")

        t0 = time.time()
        history, best_epoch, best_val = train_pooled_model(
            model, train_loader, test_loader, NUM_EPOCHS, PATIENCE, LEARNING_RATE, DEVICE
        )
        elapsed = time.time() - t0
        print(f"    Done in {elapsed:.1f}s, best epoch: {best_epoch+1}, val loss: {best_val:.6f}")

        per_session_r2 = evaluate_per_session(model, sessions_data, session_nums, region, DEVICE)
        hidden_states = extract_all_hidden_states(model, sessions_data, session_nums, region, DEVICE)

        for sn in sorted(session_nums):
            r = per_session_r2[sn]
            metrics = compute_latent_metrics(hidden_states[sn])
            info = sessions_data[sn]
            results_all.append({
                'model_type': 'condition_specific',
                'region': region.upper(),
                'condition': condition,
                'session': sn,
                'state': info['state'],
                'phase': info['phase'],
                'n_neurons': info[region]['n_neurons'],
                'test_r2': r['test_r2'],
                'test_mse': r['test_mse'],
                'pr': metrics['pr'],
                'variance': metrics['variance'],
                'speed': metrics['speed'],
                'pcs_90': metrics['pcs_90'],
                'pcs_95': metrics['pcs_95'],
            })
            print(f"    S{sn} ({info['phase']}): R2={r['test_r2']:.4f}, PR={metrics['pr']:.2f}, "
                  f"Var={metrics['variance']:.4f}, Speed={metrics['speed']:.4f}")

        # Save model
        model_path = Path("data") / f"gru_pooled_{region}_{condition.lower()}_model.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'neuron_counts': neuron_counts,
            'config': {'d_shared': D_SHARED, 'hidden_size': HIDDEN_SIZE,
                       'num_layers': NUM_LAYERS, 'seq_len': SEQ_LEN, 'bin_size_ms': BIN_SIZE_MS},
            'history': history, 'best_epoch': best_epoch,
        }, model_path)

        hist_df = pd.DataFrame(history)
        hist_df.to_csv(Path("data") / f"gru_pooled_{region}_{condition.lower()}_history.csv", index=False)
        models[condition] = model

    # --- Combined model (all 8 sessions) ---
    all_sessions = sorted(fed_sessions + fasted_sessions)
    print(f"\n  --- {region.upper()} Combined Pooled GRU (all 8 sessions) ---")

    all_neuron_counts = {sn: sessions_data[sn][region]['n_neurons'] for sn in all_sessions}
    print(f"    Neuron counts: {all_neuron_counts}")

    train_ds_all = MultiSessionDataset(sessions_data, all_sessions, region, SEQ_LEN, 'train', TRAIN_FRAC)
    test_ds_all = MultiSessionDataset(sessions_data, all_sessions, region, SEQ_LEN, 'test', TRAIN_FRAC)
    print(f"    Train: {len(train_ds_all)}, Test: {len(test_ds_all)} samples")

    train_loader_all = DataLoader(train_ds_all, batch_size=BATCH_SIZE, shuffle=True,
                                  collate_fn=collate_by_session)
    test_loader_all = DataLoader(test_ds_all, batch_size=BATCH_SIZE, shuffle=False,
                                 collate_fn=collate_by_session)

    model_all = PooledGRU(all_neuron_counts, D_SHARED, HIDDEN_SIZE, NUM_LAYERS, DROPOUT).to(DEVICE)
    n_params_all = sum(p.numel() for p in model_all.parameters())
    print(f"    Parameters: {n_params_all:,}")

    t0 = time.time()
    history_all, best_epoch_all, best_val_all = train_pooled_model(
        model_all, train_loader_all, test_loader_all, NUM_EPOCHS, PATIENCE, LEARNING_RATE, DEVICE
    )
    elapsed = time.time() - t0
    print(f"    Done in {elapsed:.1f}s, best epoch: {best_epoch_all+1}")

    all_r2 = evaluate_per_session(model_all, sessions_data, all_sessions, region, DEVICE)
    all_hidden = extract_all_hidden_states(model_all, sessions_data, all_sessions, region, DEVICE)

    for sn in all_sessions:
        r = all_r2[sn]
        metrics = compute_latent_metrics(all_hidden[sn])
        info = sessions_data[sn]
        results_all.append({
            'model_type': 'combined',
            'region': region.upper(),
            'condition': 'All',
            'session': sn,
            'state': info['state'],
            'phase': info['phase'],
            'n_neurons': info[region]['n_neurons'],
            'test_r2': r['test_r2'],
            'test_mse': r['test_mse'],
            'pr': metrics['pr'],
            'variance': metrics['variance'],
            'speed': metrics['speed'],
            'pcs_90': metrics['pcs_90'],
            'pcs_95': metrics['pcs_95'],
        })
        print(f"    S{sn} ({info['state']} {info['phase']}): R2={r['test_r2']:.4f}, "
              f"PR={metrics['pr']:.2f}, Var={metrics['variance']:.4f}, Speed={metrics['speed']:.4f}")

    # Save combined model
    torch.save({
        'model_state_dict': model_all.state_dict(),
        'neuron_counts': all_neuron_counts,
        'config': {'d_shared': D_SHARED, 'hidden_size': HIDDEN_SIZE,
                   'num_layers': NUM_LAYERS, 'seq_len': SEQ_LEN, 'bin_size_ms': BIN_SIZE_MS},
        'history': history_all, 'best_epoch': best_epoch_all,
    }, Path("data") / f"gru_pooled_{region}_all_model.pt")

    return results_all, all_hidden, all_sessions


def main():
    print(f"Device: {DEVICE}")
    print(f"Config: {BIN_SIZE_MS}ms bins, D_shared={D_SHARED}, hidden={HIDDEN_SIZE}, "
          f"layers={NUM_LAYERS}, seq_len={SEQ_LEN}")
    print()

    print("Loading session data...")
    sessions_data = load_all_sessions()
    print(f"Loaded {len(sessions_data)} sessions\n")

    fed_sessions = [sn for sn in sessions_data if sessions_data[sn]['state'] == 'Fed']
    fasted_sessions = [sn for sn in sessions_data if sessions_data[sn]['state'] == 'Fasted']

    all_results = []

    # --- Run for each region ---
    for region in ['lha', 'rsp']:
        print(f"\n{'='*70}")
        print(f"  REGION: {region.upper()}")
        print(f"{'='*70}")

        region_results, all_hidden, all_sessions = run_pooled_for_region(
            sessions_data, region, fed_sessions, fasted_sessions
        )
        all_results.extend(region_results)

    # --- Save all results ---
    df = pd.DataFrame(all_results)
    df.to_csv(Path("data") / "gru_pooled_by_region_results.csv", index=False)
    print(f"\nResults saved: data/gru_pooled_by_region_results.csv")

    # --- Statistical comparisons ---
    print(f"\n{'='*70}")
    print("STATISTICAL COMPARISONS")
    print(f"{'='*70}")

    for model_type in ['condition_specific', 'combined']:
        mdf = df[df['model_type'] == model_type]
        model_label = "Condition-Specific" if model_type == 'condition_specific' else "Combined (All Sessions)"
        print(f"\n--- {model_label} ---")

        for region in ['LHA', 'RSP']:
            rdf = mdf[mdf['region'] == region]
            fed_df = rdf[rdf['state'] == 'Fed']
            fas_df = rdf[rdf['state'] == 'Fasted']

            if len(fed_df) == 0 or len(fas_df) == 0:
                continue

            print(f"\n  {region} — Fed vs Fasted:")
            for metric in ['test_r2', 'pr', 'variance', 'speed', 'pcs_90']:
                fed_v = fed_df[metric].values
                fas_v = fas_df[metric].values
                _, p = sp_stats.mannwhitneyu(fed_v, fas_v, alternative='two-sided')
                sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
                print(f"    {metric:12s}: Fed={np.mean(fed_v):.4f} vs Fasted={np.mean(fas_v):.4f}  "
                      f"p={p:.4f} {sig}")

    # --- Comparison: pooled vs per-session ---
    print(f"\n{'='*70}")
    print("COMPARISON: Pooled vs Per-Session R2")
    print(f"{'='*70}")

    per_session_csv = Path("data") / "gru_by_region_results.csv"
    if per_session_csv.exists():
        df_ps = pd.read_csv(per_session_csv)
        comb_df = df[df['model_type'] == 'combined']

        for region in ['LHA', 'RSP']:
            print(f"\n  {region}:")
            ps_region = df_ps[df_ps['region'] == region]
            pool_region = comb_df[comb_df['region'] == region]

            for sn in sorted(ps_region['session'].unique()):
                ps_r2 = ps_region[ps_region['session'] == sn]['test_r2'].values
                pool_r2 = pool_region[pool_region['session'] == sn]['test_r2'].values
                if len(ps_r2) > 0 and len(pool_r2) > 0:
                    change = (pool_r2[0] - ps_r2[0]) / abs(ps_r2[0]) * 100 if ps_r2[0] != 0 else 0
                    print(f"    S{int(sn)}: per-session={ps_r2[0]:.4f} -> pooled={pool_r2[0]:.4f} "
                          f"({change:+.0f}%)")

    # --- Figures ---
    print("\nGenerating figures...")

    colors_state = {'Fed': '#2196F3', 'Fasted': '#F44336'}
    colors_region = {'LHA': '#FF9800', 'RSP': '#4CAF50'}

    # Figure 1: Condition-specific pooled — Fed vs Fasted by region
    fig, axes = plt.subplots(2, 5, figsize=(24, 10))
    fig.suptitle("Pooled GRU by Region — Condition-Specific Models (Fed vs Fasted)", fontsize=14)

    cond_df = df[df['model_type'] == 'condition_specific']
    metrics_list = ['test_r2', 'pr', 'pcs_90', 'variance', 'speed']
    metric_labels = ['Test R2', 'Participation Ratio', 'PCs @ 90%', 'Hidden Variance', 'Trajectory Speed']

    for row, region in enumerate(['LHA', 'RSP']):
        rdf = cond_df[cond_df['region'] == region]
        fed_r = rdf[rdf['state'] == 'Fed']
        fas_r = rdf[rdf['state'] == 'Fasted']

        for col, (metric, label) in enumerate(zip(metrics_list, metric_labels)):
            ax = axes[row, col]
            for ci, (cond, cdf, color) in enumerate(
                [('Fed', fed_r, colors_state['Fed']), ('Fasted', fas_r, colors_state['Fasted'])]
            ):
                vals = cdf[metric].values
                ax.bar(ci, np.mean(vals), width=0.5, color=color, alpha=0.6, label=cond)
                ax.errorbar(ci, np.mean(vals), yerr=np.std(vals)/np.sqrt(len(vals)),
                            color='black', capsize=5)
                jitter = np.random.uniform(-0.1, 0.1, len(vals))
                ax.scatter([ci + j for j in jitter], vals, color=color,
                           edgecolors='black', s=50, zorder=5)

            ax.set_xticks([0, 1])
            ax.set_xticklabels(['Fed', 'Fasted'])

            # p-value
            if len(fed_r) > 0 and len(fas_r) > 0:
                _, p = sp_stats.mannwhitneyu(fed_r[metric].values, fas_r[metric].values,
                                             alternative='two-sided')
                sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
                ax.set_title(f"{region} {label}\np={p:.3f} {sig}", fontsize=10)
            else:
                ax.set_title(f"{region} {label}", fontsize=10)

            if col == 0:
                ax.set_ylabel(region)

    plt.tight_layout()
    fig.savefig(Path("figures") / "gru_pooled_by_region_condition.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: figures/gru_pooled_by_region_condition.png")

    # Figure 2: Combined model — Fed vs Fasted by region
    fig2, axes2 = plt.subplots(2, 5, figsize=(24, 10))
    fig2.suptitle("Pooled GRU by Region — Combined Model (All 8 Sessions, Shared GRU)", fontsize=14)

    comb_df = df[df['model_type'] == 'combined']

    for row, region in enumerate(['LHA', 'RSP']):
        rdf = comb_df[comb_df['region'] == region]
        fed_r = rdf[rdf['state'] == 'Fed']
        fas_r = rdf[rdf['state'] == 'Fasted']

        for col, (metric, label) in enumerate(zip(metrics_list, metric_labels)):
            ax = axes2[row, col]
            for ci, (cond, cdf, color) in enumerate(
                [('Fed', fed_r, colors_state['Fed']), ('Fasted', fas_r, colors_state['Fasted'])]
            ):
                vals = cdf[metric].values
                ax.bar(ci, np.mean(vals), width=0.5, color=color, alpha=0.6, label=cond)
                ax.errorbar(ci, np.mean(vals), yerr=np.std(vals)/np.sqrt(len(vals)),
                            color='black', capsize=5)
                jitter = np.random.uniform(-0.1, 0.1, len(vals))
                ax.scatter([ci + j for j in jitter], vals, color=color,
                           edgecolors='black', s=50, zorder=5)

            ax.set_xticks([0, 1])
            ax.set_xticklabels(['Fed', 'Fasted'])

            if len(fed_r) > 0 and len(fas_r) > 0:
                _, p = sp_stats.mannwhitneyu(fed_r[metric].values, fas_r[metric].values,
                                             alternative='two-sided')
                sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
                ax.set_title(f"{region} {label}\np={p:.3f} {sig}", fontsize=10)
            else:
                ax.set_title(f"{region} {label}", fontsize=10)

            if col == 0:
                ax.set_ylabel(region)

    plt.tight_layout()
    fig2.savefig(Path("figures") / "gru_pooled_by_region_combined.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: figures/gru_pooled_by_region_combined.png")

    # Figure 3: Pooled vs per-session R2 comparison
    if per_session_csv.exists():
        df_ps = pd.read_csv(per_session_csv)
        comb_df = df[df['model_type'] == 'combined']

        fig3, axes3 = plt.subplots(1, 2, figsize=(14, 6))
        fig3.suptitle("Pooled vs Per-Session GRU: R2 Comparison by Region", fontsize=14)

        for ri, region in enumerate(['LHA', 'RSP']):
            ax = axes3[ri]
            ps_r = df_ps[df_ps['region'] == region].sort_values('session')
            pool_r = comb_df[comb_df['region'] == region].sort_values('session')

            sessions = sorted(set(ps_r['session'].values) & set(pool_r['session'].values))
            x = np.arange(len(sessions))

            ps_vals = [ps_r[ps_r['session'] == s]['test_r2'].values[0] for s in sessions]
            pool_vals = [pool_r[pool_r['session'] == s]['test_r2'].values[0] for s in sessions]

            width = 0.35
            ax.bar(x - width/2, ps_vals, width, label='Per-Session', color='gray', alpha=0.7)
            ax.bar(x + width/2, pool_vals, width, label='Pooled', color=colors_region[region], alpha=0.7)

            # Color session labels by state
            labels = []
            label_colors = []
            for s in sessions:
                state = sessions_data[int(s)]['state']
                labels.append(f"S{int(s)}\n({state[0]})")
                label_colors.append(colors_state[state])

            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=9)
            for tick, color in zip(ax.get_xticklabels(), label_colors):
                tick.set_color(color)

            ax.set_ylabel('Test R2')
            ax.set_title(f'{region}')
            ax.legend()

        plt.tight_layout()
        fig3.savefig(Path("figures") / "gru_pooled_vs_persession_by_region.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("  Saved: figures/gru_pooled_vs_persession_by_region.png")

    # Figure 4: Latent trajectories — combined model
    fig4, axes4 = plt.subplots(2, 2, figsize=(16, 14))
    fig4.suptitle("Pooled GRU by Region — Shared Latent Trajectories (Combined Model)", fontsize=14)

    all_sessions_sorted = sorted(fed_sessions + fasted_sessions)

    for ri, region in enumerate(['lha', 'rsp']):
        model_path = Path("data") / f"gru_pooled_{region}_all_model.pt"
        if not model_path.exists():
            continue

        checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
        neuron_counts = checkpoint['neuron_counts']
        model = PooledGRU(neuron_counts, D_SHARED, HIDDEN_SIZE, NUM_LAYERS, DROPOUT).to(DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])

        hidden_states = extract_all_hidden_states(model, sessions_data, all_sessions_sorted, region, DEVICE)
        all_hs = np.concatenate([hidden_states[sn] for sn in all_sessions_sorted], axis=0)
        pca = PCA(n_components=3).fit(all_hs)

        # By state
        ax = axes4[ri, 0]
        for sn in all_sessions_sorted:
            hs_pca = pca.transform(hidden_states[sn])
            state = sessions_data[sn]['state']
            ax.plot(hs_pca[:, 0], hs_pca[:, 1], alpha=0.4, linewidth=0.5,
                    color=colors_state[state])
            ax.scatter(hs_pca[0, 0], hs_pca[0, 1], marker='o', s=60,
                       color=colors_state[state], edgecolors='black', zorder=5)
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
        ax.set_title(f'{region.upper()} — By State (Blue=Fed, Red=Fasted)')

        # By session
        ax = axes4[ri, 1]
        cmap = plt.cm.tab10
        for i, sn in enumerate(all_sessions_sorted):
            hs_pca = pca.transform(hidden_states[sn])
            state = sessions_data[sn]['state']
            phase = sessions_data[sn]['phase']
            ax.plot(hs_pca[:, 0], hs_pca[:, 1], alpha=0.5, linewidth=0.5,
                    color=cmap(i), label=f"S{sn} ({state[0]},{phase[:3]})")
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
        ax.set_title(f'{region.upper()} — By Session')
        ax.legend(fontsize=7, ncol=2)

    plt.tight_layout()
    fig4.savefig(Path("figures") / "gru_pooled_by_region_latent.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: figures/gru_pooled_by_region_latent.png")

    print("\nAll done!")


if __name__ == "__main__":
    main()
