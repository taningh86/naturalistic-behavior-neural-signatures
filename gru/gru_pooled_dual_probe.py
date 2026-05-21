"""
Pooled GRU with Session-Specific Input Layers — Dual-Probe (ACA & LHA)
========================================================================
Pools sessions by condition (Fed / Fasted / HFD) for each region.
Same architecture as gru_pooled_by_region.py but for dual-probe data.

Models trained:
  - Per-condition: Fed-only, Fasted-only, HFD-only (for each region)
  - Combined: All sessions shared GRU (for each region)

Generates latent structure figures (eigenvalue spectrum, cumulative variance,
speed over time, variance per dimension, metric bar plots with p-values).
"""

import yaml
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
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

LHA_DEPTH_MIN = 0
LHA_DEPTH_MAX = 345

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open("paths.yaml") as f:
    paths_config = yaml.safe_load(f)

STATE_MAP = {'fed': 'Fed', 'fasted': 'Fasted', 'fed-HFD': 'HFD'}


# =============================================================================
# UNIT SELECTION (from gru_dual_probe.py)
# =============================================================================

def get_good_unit_ids(sorted_path_obj):
    """All good units — for ACA probe-0."""
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
    """Compute depth from templates.npy + channel_positions.npy."""
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
    return {i: depths[i] for i in range(len(depths))}


def get_good_lha_unit_ids(sorted_path_obj):
    """Good units in LHA depth range (0-345um) — for probe-1."""
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
    # Fallback
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
    """Load binned spike data for all dual-probe sessions, per region."""
    dp_config = paths_config["double_probe"]["coordinates_1"]["mouse01"]["sessions"]
    sessions_data = {}

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

        p0_sp = Path(p0_path)
        p1_sp = Path(p1_path)

        if not p0_sp.exists() or not p1_sp.exists():
            print(f"  Session {i}: path not found, skipping")
            continue

        # ACA
        sorting_aca = se.read_kilosort(p0_sp)
        aca_ids = get_good_unit_ids(p0_sp)
        avail_aca = set(sorting_aca.get_unit_ids())
        aca_ids = np.array([u for u in aca_ids if u in avail_aca])

        # LHA
        sorting_lha = se.read_kilosort(p1_sp)
        lha_ids = get_good_lha_unit_ids(p1_sp)
        avail_lha = set(sorting_lha.get_unit_ids())
        lha_ids = np.array([u for u in lha_ids if u in avail_lha])

        if len(aca_ids) < 3 or len(lha_ids) < 3:
            print(f"  Session {i}: too few units (ACA={len(aca_ids)}, LHA={len(lha_ids)}), skipping")
            continue

        aca_data, aca_bins = bin_spike_trains(sorting_aca, aca_ids)
        lha_data, lha_bins = bin_spike_trains(sorting_lha, lha_ids)

        sessions_data[i] = {
            'aca': {'data': aca_data, 'n_neurons': len(aca_ids), 'n_bins': aca_bins},
            'lha': {'data': lha_data, 'n_neurons': len(lha_ids), 'n_bins': lha_bins},
            'state': state, 'phase': phase,
        }
        print(f"  Session {i}: {state} {phase}, ACA={len(aca_ids)} ({aca_bins} bins), "
              f"LHA={len(lha_ids)} ({lha_bins} bins)")

    return sessions_data


# =============================================================================
# DATASET & MODEL
# =============================================================================

class MultiSessionDataset(Dataset):
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
            for i in range(end - start - seq_len):
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


class PooledGRU(nn.Module):
    def __init__(self, session_neuron_counts, d_shared, hidden_size, num_layers, dropout):
        super().__init__()
        self.d_shared = d_shared
        self.hidden_size = hidden_size
        self.input_projections = nn.ModuleDict()
        for sn, n_neurons in session_neuron_counts.items():
            self.input_projections[str(sn)] = nn.Linear(n_neurons, d_shared)
        self.gru = nn.GRU(input_size=d_shared, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0.0)
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
        return self.output_projections[sn_key](shared_out)

    def extract_hidden_states(self, x, session_num):
        sn_key = str(session_num)
        with torch.no_grad():
            projected = self.input_projections[sn_key](x)
            out, _ = self.gru(projected)
        return out


def collate_by_session(batch):
    by_session = {}
    for x, y, sn in batch:
        if sn not in by_session:
            by_session[sn] = {'x': [], 'y': []}
        by_session[sn]['x'].append(x)
        by_session[sn]['y'].append(y)
    return {sn: {'x': torch.stack(d['x']), 'y': torch.stack(d['y'])} for sn, d in by_session.items()}


# =============================================================================
# TRAINING & EVALUATION
# =============================================================================

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
            total_loss, total_count = 0.0, 0
            for sn, data in batch_dict.items():
                x, y = data['x'].to(device), data['y'].to(device)
                loss = criterion(model(x, sn), y)
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
                total_loss, total_count = 0.0, 0
                for sn, data in batch_dict.items():
                    x, y = data['x'].to(device), data['y'].to(device)
                    loss = criterion(model(x, sn), y)
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


def evaluate_per_session(model, sessions_data, session_nums, region, device):
    results = {}
    criterion = nn.MSELoss()
    for sn in session_nums:
        data = sessions_data[sn][region]['data']
        T = len(data)
        split_idx = int(T * TRAIN_FRAC)
        test_x = [data[i:i + SEQ_LEN] for i in range(split_idx, T - SEQ_LEN)]
        test_y = [data[i + SEQ_LEN] for i in range(split_idx, T - SEQ_LEN)]
        if not test_x:
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
        seqs = [data[i:i + SEQ_LEN] for i in range(len(data) - SEQ_LEN)]
        seqs_t = torch.tensor(np.array(seqs), dtype=torch.float32).to(device)
        model.eval()
        all_hidden = []
        for start in range(0, len(seqs_t), 512):
            h = model.extract_hidden_states(seqs_t[start:start + 512], sn)
            all_hidden.append(h[:, -1, :].cpu().numpy())
        hidden_states[sn] = np.concatenate(all_hidden, axis=0)
    return hidden_states


def compute_latent_metrics(hs):
    cov = np.cov(hs.T)
    evals = np.linalg.eigvalsh(cov)
    evals = evals[evals > 0]
    pr = (np.sum(evals)) ** 2 / np.sum(evals ** 2)
    variance = np.mean(np.var(hs, axis=0))
    diffs = np.diff(hs, axis=0)
    speed = np.mean(np.sqrt(np.sum(diffs ** 2, axis=1)))
    pca = PCA().fit(hs)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    pcs90 = np.searchsorted(cumvar, 0.90) + 1
    return {'pr': pr, 'variance': variance, 'speed': speed, 'pcs_90': pcs90}


def add_significance_bracket(ax, x1, x2, y, p, h=0.02):
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color='black', linewidth=1.2)
    ax.text((x1 + x2) / 2, y + h, f'p={p:.3f} {sig}',
            ha='center', va='bottom', fontsize=9, fontweight='bold')


# =============================================================================
# TRAIN AND EVALUATE FOR ONE REGION
# =============================================================================

def run_pooled_for_region(sessions_data, region, condition_groups):
    """Train pooled models for a single region across conditions."""
    results_all = []

    # Per-condition models
    for condition, session_nums in condition_groups.items():
        if len(session_nums) == 0:
            continue
        print(f"\n  --- {region.upper()} Pooled GRU: {condition} ({len(session_nums)} sessions) ---")
        neuron_counts = {sn: sessions_data[sn][region]['n_neurons'] for sn in session_nums}
        print(f"    Neuron counts: {neuron_counts}")

        train_ds = MultiSessionDataset(sessions_data, session_nums, region, SEQ_LEN, 'train', TRAIN_FRAC)
        test_ds = MultiSessionDataset(sessions_data, session_nums, region, SEQ_LEN, 'test', TRAIN_FRAC)
        print(f"    Train: {len(train_ds)}, Test: {len(test_ds)}")

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_by_session)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_by_session)

        model = PooledGRU(neuron_counts, D_SHARED, HIDDEN_SIZE, NUM_LAYERS, DROPOUT).to(DEVICE)
        print(f"    Parameters: {sum(p.numel() for p in model.parameters()):,}")

        t0 = time.time()
        history, best_epoch, best_val = train_pooled_model(
            model, train_loader, test_loader, NUM_EPOCHS, PATIENCE, LEARNING_RATE, DEVICE)
        print(f"    Done in {time.time()-t0:.1f}s, best epoch: {best_epoch+1}")

        per_session_r2 = evaluate_per_session(model, sessions_data, session_nums, region, DEVICE)
        hidden_states = extract_all_hidden_states(model, sessions_data, session_nums, region, DEVICE)

        for sn in sorted(session_nums):
            r = per_session_r2.get(sn, {'test_r2': np.nan, 'test_mse': np.nan})
            metrics = compute_latent_metrics(hidden_states[sn])
            info = sessions_data[sn]
            results_all.append({
                'model_type': 'condition_specific', 'region': region.upper(),
                'condition': condition, 'session': sn, 'state': info['state'],
                'phase': info['phase'], 'n_neurons': info[region]['n_neurons'],
                'test_r2': r['test_r2'], 'pr': metrics['pr'],
                'variance': metrics['variance'], 'speed': metrics['speed'],
                'pcs_90': metrics['pcs_90'],
            })
            print(f"    S{sn} ({info['phase']}): R2={r['test_r2']:.4f}, PR={metrics['pr']:.2f}, "
                  f"Var={metrics['variance']:.4f}, Spd={metrics['speed']:.4f}")

        torch.save({
            'model_state_dict': model.state_dict(), 'neuron_counts': neuron_counts,
            'config': {'d_shared': D_SHARED, 'hidden_size': HIDDEN_SIZE,
                       'num_layers': NUM_LAYERS, 'seq_len': SEQ_LEN, 'bin_size_ms': BIN_SIZE_MS},
            'history': history, 'best_epoch': best_epoch,
        }, Path("data") / f"gru_pooled_dp_{region}_{condition.lower()}_model.pt")

    # Combined model (all sessions)
    all_session_nums = []
    for sns in condition_groups.values():
        all_session_nums.extend(sns)
    all_session_nums = sorted(all_session_nums)

    print(f"\n  --- {region.upper()} Combined Pooled GRU ({len(all_session_nums)} sessions) ---")
    all_neuron_counts = {sn: sessions_data[sn][region]['n_neurons'] for sn in all_session_nums}
    print(f"    Sessions: {all_session_nums}")

    train_ds = MultiSessionDataset(sessions_data, all_session_nums, region, SEQ_LEN, 'train', TRAIN_FRAC)
    test_ds = MultiSessionDataset(sessions_data, all_session_nums, region, SEQ_LEN, 'test', TRAIN_FRAC)
    print(f"    Train: {len(train_ds)}, Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_by_session)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_by_session)

    model_all = PooledGRU(all_neuron_counts, D_SHARED, HIDDEN_SIZE, NUM_LAYERS, DROPOUT).to(DEVICE)
    print(f"    Parameters: {sum(p.numel() for p in model_all.parameters()):,}")

    t0 = time.time()
    history_all, best_epoch_all, _ = train_pooled_model(
        model_all, train_loader, test_loader, NUM_EPOCHS, PATIENCE, LEARNING_RATE, DEVICE)
    print(f"    Done in {time.time()-t0:.1f}s, best epoch: {best_epoch_all+1}")

    all_r2 = evaluate_per_session(model_all, sessions_data, all_session_nums, region, DEVICE)
    all_hidden = extract_all_hidden_states(model_all, sessions_data, all_session_nums, region, DEVICE)

    for sn in all_session_nums:
        r = all_r2.get(sn, {'test_r2': np.nan, 'test_mse': np.nan})
        metrics = compute_latent_metrics(all_hidden[sn])
        info = sessions_data[sn]
        results_all.append({
            'model_type': 'combined', 'region': region.upper(),
            'condition': 'All', 'session': sn, 'state': info['state'],
            'phase': info['phase'], 'n_neurons': info[region]['n_neurons'],
            'test_r2': r['test_r2'], 'pr': metrics['pr'],
            'variance': metrics['variance'], 'speed': metrics['speed'],
            'pcs_90': metrics['pcs_90'],
        })
        print(f"    S{sn} ({info['state']} {info['phase']}): R2={r['test_r2']:.4f}, "
              f"PR={metrics['pr']:.2f}, Var={metrics['variance']:.4f}, Spd={metrics['speed']:.4f}")

    torch.save({
        'model_state_dict': model_all.state_dict(), 'neuron_counts': all_neuron_counts,
        'config': {'d_shared': D_SHARED, 'hidden_size': HIDDEN_SIZE,
                   'num_layers': NUM_LAYERS, 'seq_len': SEQ_LEN, 'bin_size_ms': BIN_SIZE_MS},
        'history': history_all, 'best_epoch': best_epoch_all,
    }, Path("data") / f"gru_pooled_dp_{region}_all_model.pt")

    return results_all, all_hidden, all_session_nums


# =============================================================================
# FIGURE GENERATION
# =============================================================================

def generate_latent_structure_figure(sessions_data, region, condition_groups,
                                     model_path, output_path):
    """Generate 3-row latent structure figure for one region."""
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    neuron_counts = checkpoint['neuron_counts']
    model = PooledGRU(neuron_counts, D_SHARED, HIDDEN_SIZE, NUM_LAYERS, DROPOUT).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])

    all_sessions = sorted(neuron_counts.keys())
    hidden_states = extract_all_hidden_states(model, sessions_data, all_sessions, region, DEVICE)

    # Compute per-session metrics
    session_metrics = {}
    for sn in all_sessions:
        hs = hidden_states[sn]
        cov = np.cov(hs.T)
        evals = np.sort(np.linalg.eigvalsh(cov))[::-1]
        evals_pos = evals[evals > 0]
        pr = (np.sum(evals_pos)) ** 2 / np.sum(evals_pos ** 2)
        diffs = np.diff(hs, axis=0)
        speed = np.mean(np.sqrt(np.sum(diffs ** 2, axis=1)))
        variance = np.mean(np.var(hs, axis=0))
        pca = PCA().fit(hs)
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        pcs90 = np.searchsorted(cumvar, 0.90) + 1
        session_metrics[sn] = {
            'pr': pr, 'speed': speed, 'variance': variance,
            'pcs90': pcs90, 'evals_norm': evals / evals.sum(), 'cumvar': cumvar,
        }

    colors = {'Fed': '#2196F3', 'Fasted': '#F44336', 'HFD': '#FF9800'}
    conditions = [c for c in ['Fed', 'Fasted', 'HFD'] if c in condition_groups and len(condition_groups[c]) > 0]

    fig = plt.figure(figsize=(20, 18))
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 1], hspace=0.35, wspace=0.4)
    fig.suptitle(f"{region.upper()} Pooled GRU (Dual-Probe) -- Latent Structure",
                 fontsize=16, fontweight='bold')

    ax_eig = fig.add_subplot(gs[0, 0:2])
    ax_cum = fig.add_subplot(gs[0, 2:4])
    ax_spd = fig.add_subplot(gs[1, 0:2])
    ax_var = fig.add_subplot(gs[1, 2:4])
    bar_axes = [fig.add_subplot(gs[2, i]) for i in range(4)]

    # Row 1: Eigenvalue spectrum
    for cond in conditions:
        sns = condition_groups[cond]
        all_evals = [session_metrics[sn]['evals_norm'] for sn in sns]
        for ev in all_evals:
            ax_eig.plot(range(1, len(ev)+1), ev, alpha=0.15, color=colors[cond], linewidth=0.7)
        mean_ev = np.mean(all_evals, axis=0)
        sem_ev = np.std(all_evals, axis=0) / np.sqrt(len(sns))
        x = np.arange(1, len(mean_ev)+1)
        ax_eig.plot(x, mean_ev, color=colors[cond], linewidth=2.5, label=cond)
        ax_eig.fill_between(x, mean_ev-sem_ev, mean_ev+sem_ev, color=colors[cond], alpha=0.12)
    ax_eig.set_xlabel('Principal Component', fontsize=12)
    ax_eig.set_ylabel('Normalized Eigenvalue', fontsize=12)
    ax_eig.set_title('Eigenvalue Spectrum', fontsize=13)
    ax_eig.legend(fontsize=11)
    ax_eig.set_xlim(1, 32)

    # Row 1: Cumulative variance
    for cond in conditions:
        sns = condition_groups[cond]
        all_cv = [session_metrics[sn]['cumvar'] for sn in sns]
        pcs90_vals = [session_metrics[sn]['pcs90'] for sn in sns]
        for cv in all_cv:
            ax_cum.plot(range(1, len(cv)+1), cv, alpha=0.15, color=colors[cond], linewidth=0.7)
        mean_cv = np.mean(all_cv, axis=0)
        sem_cv = np.std(all_cv, axis=0) / np.sqrt(len(sns))
        x = np.arange(1, len(mean_cv)+1)
        ax_cum.plot(x, mean_cv, color=colors[cond], linewidth=2.5,
                    label=f'{cond} (PCs@90%={np.mean(pcs90_vals):.1f})')
        ax_cum.fill_between(x, mean_cv-sem_cv, mean_cv+sem_cv, color=colors[cond], alpha=0.12)
        ax_cum.axvline(np.mean(pcs90_vals), color=colors[cond], linestyle='--', alpha=0.5, linewidth=1.5)
    ax_cum.axhline(0.90, color='gray', linestyle='--', alpha=0.4)
    ax_cum.text(31, 0.895, '90%', ha='right', va='top', color='gray', fontsize=10)
    ax_cum.set_xlabel('Number of PCs', fontsize=12)
    ax_cum.set_ylabel('Cumulative Variance Explained', fontsize=12)
    ax_cum.set_title('Cumulative Variance', fontsize=13)
    ax_cum.legend(fontsize=9)
    ax_cum.set_xlim(1, 32)
    ax_cum.set_ylim(0, 1.02)

    # Row 2: Speed over time
    n_interp = 500
    for cond in conditions:
        sns = condition_groups[cond]
        all_speeds = []
        for sn in sns:
            hs = hidden_states[sn]
            diffs = np.diff(hs, axis=0)
            speed = np.sqrt(np.sum(diffs**2, axis=1))
            kernel = np.ones(60) / 60
            speed_smooth = np.convolve(speed, kernel, mode='valid')
            t_norm = np.linspace(0, 1, len(speed_smooth))
            all_speeds.append(np.interp(np.linspace(0, 1, n_interp), t_norm, speed_smooth))
        mean_s = np.mean(all_speeds, axis=0)
        sem_s = np.std(all_speeds, axis=0) / np.sqrt(len(sns))
        t_min = np.linspace(0, 30, n_interp)
        ax_spd.plot(t_min, mean_s, color=colors[cond], linewidth=2.5, label=cond)
        ax_spd.fill_between(t_min, mean_s-sem_s, mean_s+sem_s, color=colors[cond], alpha=0.12)
    ax_spd.set_xlabel('Time (minutes)', fontsize=12)
    ax_spd.set_ylabel('Trajectory Speed (32D)', fontsize=12)
    ax_spd.set_title('Speed Over Time (30s smoothing)', fontsize=13)
    ax_spd.legend(fontsize=11)

    # Row 2: Variance per dimension
    for cond in conditions:
        sns = condition_groups[cond]
        all_v = []
        for sn in sns:
            v = np.sort(np.var(hidden_states[sn], axis=0))[::-1]
            all_v.append(v)
        mean_v = np.mean(all_v, axis=0)
        sem_v = np.std(all_v, axis=0) / np.sqrt(len(sns))
        x = np.arange(1, len(mean_v)+1)
        ax_var.plot(x, mean_v, color=colors[cond], linewidth=2.5, label=cond)
        ax_var.fill_between(x, mean_v-sem_v, mean_v+sem_v, color=colors[cond], alpha=0.12)
    ax_var.set_xlabel('Hidden Dimension (sorted)', fontsize=12)
    ax_var.set_ylabel('Variance', fontsize=12)
    ax_var.set_title('Variance per Hidden Dimension', fontsize=13)
    ax_var.legend(fontsize=11)
    ax_var.set_xlim(1, 32)

    # Row 3: Bar plots — PR, PCs@90%, Speed, Variance
    metrics_bar = [
        ('Participation\nRatio', 'pr'),
        ('PCs for\n90% Variance', 'pcs90'),
        ('Trajectory\nSpeed', 'speed'),
        ('Hidden\nVariance', 'variance'),
    ]

    for ax, (title, metric_key) in zip(bar_axes, metrics_bar):
        cond_vals = {}
        for cond in conditions:
            cond_vals[cond] = [session_metrics[sn][metric_key] for sn in condition_groups[cond]]

        n_conds = len(conditions)
        bar_width = 0.6 / n_conds if n_conds > 2 else 0.4
        x_positions = np.arange(n_conds)

        for ci, cond in enumerate(conditions):
            vals = cond_vals[cond]
            ax.bar(ci, np.mean(vals), width=bar_width * 1.3, color=colors[cond],
                   alpha=0.6, edgecolor='black', linewidth=0.8)
            ax.errorbar(ci, np.mean(vals), yerr=np.std(vals)/np.sqrt(len(vals)),
                        color='black', capsize=5, linewidth=1.5)
            jitter = np.random.uniform(-0.12, 0.12, len(vals))
            ax.scatter([ci + j for j in jitter], vals, color=colors[cond],
                       edgecolors='black', s=60, zorder=5, linewidth=0.8)

        ax.set_xticks(range(n_conds))
        ax.set_xticklabels(conditions, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')

        # Pairwise significance brackets
        if n_conds >= 2:
            pairs = [(0, 1)]
            if n_conds == 3:
                pairs = [(0, 1), (0, 2), (1, 2)]

            all_vals_flat = [v for vals in cond_vals.values() for v in vals]
            y_base = max(all_vals_flat) * 1.08
            y_step = max(all_vals_flat) * 0.08

            for pi, (i1, i2) in enumerate(pairs):
                c1, c2 = conditions[i1], conditions[i2]
                _, p = sp_stats.mannwhitneyu(cond_vals[c1], cond_vals[c2], alternative='two-sided')
                y = y_base + pi * y_step
                add_significance_bracket(ax, i1, i2, y, p, h=y_step * 0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print(f"Device: {DEVICE}")
    print(f"Config: {BIN_SIZE_MS}ms bins, D_shared={D_SHARED}, hidden={HIDDEN_SIZE}")
    print()

    print("Loading dual-probe session data...")
    sessions_data = load_all_sessions()
    print(f"Loaded {len(sessions_data)} sessions\n")

    # Group by condition
    condition_groups = {'Fed': [], 'Fasted': [], 'HFD': []}
    for sn, info in sessions_data.items():
        if info['state'] in condition_groups:
            condition_groups[info['state']].append(sn)
    for c in condition_groups:
        condition_groups[c] = sorted(condition_groups[c])

    print(f"Fed: {condition_groups['Fed']}")
    print(f"Fasted: {condition_groups['Fasted']}")
    print(f"HFD: {condition_groups['HFD']}")

    all_results = []

    for region in ['aca', 'lha']:
        print(f"\n{'='*70}")
        print(f"  REGION: {region.upper()}")
        print(f"{'='*70}")

        region_results, all_hidden, all_session_nums = run_pooled_for_region(
            sessions_data, region, condition_groups)
        all_results.extend(region_results)

    # Save results
    df = pd.DataFrame(all_results)
    df.to_csv(Path("data") / "gru_pooled_dp_results.csv", index=False)
    print(f"\nResults saved: data/gru_pooled_dp_results.csv")

    # Statistics
    print(f"\n{'='*70}")
    print("STATISTICAL COMPARISONS — Combined Model")
    print(f"{'='*70}")

    comb_df = df[df['model_type'] == 'combined']

    for region in ['ACA', 'LHA']:
        rdf = comb_df[comb_df['region'] == region]
        print(f"\n  {region}:")

        conditions = ['Fed', 'Fasted', 'HFD']
        for metric in ['test_r2', 'pr', 'variance', 'speed', 'pcs_90']:
            vals = {}
            for c in conditions:
                v = rdf[rdf['state'] == c][metric].values
                if len(v) > 0:
                    vals[c] = v

            # Print means
            means_str = " | ".join([f"{c}={np.mean(v):.4f}" for c, v in vals.items()])
            print(f"    {metric:12s}: {means_str}")

            # Pairwise
            pairs_tested = []
            for i, c1 in enumerate(conditions):
                for c2 in conditions[i+1:]:
                    if c1 in vals and c2 in vals and len(vals[c1]) > 1 and len(vals[c2]) > 1:
                        _, p = sp_stats.mannwhitneyu(vals[c1], vals[c2], alternative='two-sided')
                        sig = '*' if p < 0.05 else 'ns'
                        pairs_tested.append(f"{c1[:3]}v{c2[:3]} p={p:.4f}{sig}")
            if pairs_tested:
                print(f"{'':16s} {' | '.join(pairs_tested)}")

            # 3-way KW
            active = [v for v in vals.values() if len(v) > 1]
            if len(active) >= 3:
                kw_stat, kw_p = sp_stats.kruskal(*active)
                sig = '*' if kw_p < 0.05 else 'ns'
                print(f"{'':16s} KW p={kw_p:.4f} {sig}")

    # Generate figures
    print(f"\n{'='*70}")
    print("GENERATING FIGURES")
    print(f"{'='*70}")

    for region in ['aca', 'lha']:
        model_path = Path("data") / f"gru_pooled_dp_{region}_all_model.pt"
        if model_path.exists():
            generate_latent_structure_figure(
                sessions_data, region, condition_groups, model_path,
                Path("figures") / f"gru_pooled_dp_{region}_latent_structure.png"
            )

    print("\nAll done!")


if __name__ == "__main__":
    main()
