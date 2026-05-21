"""
GRU Baseline Controls — Verify Training Quality
=================================================
Tests whether the pooled GRU learns meaningful temporal structure by comparing
against baselines:

1. Persistence baseline: y_hat = y(t) (predict last observed time step)
2. Mean baseline: y_hat = 0 (session mean, since z-scored)
3. Shuffle control: Train GRU on temporally shuffled data
4. Prediction visualization: actual vs predicted for example neurons

Runs on both single-probe (LHA/RSP) and dual-probe (ACA/LHA) pooled models.
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
import copy

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

# Single-probe depth
SP_LHA_DEPTH_MAX = 1300
SP_RSP_DEPTH_MIN = 1300

# Dual-probe depth
DP_LHA_DEPTH_MIN = 0
DP_LHA_DEPTH_MAX = 345

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open("paths.yaml") as f:
    paths_config = yaml.safe_load(f)

STATE_MAP = {'fed': 'Fed', 'fasted': 'Fasted', 'fed-HFD': 'HFD'}


# =============================================================================
# DATA LOADING — Single-Probe
# =============================================================================

SP_SESSION_INFO = {
    1: {'state': 'Fed', 'phase': 'Exploration'},
    2: {'state': 'Fed', 'phase': 'Foraging'},
    3: {'state': 'Fed', 'phase': 'Exploration'},
    4: {'state': 'Fed', 'phase': 'Foraging'},
    5: {'state': 'Fasted', 'phase': 'Exploration'},
    6: {'state': 'Fasted', 'phase': 'Foraging'},
    7: {'state': 'Fasted', 'phase': 'Exploration'},
    8: {'state': 'Fasted', 'phase': 'Foraging'},
}


def get_good_units_by_region_sp(sorted_path_obj):
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
    lha = good[good['depth'] < SP_LHA_DEPTH_MAX]['cluster_id'].values
    rsp = good[good['depth'] >= SP_RSP_DEPTH_MIN]['cluster_id'].values
    return lha, rsp


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


def load_single_probe_sessions():
    sessions_data = {}
    sp = paths_config['single_probe']['coordinates_1']['mouse01']['sessions']
    for sess_num, info in SP_SESSION_INFO.items():
        key = f"session_{sess_num}"
        sc = sp[key]
        sorted_path = Path(sc['sorted'])
        if not sorted_path.exists():
            continue
        sorting = se.read_kilosort(sorted_path)
        lha_ids, rsp_ids = get_good_units_by_region_sp(sorted_path)
        if len(lha_ids) < 3 or len(rsp_ids) < 3:
            continue
        lha_data, lha_bins = bin_spike_trains(sorting, lha_ids)
        rsp_data, rsp_bins = bin_spike_trains(sorting, rsp_ids)
        sessions_data[sess_num] = {
            'lha': {'data': lha_data, 'n_neurons': len(lha_ids)},
            'rsp': {'data': rsp_data, 'n_neurons': len(rsp_ids)},
            'state': info['state'], 'phase': info['phase'],
        }
    return sessions_data


# =============================================================================
# DATA LOADING — Dual-Probe
# =============================================================================

def get_good_unit_ids_dp(sorted_path_obj):
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
            t4c = spike_templates[mask]
            if len(t4c) > 0:
                mct = np.bincount(t4c).argmax()
                if mct < len(depths):
                    cluster_depths[cid] = depths[mct]
        return cluster_depths
    return {i: depths[i] for i in range(len(depths))}


def get_good_lha_unit_ids_dp(sorted_path_obj):
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
                              (df['depth'] >= DP_LHA_DEPTH_MIN) &
                              (df['depth'] <= DP_LHA_DEPTH_MAX)]
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
    return np.array([cid for cid in good_ids
                     if cid in cluster_depths and DP_LHA_DEPTH_MIN <= cluster_depths[cid] <= DP_LHA_DEPTH_MAX])


def load_dual_probe_sessions():
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
        p0_sp, p1_sp = Path(p0_path), Path(p1_path)
        if not p0_sp.exists() or not p1_sp.exists():
            continue
        sorting_aca = se.read_kilosort(p0_sp)
        aca_ids = get_good_unit_ids_dp(p0_sp)
        avail_aca = set(sorting_aca.get_unit_ids())
        aca_ids = np.array([u for u in aca_ids if u in avail_aca])
        sorting_lha = se.read_kilosort(p1_sp)
        lha_ids = get_good_lha_unit_ids_dp(p1_sp)
        avail_lha = set(sorting_lha.get_unit_ids())
        lha_ids = np.array([u for u in lha_ids if u in avail_lha])
        if len(aca_ids) < 3 or len(lha_ids) < 3:
            continue
        aca_data, aca_bins = bin_spike_trains(sorting_aca, aca_ids)
        lha_data, lha_bins = bin_spike_trains(sorting_lha, lha_ids)
        sessions_data[i] = {
            'aca': {'data': aca_data, 'n_neurons': len(aca_ids)},
            'lha': {'data': lha_data, 'n_neurons': len(lha_ids)},
            'state': state, 'phase': phase,
        }
    return sessions_data


# =============================================================================
# POOLED GRU MODEL + DATASET
# =============================================================================

class MultiSessionDataset(Dataset):
    def __init__(self, sessions_data, session_nums, region, seq_len, split='train',
                 train_frac=0.8, shuffle_time=False):
        self.seq_len = seq_len
        self.samples = []
        self.region = region
        self.sessions_data = sessions_data
        self.shuffle_time = shuffle_time
        self.shuffled_data = {}

        for sn in session_nums:
            data = sessions_data[sn][region]['data']
            if shuffle_time:
                # Shuffle time axis independently for each neuron
                data_shuffled = data.copy()
                for col in range(data_shuffled.shape[1]):
                    np.random.shuffle(data_shuffled[:, col])
                self.shuffled_data[sn] = data_shuffled
                data = data_shuffled

            T = len(data)
            split_idx = int(T * train_frac)
            if split == 'train':
                start, end = 0, split_idx
            else:
                start, end = split_idx, T
            for i in range(end - start - seq_len):
                self.samples.append((sn, start + i))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sn, start = self.samples[idx]
        if self.shuffle_time and sn in self.shuffled_data:
            data = self.shuffled_data[sn]
        else:
            data = self.sessions_data[sn][self.region]['data']
        x = torch.tensor(data[start:start + self.seq_len], dtype=torch.float32)
        y = torch.tensor(data[start + self.seq_len], dtype=torch.float32)
        return x, y, sn


class PooledGRU(nn.Module):
    def __init__(self, session_neuron_counts, d_shared, hidden_size, num_layers, dropout):
        super().__init__()
        self.input_projections = nn.ModuleDict()
        for sn, n in session_neuron_counts.items():
            self.input_projections[str(sn)] = nn.Linear(n, d_shared)
        self.gru = nn.GRU(input_size=d_shared, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0.0)
        self.fc_shared = nn.Linear(hidden_size, d_shared)
        self.output_projections = nn.ModuleDict()
        for sn, n in session_neuron_counts.items():
            self.output_projections[str(sn)] = nn.Linear(d_shared, n)

    def forward(self, x, session_num):
        sn_key = str(session_num)
        projected = self.input_projections[sn_key](x)
        out, _ = self.gru(projected)
        out = out[:, -1, :]
        shared_out = self.fc_shared(out)
        return self.output_projections[sn_key](shared_out)


def collate_by_session(batch):
    by_session = {}
    for x, y, sn in batch:
        if sn not in by_session:
            by_session[sn] = {'x': [], 'y': []}
        by_session[sn]['x'].append(x)
        by_session[sn]['y'].append(y)
    return {sn: {'x': torch.stack(d['x']), 'y': torch.stack(d['y'])} for sn, d in by_session.items()}


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
    if best_state is not None:
        model.load_state_dict(best_state)
    return history, best_epoch, best_val_loss


# =============================================================================
# BASELINE COMPUTATIONS
# =============================================================================

def compute_baselines(sessions_data, session_nums, region):
    """Compute persistence and mean baseline R2 for each session."""
    results = {}
    for sn in session_nums:
        data = sessions_data[sn][region]['data']
        T = len(data)
        split_idx = int(T * TRAIN_FRAC)

        # Test targets: data[split_idx + SEQ_LEN : T]
        test_y = data[split_idx + SEQ_LEN:]
        if len(test_y) < 10:
            continue

        # Persistence: y_hat = y(t) = data[split_idx + SEQ_LEN - 1 : T - 1]
        persist_pred = data[split_idx + SEQ_LEN - 1: T - 1]
        # Ensure same length
        min_len = min(len(test_y), len(persist_pred))
        test_y_trim = test_y[:min_len]
        persist_pred = persist_pred[:min_len]

        ss_res_persist = np.sum((test_y_trim - persist_pred) ** 2)
        ss_tot = np.sum((test_y_trim - test_y_trim.mean(axis=0)) ** 2)
        r2_persist = 1 - ss_res_persist / ss_tot if ss_tot > 0 else 0.0

        # Mean baseline: y_hat = 0 (z-scored data)
        ss_res_mean = np.sum(test_y_trim ** 2)
        r2_mean = 1 - ss_res_mean / ss_tot if ss_tot > 0 else 0.0

        results[sn] = {
            'r2_persistence': r2_persist,
            'r2_mean': r2_mean,
            'n_test': min_len,
        }
    return results


def evaluate_gru_per_session(model, sessions_data, session_nums, region, device):
    """Get GRU R2 and example predictions per session."""
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
        test_x_t = torch.tensor(np.array(test_x), dtype=torch.float32).to(device)
        test_y_t = torch.tensor(np.array(test_y), dtype=torch.float32).to(device)
        model.eval()
        with torch.no_grad():
            pred = model(test_x_t, sn)
            mse = criterion(pred, test_y_t).item()
            ss_res = ((test_y_t - pred) ** 2).sum().item()
            ss_tot = ((test_y_t - test_y_t.mean(dim=0)) ** 2).sum().item()
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        results[sn] = {
            'r2_gru': r2,
            'predictions': pred.cpu().numpy(),
            'actuals': test_y_t.cpu().numpy(),
        }
    return results


def train_shuffle_control(sessions_data, session_nums, region, n_repeats=3):
    """Train GRU on time-shuffled data. Returns mean R2 across repeats."""
    neuron_counts = {sn: sessions_data[sn][region]['n_neurons'] for sn in session_nums}
    all_r2s = {sn: [] for sn in session_nums}

    for rep in range(n_repeats):
        train_ds = MultiSessionDataset(sessions_data, session_nums, region, SEQ_LEN,
                                       'train', TRAIN_FRAC, shuffle_time=True)
        test_ds = MultiSessionDataset(sessions_data, session_nums, region, SEQ_LEN,
                                      'test', TRAIN_FRAC, shuffle_time=True)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                  collate_fn=collate_by_session)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                                 collate_fn=collate_by_session)

        model = PooledGRU(neuron_counts, D_SHARED, HIDDEN_SIZE, NUM_LAYERS, DROPOUT).to(DEVICE)
        _, _, _ = train_pooled_model(model, train_loader, test_loader,
                                     NUM_EPOCHS, PATIENCE, LEARNING_RATE, DEVICE)

        # Evaluate on shuffled test data
        criterion = nn.MSELoss()
        for sn in session_nums:
            if sn in test_ds.shuffled_data:
                data = test_ds.shuffled_data[sn]
            else:
                data = sessions_data[sn][region]['data']
            T = len(data)
            split_idx = int(T * TRAIN_FRAC)
            tx = [data[i:i + SEQ_LEN] for i in range(split_idx, T - SEQ_LEN)]
            ty = [data[i + SEQ_LEN] for i in range(split_idx, T - SEQ_LEN)]
            if not tx:
                continue
            tx_t = torch.tensor(np.array(tx), dtype=torch.float32).to(DEVICE)
            ty_t = torch.tensor(np.array(ty), dtype=torch.float32).to(DEVICE)
            model.eval()
            with torch.no_grad():
                pred = model(tx_t, sn)
                ss_res = ((ty_t - pred) ** 2).sum().item()
                ss_tot = ((ty_t - ty_t.mean(dim=0)) ** 2).sum().item()
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            all_r2s[sn].append(r2)

    return {sn: np.mean(r2s) if r2s else np.nan for sn, r2s in all_r2s.items()}


# =============================================================================
# MAIN
# =============================================================================

def run_controls_for_dataset(dataset_name, sessions_data, region, session_nums, model_path):
    """Run all baseline controls for one dataset/region combination."""
    print(f"\n{'='*70}")
    print(f"  {dataset_name} — {region.upper()}")
    print(f"{'='*70}")
    print(f"  Sessions: {sorted(session_nums)}")

    # 1. Load trained GRU model
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    neuron_counts = checkpoint['neuron_counts']
    model = PooledGRU(neuron_counts, D_SHARED, HIDDEN_SIZE, NUM_LAYERS, DROPOUT).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"  Loaded model: {model_path.name}")

    # 2. Compute baselines
    print(f"  Computing baselines...")
    baselines = compute_baselines(sessions_data, session_nums, region)

    # 3. Evaluate GRU
    gru_results = evaluate_gru_per_session(model, sessions_data, session_nums, region, DEVICE)

    # 4. Shuffle control
    print(f"  Training shuffle controls (3 repeats)...")
    t0 = time.time()
    shuffle_r2s = train_shuffle_control(sessions_data, session_nums, region, n_repeats=3)
    print(f"  Shuffle controls done in {time.time()-t0:.1f}s")

    # Compile results
    results = []
    print(f"\n  {'Sess':>4s} {'State':>7s} {'Mean':>8s} {'Persist':>8s} {'Shuffle':>8s} {'GRU':>8s} {'GRU-Persist':>12s}")
    print(f"  {'-'*60}")

    for sn in sorted(session_nums):
        if sn not in baselines or sn not in gru_results:
            continue
        b = baselines[sn]
        g = gru_results[sn]
        s = shuffle_r2s.get(sn, np.nan)
        state = sessions_data[sn]['state']
        delta = g['r2_gru'] - b['r2_persistence']

        print(f"  {sn:>4d} {state:>7s} {b['r2_mean']:>8.4f} {b['r2_persistence']:>8.4f} "
              f"{s:>8.4f} {g['r2_gru']:>8.4f} {delta:>+12.4f}")

        results.append({
            'dataset': dataset_name, 'region': region.upper(),
            'session': sn, 'state': state,
            'r2_mean': b['r2_mean'], 'r2_persistence': b['r2_persistence'],
            'r2_shuffle': s, 'r2_gru': g['r2_gru'],
            'gru_minus_persist': delta,
        })

    # Summary stats
    gru_vals = [r['r2_gru'] for r in results]
    persist_vals = [r['r2_persistence'] for r in results]
    shuffle_vals = [r['r2_shuffle'] for r in results if not np.isnan(r['r2_shuffle'])]
    deltas = [r['gru_minus_persist'] for r in results]

    print(f"\n  Summary:")
    print(f"    Mean baseline R2:       {np.mean([r['r2_mean'] for r in results]):.4f}")
    print(f"    Persistence baseline:   {np.mean(persist_vals):.4f}")
    print(f"    Shuffle GRU:            {np.mean(shuffle_vals):.4f}")
    print(f"    Trained GRU:            {np.mean(gru_vals):.4f}")
    print(f"    GRU - Persistence:      {np.mean(deltas):+.4f}")

    if len(gru_vals) > 1 and len(persist_vals) > 1:
        _, p = sp_stats.wilcoxon(gru_vals, persist_vals)
        print(f"    Wilcoxon GRU vs Persist: p={p:.4f} {'*' if p < 0.05 else 'ns'}")

    if len(gru_vals) > 1 and len(shuffle_vals) > 1:
        _, p = sp_stats.mannwhitneyu(gru_vals, shuffle_vals, alternative='greater')
        print(f"    MWU GRU > Shuffle:       p={p:.4f} {'*' if p < 0.05 else 'ns'}")

    return results, gru_results


def generate_figures(all_results, all_predictions, sessions_data):
    """Generate summary figure and prediction examples."""

    df = pd.DataFrame(all_results)
    df.to_csv(Path("data") / "gru_baseline_controls.csv", index=False)
    print(f"\nResults saved: data/gru_baseline_controls.csv")

    # --- Figure 1: R2 comparison across all datasets/regions ---
    unique_combos = df[['dataset', 'region']].drop_duplicates().values.tolist()
    n_combos = len(unique_combos)
    fig, axes = plt.subplots(1, n_combos, figsize=(6 * n_combos, 6))
    if n_combos == 1:
        axes = [axes]
    fig.suptitle("GRU vs Baselines: R2 Comparison", fontsize=15, fontweight='bold')

    colors_method = {
        'Mean': '#BDBDBD',
        'Persistence': '#FFB74D',
        'Shuffle GRU': '#CE93D8',
        'Trained GRU': '#4CAF50',
    }

    for ax, (ds, reg) in zip(axes, unique_combos):
        sub = df[(df['dataset'] == ds) & (df['region'] == reg)]

        methods = ['r2_mean', 'r2_persistence', 'r2_shuffle', 'r2_gru']
        labels = ['Mean', 'Persistence', 'Shuffle GRU', 'Trained GRU']
        x_pos = np.arange(len(methods))

        for xi, (method, label) in enumerate(zip(methods, labels)):
            vals = sub[method].dropna().values
            ax.bar(xi, np.mean(vals), width=0.6, color=colors_method[label],
                   alpha=0.7, edgecolor='black', linewidth=0.8)
            ax.errorbar(xi, np.mean(vals), yerr=np.std(vals)/np.sqrt(len(vals)),
                        color='black', capsize=5, linewidth=1.5)
            jitter = np.random.uniform(-0.12, 0.12, len(vals))
            ax.scatter([xi + j for j in jitter], vals, color=colors_method[label],
                       edgecolors='black', s=50, zorder=5, linewidth=0.8)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, fontsize=10, rotation=15, ha='right')
        ax.set_ylabel('Test R2', fontsize=12)
        ax.set_title(f'{ds} {reg}', fontsize=13, fontweight='bold')
        ax.axhline(0, color='gray', linestyle='--', alpha=0.3)

        # Add significance annotation
        gru_vals = sub['r2_gru'].values
        persist_vals = sub['r2_persistence'].values
        if len(gru_vals) > 1:
            _, p = sp_stats.wilcoxon(gru_vals, persist_vals)
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
            y_max = max(np.max(gru_vals), np.max(persist_vals))
            ax.text(2.5, y_max * 1.1, f'GRU vs Persist\np={p:.4f} {sig}',
                    ha='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    fig.savefig(Path("figures") / "gru_baseline_controls_r2.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: figures/gru_baseline_controls_r2.png")

    # --- Figure 2: Example predictions (one per dataset/region) ---
    fig2, axes2 = plt.subplots(n_combos, 1, figsize=(16, 4 * n_combos))
    if n_combos == 1:
        axes2 = [axes2]
    fig2.suptitle("Example Predictions: Actual vs GRU (3 neurons, 200 time steps)",
                  fontsize=14, fontweight='bold')

    for ax, (ds_reg_key, pred_data) in zip(axes2, all_predictions.items()):
        # Pick first session with predictions
        sn = sorted(pred_data.keys())[0]
        actuals = pred_data[sn]['actuals']
        preds = pred_data[sn]['predictions']

        # Show first 200 time steps, 3 neurons
        n_show = min(200, len(actuals))
        n_neurons_show = min(3, actuals.shape[1])
        t = np.arange(n_show) * BIN_SIZE_MS / 1000  # seconds

        for ni in range(n_neurons_show):
            offset = ni * 4  # vertical offset
            ax.plot(t, actuals[:n_show, ni] + offset, color='black',
                    linewidth=0.8, alpha=0.8, label='Actual' if ni == 0 else '')
            ax.plot(t, preds[:n_show, ni] + offset, color='#4CAF50',
                    linewidth=0.8, alpha=0.7, label='GRU pred' if ni == 0 else '')

            # Per-neuron R2
            ss_res = np.sum((actuals[:, ni] - preds[:, ni]) ** 2)
            ss_tot = np.sum((actuals[:, ni] - actuals[:, ni].mean()) ** 2)
            nr2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            ax.text(t[-1] + 1, offset, f'n{ni} R2={nr2:.3f}', fontsize=9, va='center')

        ax.set_xlabel('Time (s)', fontsize=11)
        ax.set_ylabel('Activity (z-scored + offset)', fontsize=11)
        ax.set_title(f'{ds_reg_key} — Session {sn}', fontsize=12)
        ax.legend(fontsize=10)

    plt.tight_layout()
    fig2.savefig(Path("figures") / "gru_baseline_controls_predictions.png",
                 dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: figures/gru_baseline_controls_predictions.png")


# =============================================================================

def main():
    print(f"Device: {DEVICE}")
    print(f"Baseline controls for pooled GRU models\n")

    all_results = []
    all_predictions = {}

    # --- Single-Probe ---
    print("Loading single-probe data...")
    sp_data = load_single_probe_sessions()
    sp_sessions = sorted(sp_data.keys())
    print(f"  Loaded {len(sp_sessions)} sessions")

    for region in ['lha', 'rsp']:
        model_path = Path("data") / f"gru_pooled_{region}_all_model.pt"
        if not model_path.exists():
            print(f"  Model not found: {model_path}, skipping")
            continue
        results, gru_res = run_controls_for_dataset(
            'Single-Probe', sp_data, region, sp_sessions, model_path)
        all_results.extend(results)
        all_predictions[f'SP-{region.upper()}'] = gru_res

    # --- Dual-Probe ---
    print("\n\nLoading dual-probe data...")
    dp_data = load_dual_probe_sessions()
    dp_sessions = sorted(dp_data.keys())
    print(f"  Loaded {len(dp_sessions)} sessions")

    for region in ['aca', 'lha']:
        model_path = Path("data") / f"gru_pooled_dp_{region}_all_model.pt"
        if not model_path.exists():
            print(f"  Model not found: {model_path}, skipping")
            continue
        results, gru_res = run_controls_for_dataset(
            'Dual-Probe', dp_data, region, dp_sessions, model_path)
        all_results.extend(results)
        all_predictions[f'DP-{region.upper()}'] = gru_res

    # --- Figures ---
    print("\n\nGenerating figures...")
    generate_figures(all_results, all_predictions, None)

    print("\nAll done!")


if __name__ == "__main__":
    main()
