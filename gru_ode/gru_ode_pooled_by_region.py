"""
Pooled GRU-ODE with Session-Specific Input Layers -- By Region (LHA / RSP)
===========================================================================
Continuous-time Gated Neural ODE replacing the discrete GRU.

Between observations: hidden state evolves via ODE  dh/dt = (1-z(h))*(n(h)-h)
At each observation: hidden state updated by GRU cell (discrete jump)
Prediction: ODE evolve one more step (no observation)

Same pooled architecture as gru_pooled_by_region.py:
  - Session-specific input/output projections
  - Shared ODE core + shared observation cell

For each region x condition:
  - Condition-specific pooled model (Fed-only or Fasted-only)
  - Combined model (all 8 sessions)
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
from torchdiffeq import odeint
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
ODE_GATE_HIDDEN = 64       # Internal dim for ODE gate networks
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
NUM_EPOCHS = 150
PATIENCE = 15
TRAIN_FRAC = 0.8
GRAD_CLIP = 1.0            # Max gradient norm for clipping

# ODE solver settings
ODE_SOLVER = 'rk4'          # Fixed-step 4th-order Runge-Kutta (6x faster than dopri5)
ODE_STEP_SIZE = 1.0          # Single RK4 step per integration interval
ODE_DT = 1.0                # Normalized time between observations

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
# DATA LOADING (identical to gru_pooled_by_region.py)
# =============================================================================

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


def load_all_sessions():
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
# DATASET (identical to gru_pooled_by_region.py)
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
# GRU-ODE MODEL
# =============================================================================

class GRUODEFunc(nn.Module):
    """
    ODE dynamics: dh/dt = (1 - z(h)) * (n(h) - h)

    z(h) = sigmoid(Linear(h))   -- update gate, controls rate of change
    n(h) = tanh(Linear(h))      -- candidate state, where dynamics point

    When z is close to 1: dh/dt approaches 0 (stable, slow dynamics)
    When z is close to 0: dh/dt = n(h) - h (fast dynamics toward candidate)
    """

    def __init__(self, hidden_size, gate_hidden=64):
        super().__init__()
        self.update_gate = nn.Sequential(
            nn.Linear(hidden_size, gate_hidden),
            nn.Tanh(),
            nn.Linear(gate_hidden, hidden_size),
            nn.Sigmoid(),
        )
        self.candidate = nn.Sequential(
            nn.Linear(hidden_size, gate_hidden),
            nn.Tanh(),
            nn.Linear(gate_hidden, hidden_size),
            nn.Tanh(),
        )

    def forward(self, t, h):
        z = self.update_gate(h)
        n = self.candidate(h)
        dhdt = (1 - z) * (n - h)
        return dhdt


class PooledGRUODE(nn.Module):
    """
    Pooled GRU-ODE with session-specific input/output projections.

    Processing for each time step:
      1. Evolve hidden state via ODE (continuous dynamics)
      2. Project input into shared space
      3. Update hidden state with observation (discrete GRU cell)

    Prediction:
      4. Evolve one more ODE step (no observation)
      5. Decode via shared + session-specific layers
    """

    def __init__(self, session_neuron_counts, d_shared, hidden_size, gate_hidden=64):
        super().__init__()
        self.d_shared = d_shared
        self.hidden_size = hidden_size

        # Session-specific input projections
        self.input_projections = nn.ModuleDict()
        for sn, n_neurons in session_neuron_counts.items():
            self.input_projections[str(sn)] = nn.Linear(n_neurons, d_shared)

        # Shared ODE dynamics (continuous evolution between observations)
        self.ode_func = GRUODEFunc(hidden_size, gate_hidden)

        # Shared observation update cell (discrete jump at each observation)
        self.obs_cell = nn.GRUCell(input_size=d_shared, hidden_size=hidden_size)

        # Shared decoder
        self.fc_shared = nn.Linear(hidden_size, d_shared)

        # Session-specific output projections
        self.output_projections = nn.ModuleDict()
        for sn, n_neurons in session_neuron_counts.items():
            self.output_projections[str(sn)] = nn.Linear(d_shared, n_neurons)

        # Time span for one ODE step
        self.register_buffer('t_span', torch.tensor([0.0, ODE_DT]))

    def _ode_evolve(self, h):
        """Evolve hidden state forward one time step via ODE."""
        h_evolved = odeint(
            self.ode_func, h, self.t_span,
            method=ODE_SOLVER, options={'step_size': ODE_STEP_SIZE},
        )
        return h_evolved[-1]  # Return state at t=dt

    def forward(self, x, session_num):
        """
        x: (batch, seq_len, n_neurons)
        session_num: which session (for input/output projections)
        """
        sn_key = str(session_num)
        batch_size = x.shape[0]
        seq_len = x.shape[1]

        # Initialize hidden state as zeros
        h = torch.zeros(batch_size, self.hidden_size, device=x.device)

        # Process each time step: ODE evolve -> observe -> update
        for k in range(seq_len):
            # 1. Evolve hidden state via ODE
            h = self._ode_evolve(h)

            # 2. Project observation into shared space
            x_proj = self.input_projections[sn_key](x[:, k, :])

            # 3. Update hidden state with observation (GRU cell)
            h = self.obs_cell(x_proj, h)

        # 4. Predict: evolve one more step via ODE (no observation)
        h_final = self._ode_evolve(h)

        # 5. Decode
        shared_out = self.fc_shared(h_final)
        pred = self.output_projections[sn_key](shared_out)
        return pred

    def extract_hidden_states(self, x, session_num):
        """Extract hidden states at each time step for latent analysis."""
        sn_key = str(session_num)
        batch_size = x.shape[0]
        seq_len = x.shape[1]

        with torch.no_grad():
            h = torch.zeros(batch_size, self.hidden_size, device=x.device)
            hidden_seq = []

            for k in range(seq_len):
                h = self._ode_evolve(h)
                x_proj = self.input_projections[sn_key](x[:, k, :])
                h = self.obs_cell(x_proj, h)
                hidden_seq.append(h.unsqueeze(1))

            # Return all hidden states: (batch, seq_len, hidden_size)
            return torch.cat(hidden_seq, dim=1)


# =============================================================================
# COLLATE (identical to gru_pooled_by_region.py)
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


# =============================================================================
# TRAINING (with gradient clipping for ODE stability)
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
            # Gradient clipping for ODE stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
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

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"      Epoch {epoch+1}: train={mean_train:.6f} val={mean_val:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return history, best_epoch, best_val_loss


# =============================================================================
# EVALUATION (same interface as GRU)
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
        # Process in chunks (ODE uses more memory per sample)
        chunk_size = 256
        all_pred = []
        with torch.no_grad():
            for start in range(0, len(test_x), chunk_size):
                chunk_x = test_x[start:start + chunk_size]
                pred = model(chunk_x, sn)
                all_pred.append(pred)
        pred_all = torch.cat(all_pred, dim=0)
        mse = criterion(pred_all, test_y).item()
        ss_res = ((test_y - pred_all) ** 2).sum().item()
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
        chunk_size = 256  # Smaller chunks for ODE memory
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
    results_all = []
    models = {}

    for condition, session_nums in [('Fed', fed_sessions), ('Fasted', fasted_sessions)]:
        print(f"\n  --- {region.upper()} Pooled GRU-ODE: {condition} ({len(session_nums)} sessions) ---")

        neuron_counts = {sn: sessions_data[sn][region]['n_neurons'] for sn in session_nums}
        print(f"    Neuron counts: {neuron_counts}")

        train_ds = MultiSessionDataset(sessions_data, session_nums, region, SEQ_LEN, 'train', TRAIN_FRAC)
        test_ds = MultiSessionDataset(sessions_data, session_nums, region, SEQ_LEN, 'test', TRAIN_FRAC)
        print(f"    Train: {len(train_ds)}, Test: {len(test_ds)} samples")

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                  collate_fn=collate_by_session)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                                 collate_fn=collate_by_session)

        model = PooledGRUODE(neuron_counts, D_SHARED, HIDDEN_SIZE, ODE_GATE_HIDDEN).to(DEVICE)
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

        model_path = Path("data") / f"gru_ode_pooled_{region}_{condition.lower()}_model.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'neuron_counts': neuron_counts,
            'config': {'d_shared': D_SHARED, 'hidden_size': HIDDEN_SIZE,
                       'gate_hidden': ODE_GATE_HIDDEN, 'seq_len': SEQ_LEN,
                       'bin_size_ms': BIN_SIZE_MS, 'ode_solver': ODE_SOLVER,
                       'ode_step_size': ODE_STEP_SIZE, 'ode_dt': ODE_DT},
            'history': history, 'best_epoch': best_epoch,
        }, model_path)

        hist_df = pd.DataFrame(history)
        hist_df.to_csv(Path("data") / f"gru_ode_pooled_{region}_{condition.lower()}_history.csv", index=False)
        models[condition] = model

    # --- Combined model (all 8 sessions) ---
    all_sessions = sorted(fed_sessions + fasted_sessions)
    print(f"\n  --- {region.upper()} Combined Pooled GRU-ODE (all 8 sessions) ---")

    all_neuron_counts = {sn: sessions_data[sn][region]['n_neurons'] for sn in all_sessions}
    print(f"    Neuron counts: {all_neuron_counts}")

    train_ds_all = MultiSessionDataset(sessions_data, all_sessions, region, SEQ_LEN, 'train', TRAIN_FRAC)
    test_ds_all = MultiSessionDataset(sessions_data, all_sessions, region, SEQ_LEN, 'test', TRAIN_FRAC)
    print(f"    Train: {len(train_ds_all)}, Test: {len(test_ds_all)} samples")

    train_loader_all = DataLoader(train_ds_all, batch_size=BATCH_SIZE, shuffle=True,
                                  collate_fn=collate_by_session)
    test_loader_all = DataLoader(test_ds_all, batch_size=BATCH_SIZE, shuffle=False,
                                 collate_fn=collate_by_session)

    model_all = PooledGRUODE(all_neuron_counts, D_SHARED, HIDDEN_SIZE, ODE_GATE_HIDDEN).to(DEVICE)
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

    torch.save({
        'model_state_dict': model_all.state_dict(),
        'neuron_counts': all_neuron_counts,
        'config': {'d_shared': D_SHARED, 'hidden_size': HIDDEN_SIZE,
                   'gate_hidden': ODE_GATE_HIDDEN, 'seq_len': SEQ_LEN,
                   'bin_size_ms': BIN_SIZE_MS, 'ode_solver': ODE_SOLVER,
                   'ode_step_size': ODE_STEP_SIZE, 'ode_dt': ODE_DT},
        'history': history_all, 'best_epoch': best_epoch_all,
    }, Path("data") / f"gru_ode_pooled_{region}_all_model.pt")

    return results_all, all_hidden, all_sessions


def main():
    print(f"Device: {DEVICE}")
    print(f"Config: {BIN_SIZE_MS}ms bins, D_shared={D_SHARED}, hidden={HIDDEN_SIZE}, "
          f"ODE_gate_hidden={ODE_GATE_HIDDEN}, seq_len={SEQ_LEN}")
    print(f"ODE: solver={ODE_SOLVER}, step_size={ODE_STEP_SIZE}, dt={ODE_DT}")
    print(f"Gradient clipping: max_norm={GRAD_CLIP}")
    print()

    print("Loading session data...")
    sessions_data = load_all_sessions()
    print(f"Loaded {len(sessions_data)} sessions\n")

    fed_sessions = [sn for sn in sessions_data if sessions_data[sn]['state'] == 'Fed']
    fasted_sessions = [sn for sn in sessions_data if sessions_data[sn]['state'] == 'Fasted']

    all_results = []

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
    df.to_csv(Path("data") / "gru_ode_pooled_by_region_results.csv", index=False)
    print(f"\nResults saved: data/gru_ode_pooled_by_region_results.csv")

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

            print(f"\n  {region} -- Fed vs Fasted:")
            for metric in ['test_r2', 'pr', 'variance', 'speed', 'pcs_90']:
                fed_v = fed_df[metric].values
                fas_v = fas_df[metric].values
                _, p = sp_stats.mannwhitneyu(fed_v, fas_v, alternative='two-sided')
                sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
                print(f"    {metric:12s}: Fed={np.mean(fed_v):.4f} vs Fasted={np.mean(fas_v):.4f}  "
                      f"p={p:.4f} {sig}")

    # --- Comparison with GRU ---
    print(f"\n{'='*70}")
    print("COMPARISON: GRU-ODE vs GRU")
    print(f"{'='*70}")

    gru_csv = Path("data") / "gru_pooled_by_region_results.csv"
    if gru_csv.exists():
        df_gru = pd.read_csv(gru_csv)

        for model_type in ['condition_specific', 'combined']:
            print(f"\n--- {model_type} ---")
            ode_mt = df[df['model_type'] == model_type]
            gru_mt = df_gru[df_gru['model_type'] == model_type]

            for region in ['LHA', 'RSP']:
                ode_r = ode_mt[ode_mt['region'] == region].sort_values('session')
                gru_r = gru_mt[gru_mt['region'] == region].sort_values('session')

                common_sessions = sorted(set(ode_r['session'].values) & set(gru_r['session'].values))
                if not common_sessions:
                    continue

                print(f"\n  {region}:")
                for metric in ['test_r2', 'pr', 'variance', 'speed']:
                    ode_vals = [ode_r[ode_r['session'] == s][metric].values[0] for s in common_sessions]
                    gru_vals = [gru_r[gru_r['session'] == s][metric].values[0] for s in common_sessions]
                    ode_mean = np.mean(ode_vals)
                    gru_mean = np.mean(gru_vals)
                    diff = ode_mean - gru_mean
                    print(f"    {metric:12s}: GRU={gru_mean:.4f}  ODE={ode_mean:.4f}  diff={diff:+.4f}")
    else:
        print("  GRU results not found, skipping comparison")

    # --- Figures ---
    print("\nGenerating figures...")

    colors_state = {'Fed': '#2196F3', 'Fasted': '#F44336'}

    # Figure: Combined model -- Fed vs Fasted by region
    fig, axes = plt.subplots(2, 5, figsize=(24, 10))
    fig.suptitle("Pooled GRU-ODE by Region -- Combined Model (All 8 Sessions)", fontsize=14)

    comb_df = df[df['model_type'] == 'combined']
    metrics_list = ['test_r2', 'pr', 'pcs_90', 'variance', 'speed']
    metric_labels = ['Test R2', 'Participation Ratio', 'PCs @ 90%', 'Hidden Variance', 'Trajectory Speed']

    for row, region in enumerate(['LHA', 'RSP']):
        rdf = comb_df[comb_df['region'] == region]
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
    fig.savefig(Path("figures") / "gru_ode_pooled_by_region_combined.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: figures/gru_ode_pooled_by_region_combined.png")

    # Figure: GRU vs GRU-ODE comparison
    if gru_csv.exists():
        df_gru = pd.read_csv(gru_csv)
        comb_gru = df_gru[df_gru['model_type'] == 'combined']
        comb_ode = df[df['model_type'] == 'combined']

        fig2, axes2 = plt.subplots(2, 5, figsize=(24, 10))
        fig2.suptitle("GRU vs GRU-ODE Comparison -- Combined Pooled Model", fontsize=14)

        for row, region in enumerate(['LHA', 'RSP']):
            gru_r = comb_gru[comb_gru['region'] == region].sort_values('session')
            ode_r = comb_ode[comb_ode['region'] == region].sort_values('session')
            sessions = sorted(set(gru_r['session'].values) & set(ode_r['session'].values))

            for col, (metric, label) in enumerate(zip(metrics_list, metric_labels)):
                ax = axes2[row, col]
                gru_vals = [gru_r[gru_r['session'] == s][metric].values[0] for s in sessions]
                ode_vals = [ode_r[ode_r['session'] == s][metric].values[0] for s in sessions]

                x = np.arange(len(sessions))
                width = 0.35
                ax.bar(x - width/2, gru_vals, width, label='GRU', color='gray', alpha=0.7)
                ax.bar(x + width/2, ode_vals, width, label='GRU-ODE', color='#E91E63', alpha=0.7)

                labels_x = []
                for s in sessions:
                    state = sessions_data[int(s)]['state']
                    labels_x.append(f"S{int(s)}\n({state[0]})")
                ax.set_xticks(x)
                ax.set_xticklabels(labels_x, fontsize=8)
                ax.set_title(f"{region} {label}", fontsize=10)
                if row == 0 and col == 0:
                    ax.legend(fontsize=9)

        plt.tight_layout()
        fig2.savefig(Path("figures") / "gru_ode_vs_gru_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("  Saved: figures/gru_ode_vs_gru_comparison.png")

    print("\nAll done!")


if __name__ == "__main__":
    main()
