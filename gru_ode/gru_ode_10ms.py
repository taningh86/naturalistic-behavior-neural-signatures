"""
GRU-ODE at 10ms with Poisson Loss and 100ms Prediction Window
==============================================================
Fine-grained temporal resolution exploiting the neural ODE's continuous dynamics.

Key design:
  - Input: 10ms binned spike counts, z-scored (SEQ_LEN=50 = 500ms context)
  - Target: Raw spike counts summed over next 100ms (10 bins)
  - Loss: Poisson NLL (model outputs log-rate λ)
  - Prediction: ODE evolves 10 steps (100ms) with no observations
  - Eval: Poisson deviance explained (D²), correlation

This aligns with:
  - The data's generative process (spikes are Poisson-distributed)
  - The 100ms behavioral data resolution
  - The ODE's ability to forecast over multi-step horizons
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

BIN_SIZE_MS = 10               # 10ms input bins
PRED_WINDOW_MS = 100           # 100ms prediction window
PRED_BINS = PRED_WINDOW_MS // BIN_SIZE_MS  # 10 bins to sum for target
FS = 30000
BIN_SAMPLES = int(BIN_SIZE_MS * FS / 1000)  # 300 samples per bin

SEQ_LEN = 50                   # 50 bins = 500ms of context
STRIDE = 50                    # Non-overlapping for test run (switch to 10 for full)
D_SHARED = 32
HIDDEN_SIZE = 32
ODE_GATE_HIDDEN = 64
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
NUM_EPOCHS = 150
PATIENCE = 20
TRAIN_FRAC = 0.8
GRAD_CLIP = 1.0

# ODE solver settings
ODE_SOLVER = 'rk4'
ODE_STEP_SIZE = 1.0
ODE_DT = 1.0

LHA_DEPTH_MAX = 1300
RSP_DEPTH_MIN = 1300

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Test mode: only train LHA Fed to calibrate
TEST_MODE = False

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
    """Bin spike trains at 10ms. Returns both z-scored (input) and raw (target)."""
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

    raw_data = data.copy()  # Raw counts for Poisson targets

    # Z-score for model input
    means = data.mean(axis=0, keepdims=True)
    stds = data.std(axis=0, keepdims=True)
    stds[stds < 1e-8] = 1.0
    zscore_data = (data - means) / stds

    return zscore_data, raw_data, n_bins


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
            print(f"  Session {sess_num}: too few units, skipping")
            continue

        lha_zscore, lha_raw, lha_bins = bin_spike_trains(sorting, lha_ids)
        rsp_zscore, rsp_raw, rsp_bins = bin_spike_trains(sorting, rsp_ids)

        # Compute mean firing rates for reporting
        lha_mean_rate = lha_raw.mean(axis=0).mean() * (1000 / BIN_SIZE_MS)  # Hz
        rsp_mean_rate = rsp_raw.mean(axis=0).mean() * (1000 / BIN_SIZE_MS)

        sessions_data[sess_num] = {
            'lha': {
                'zscore': lha_zscore, 'raw': lha_raw,
                'n_neurons': len(lha_ids), 'n_bins': lha_bins,
            },
            'rsp': {
                'zscore': rsp_zscore, 'raw': rsp_raw,
                'n_neurons': len(rsp_ids), 'n_bins': rsp_bins,
            },
            'state': info['state'],
            'phase': info['phase'],
        }
        print(f"  Session {sess_num}: {info['state']} {info['phase']}, "
              f"LHA={len(lha_ids)} ({lha_bins} bins, {lha_mean_rate:.1f} Hz), "
              f"RSP={len(rsp_ids)} ({rsp_bins} bins, {rsp_mean_rate:.1f} Hz)")
    return sessions_data


# =============================================================================
# DATASET
# =============================================================================

class MultiSessionDataset(Dataset):
    def __init__(self, sessions_data, session_nums, region, seq_len, pred_bins,
                 stride, split='train', train_frac=0.8):
        self.seq_len = seq_len
        self.pred_bins = pred_bins
        self.samples = []
        self.region = region
        for sn in session_nums:
            data = sessions_data[sn][region]['zscore']
            T = len(data)
            split_idx = int(T * train_frac)
            if split == 'train':
                start, end = 0, split_idx
            else:
                start, end = split_idx, T
            # Need seq_len input bins + pred_bins target bins
            for i in range(0, end - start - seq_len - pred_bins, stride):
                self.samples.append((sn, start + i))
        self.sessions_data = sessions_data

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sn, start = self.samples[idx]
        zscore = self.sessions_data[sn][self.region]['zscore']
        raw = self.sessions_data[sn][self.region]['raw']
        # Input: z-scored 10ms bins
        x = torch.tensor(zscore[start:start + self.seq_len], dtype=torch.float32)
        # Target: raw spike counts summed over next 100ms (10 bins)
        y = torch.tensor(
            raw[start + self.seq_len:start + self.seq_len + self.pred_bins].sum(axis=0),
            dtype=torch.float32
        )
        return x, y, sn


# =============================================================================
# GRU-ODE MODEL
# =============================================================================

class GRUODEFunc(nn.Module):
    """ODE dynamics: dh/dt = (1 - z(h)) * (n(h) - h)"""
    def __init__(self, hidden_size, gate_hidden=64):
        super().__init__()
        self.update_gate = nn.Sequential(
            nn.Linear(hidden_size, gate_hidden), nn.Tanh(),
            nn.Linear(gate_hidden, hidden_size), nn.Sigmoid(),
        )
        self.candidate = nn.Sequential(
            nn.Linear(hidden_size, gate_hidden), nn.Tanh(),
            nn.Linear(gate_hidden, hidden_size), nn.Tanh(),
        )

    def forward(self, t, h):
        z = self.update_gate(h)
        n = self.candidate(h)
        return (1 - z) * (n - h)


class PooledGRUODE(nn.Module):
    def __init__(self, session_neuron_counts, d_shared, hidden_size,
                 gate_hidden=64, pred_steps=10):
        super().__init__()
        self.d_shared = d_shared
        self.hidden_size = hidden_size
        self.pred_steps = pred_steps

        self.input_projections = nn.ModuleDict()
        for sn, n_neurons in session_neuron_counts.items():
            self.input_projections[str(sn)] = nn.Linear(n_neurons, d_shared)

        self.ode_func = GRUODEFunc(hidden_size, gate_hidden)
        self.obs_cell = nn.GRUCell(input_size=d_shared, hidden_size=hidden_size)
        self.fc_shared = nn.Linear(hidden_size, d_shared)

        self.output_projections = nn.ModuleDict()
        for sn, n_neurons in session_neuron_counts.items():
            self.output_projections[str(sn)] = nn.Linear(d_shared, n_neurons)

        self.register_buffer('t_span', torch.tensor([0.0, ODE_DT]))

    def _ode_evolve(self, h):
        h_evolved = odeint(
            self.ode_func, h, self.t_span,
            method=ODE_SOLVER, options={'step_size': ODE_STEP_SIZE},
        )
        return h_evolved[-1]

    def forward(self, x, session_num):
        sn_key = str(session_num)
        batch_size = x.shape[0]
        h = torch.zeros(batch_size, self.hidden_size, device=x.device)

        # Process input sequence (50 bins = 500ms)
        for k in range(x.shape[1]):
            h = self._ode_evolve(h)
            x_proj = self.input_projections[sn_key](x[:, k, :])
            h = self.obs_cell(x_proj, h)

        # Predict: evolve ODE for pred_steps (10 steps = 100ms) with no observations
        for _ in range(self.pred_steps):
            h = self._ode_evolve(h)

        # Decode to log-rate
        shared_out = self.fc_shared(h)
        log_rate = self.output_projections[sn_key](shared_out)
        return log_rate  # (batch, N_neurons) — log(λ) for 100ms window

    def extract_hidden_states(self, x, session_num):
        """Extract hidden states at each input time step."""
        sn_key = str(session_num)
        batch_size = x.shape[0]
        with torch.no_grad():
            h = torch.zeros(batch_size, self.hidden_size, device=x.device)
            hidden_seq = []
            for k in range(x.shape[1]):
                h = self._ode_evolve(h)
                x_proj = self.input_projections[sn_key](x[:, k, :])
                h = self.obs_cell(x_proj, h)
                hidden_seq.append(h.unsqueeze(1))
            return torch.cat(hidden_seq, dim=1)


# =============================================================================
# COLLATE
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
# TRAINING
# =============================================================================

def train_pooled_model(model, train_loader, val_loader, n_epochs, patience, lr, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.PoissonNLLLoss(log_input=True, reduction='mean')
    best_val_loss = np.inf
    best_epoch = 0
    best_state = None
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(n_epochs):
        epoch_start = time.time()
        model.train()
        train_losses = []
        for batch_dict in train_loader:
            total_loss = 0.0
            total_count = 0
            for sn, data in batch_dict.items():
                x = data['x'].to(device)
                y = data['y'].to(device)  # Raw spike counts (100ms sum)
                log_rate = model(x, sn)
                loss = criterion(log_rate, y)
                total_loss += loss * len(x)
                total_count += len(x)
            avg_loss = total_loss / total_count
            optimizer.zero_grad()
            avg_loss.backward()
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
                    log_rate = model(x, sn)
                    loss = criterion(log_rate, y)
                    total_loss += loss * len(x)
                    total_count += len(x)
                if total_count > 0:
                    val_losses.append((total_loss / total_count).item())

        mean_train = np.mean(train_losses)
        mean_val = np.mean(val_losses) if val_losses else np.inf
        history['train_loss'].append(mean_train)
        history['val_loss'].append(mean_val)
        epoch_time = time.time() - epoch_start

        if mean_val < best_val_loss:
            best_val_loss = mean_val
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch - best_epoch >= patience:
            print(f"      Early stopping at epoch {epoch+1}")
            break

        if (epoch + 1) % 2 == 0 or epoch == 0:
            print(f"      Epoch {epoch+1}: train={mean_train:.6f} val={mean_val:.6f} "
                  f"({epoch_time:.1f}s/epoch)")

    if best_state is not None:
        model.load_state_dict(best_state)
    return history, best_epoch, best_val_loss


# =============================================================================
# EVALUATION
# =============================================================================

def poisson_deviance_explained(y_true, log_rate_pred):
    """
    Poisson deviance explained (D²) — analog of R² for count data.
    D² = 1 - deviance_model / deviance_null
    """
    rate_pred = np.exp(np.clip(log_rate_pred, -20, 20))  # Prevent overflow
    eps = 1e-8

    # Model deviance: 2 * sum(y * log(y/λ) - (y - λ))
    with np.errstate(divide='ignore', invalid='ignore'):
        term1 = np.where(y_true > 0, y_true * np.log(y_true / (rate_pred + eps)), 0.0)
    dev_model = 2 * np.sum(term1 - (y_true - rate_pred))

    # Null deviance: predict mean rate per neuron
    mean_rate = np.maximum(y_true.mean(axis=0, keepdims=True), eps)
    with np.errstate(divide='ignore', invalid='ignore'):
        term1_null = np.where(y_true > 0, y_true * np.log(y_true / mean_rate), 0.0)
    dev_null = 2 * np.sum(term1_null - (y_true - mean_rate))

    if dev_null <= 0:
        return 0.0
    return 1 - dev_model / dev_null


def evaluate_per_session(model, sessions_data, session_nums, region, device):
    """Evaluate using Poisson deviance and correlation."""
    results = {}
    criterion = nn.PoissonNLLLoss(log_input=True, reduction='mean')

    for sn in session_nums:
        zscore = sessions_data[sn][region]['zscore']
        raw = sessions_data[sn][region]['raw']
        T = len(zscore)
        split_idx = int(T * TRAIN_FRAC)

        test_x, test_y = [], []
        for i in range(split_idx, T - SEQ_LEN - PRED_BINS, STRIDE):
            test_x.append(zscore[i:i + SEQ_LEN])
            test_y.append(raw[i + SEQ_LEN:i + SEQ_LEN + PRED_BINS].sum(axis=0))

        if len(test_x) == 0:
            continue

        test_x = torch.tensor(np.array(test_x), dtype=torch.float32).to(device)
        test_y_np = np.array(test_y)
        test_y_t = torch.tensor(test_y_np, dtype=torch.float32).to(device)

        model.eval()
        chunk_size = 128
        all_log_rate = []
        with torch.no_grad():
            for start in range(0, len(test_x), chunk_size):
                chunk_x = test_x[start:start + chunk_size]
                log_rate = model(chunk_x, sn)
                all_log_rate.append(log_rate.cpu().numpy())

        log_rate_all = np.concatenate(all_log_rate, axis=0)
        rate_pred = np.exp(np.clip(log_rate_all, -20, 20))

        # Poisson NLL on test set
        log_rate_t = torch.tensor(log_rate_all, dtype=torch.float32)
        test_nll = criterion(log_rate_t, torch.tensor(test_y_np, dtype=torch.float32)).item()

        # Poisson deviance explained
        d2 = poisson_deviance_explained(test_y_np, log_rate_all)

        # Correlation between predicted rate and actual counts (per neuron, then average)
        n_neurons = test_y_np.shape[1]
        corrs = []
        for j in range(n_neurons):
            if test_y_np[:, j].std() > 0 and rate_pred[:, j].std() > 0:
                r, _ = sp_stats.pearsonr(test_y_np[:, j], rate_pred[:, j])
                corrs.append(r)
        mean_corr = np.mean(corrs) if corrs else 0.0

        # Mean predicted rate vs mean actual count (sanity check)
        mean_pred_rate = rate_pred.mean()
        mean_actual_count = test_y_np.mean()

        results[sn] = {
            'test_nll': test_nll,
            'd2': d2,
            'mean_corr': mean_corr,
            'mean_pred_rate': mean_pred_rate,
            'mean_actual_count': mean_actual_count,
        }
    return results


def extract_all_hidden_states(model, sessions_data, session_nums, region, device):
    """Extract hidden states for latent analysis."""
    hidden_states = {}
    for sn in session_nums:
        zscore = sessions_data[sn][region]['zscore']
        T = len(zscore)
        seqs = []
        for i in range(0, T - SEQ_LEN - PRED_BINS, STRIDE):
            seqs.append(zscore[i:i + SEQ_LEN])
        seqs_t = torch.tensor(np.array(seqs), dtype=torch.float32).to(device)
        model.eval()
        chunk_size = 128
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
    return {'pr': pr, 'variance': variance, 'speed': speed}


# =============================================================================
# MAIN
# =============================================================================

def main():
    print(f"Device: {DEVICE}")
    print(f"Config: {BIN_SIZE_MS}ms input bins, {PRED_WINDOW_MS}ms prediction window "
          f"({PRED_BINS} bins summed)")
    print(f"SEQ_LEN={SEQ_LEN} ({SEQ_LEN * BIN_SIZE_MS}ms context), STRIDE={STRIDE}")
    print(f"Architecture: D_shared={D_SHARED}, hidden={HIDDEN_SIZE}, "
          f"ODE_gate_hidden={ODE_GATE_HIDDEN}")
    print(f"ODE: solver={ODE_SOLVER}, step_size={ODE_STEP_SIZE}, dt={ODE_DT}")
    print(f"Loss: PoissonNLLLoss(log_input=True)")
    print(f"Training: lr={LEARNING_RATE}, batch={BATCH_SIZE}, patience={PATIENCE}")
    if TEST_MODE:
        print(f"\n*** TEST MODE: LHA Fed only ***\n")
    print()

    print("Loading session data...")
    sessions_data = load_all_sessions()
    print(f"Loaded {len(sessions_data)} sessions\n")

    fed_sessions = [sn for sn in sessions_data if sessions_data[sn]['state'] == 'Fed']
    fasted_sessions = [sn for sn in sessions_data if sessions_data[sn]['state'] == 'Fasted']

    if TEST_MODE:
        regions = ['lha']
        conditions = [('Fed', fed_sessions)]
    else:
        regions = ['lha', 'rsp']
        all_sessions = sorted(fed_sessions + fasted_sessions)
        conditions = [('Fed', fed_sessions), ('Fasted', fasted_sessions),
                      ('Combined', all_sessions)]

    all_results = []

    for region in regions:
        print(f"\n{'='*70}")
        print(f"  REGION: {region.upper()}")
        print(f"{'='*70}")

        for condition, session_nums in conditions:
            print(f"\n  --- {region.upper()} GRU-ODE 10ms Poisson: {condition} "
                  f"({len(session_nums)} sessions) ---")

            neuron_counts = {sn: sessions_data[sn][region]['n_neurons'] for sn in session_nums}
            print(f"    Neuron counts: {neuron_counts}")

            # Report mean firing rates per session
            for sn in sorted(session_nums):
                raw = sessions_data[sn][region]['raw']
                n_full = (len(raw) // PRED_BINS) * PRED_BINS
                mean_count_100ms = raw[:n_full].reshape(-1, PRED_BINS, raw.shape[1]).sum(axis=1).mean()
                print(f"    S{sn} mean count per 100ms bin: {mean_count_100ms:.2f}")

            train_ds = MultiSessionDataset(
                sessions_data, session_nums, region, SEQ_LEN, PRED_BINS,
                STRIDE, 'train', TRAIN_FRAC)
            test_ds = MultiSessionDataset(
                sessions_data, session_nums, region, SEQ_LEN, PRED_BINS,
                STRIDE, 'test', TRAIN_FRAC)
            print(f"    Train: {len(train_ds)}, Test: {len(test_ds)} samples")

            train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                      collate_fn=collate_by_session)
            test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                                     collate_fn=collate_by_session)

            model = PooledGRUODE(
                neuron_counts, D_SHARED, HIDDEN_SIZE, ODE_GATE_HIDDEN, PRED_BINS
            ).to(DEVICE)
            n_params = sum(p.numel() for p in model.parameters())
            print(f"    Parameters: {n_params:,}")
            print(f"    ODE calls per forward pass: {SEQ_LEN} + {PRED_BINS} = {SEQ_LEN + PRED_BINS}")

            t0 = time.time()
            history, best_epoch, best_val = train_pooled_model(
                model, train_loader, test_loader, NUM_EPOCHS, PATIENCE, LEARNING_RATE, DEVICE
            )
            elapsed = time.time() - t0
            print(f"    Done in {elapsed:.1f}s ({elapsed/60:.1f} min), "
                  f"best epoch: {best_epoch+1}, val loss: {best_val:.6f}")

            # Evaluate
            per_session = evaluate_per_session(
                model, sessions_data, session_nums, region, DEVICE)
            hidden_states = extract_all_hidden_states(
                model, sessions_data, session_nums, region, DEVICE)

            for sn in sorted(session_nums):
                r = per_session[sn]
                metrics = compute_latent_metrics(hidden_states[sn])
                info = sessions_data[sn]
                all_results.append({
                    'model_type': 'combined' if condition == 'Combined' else 'condition_specific',
                    'bin_size_ms': BIN_SIZE_MS,
                    'pred_window_ms': PRED_WINDOW_MS,
                    'region': region.upper(),
                    'condition': condition,
                    'session': sn,
                    'state': info['state'],
                    'phase': info['phase'],
                    'n_neurons': info[region]['n_neurons'],
                    'test_nll': r['test_nll'],
                    'd2': r['d2'],
                    'mean_corr': r['mean_corr'],
                    'mean_pred_rate': r['mean_pred_rate'],
                    'mean_actual_count': r['mean_actual_count'],
                    'pr': metrics['pr'],
                    'variance': metrics['variance'],
                    'speed': metrics['speed'],
                    'training_time_s': elapsed,
                    'best_epoch': best_epoch + 1,
                })
                print(f"    S{sn} ({info['phase']}): D2={r['d2']:.4f}, "
                      f"corr={r['mean_corr']:.4f}, "
                      f"pred_rate={r['mean_pred_rate']:.2f} vs actual={r['mean_actual_count']:.2f}, "
                      f"PR={metrics['pr']:.2f}")

            # Save model
            tag = '10ms_poisson_test' if TEST_MODE else '10ms_poisson'
            model_path = Path("data") / f"gru_ode_{tag}_{region}_{condition.lower()}_model.pt"
            torch.save({
                'model_state_dict': model.state_dict(),
                'neuron_counts': neuron_counts,
                'config': {
                    'd_shared': D_SHARED, 'hidden_size': HIDDEN_SIZE,
                    'gate_hidden': ODE_GATE_HIDDEN, 'seq_len': SEQ_LEN,
                    'pred_bins': PRED_BINS, 'pred_window_ms': PRED_WINDOW_MS,
                    'stride': STRIDE, 'bin_size_ms': BIN_SIZE_MS,
                    'ode_solver': ODE_SOLVER, 'ode_step_size': ODE_STEP_SIZE,
                    'ode_dt': ODE_DT, 'loss': 'PoissonNLL',
                },
                'history': history, 'best_epoch': best_epoch,
            }, model_path)
            print(f"    Saved: {model_path}")

            hist_df = pd.DataFrame(history)
            hist_path = Path("data") / f"gru_ode_{tag}_{region}_{condition.lower()}_history.csv"
            hist_df.to_csv(hist_path, index=False)

    # Save results
    df = pd.DataFrame(all_results)
    tag = '10ms_poisson_test' if TEST_MODE else '10ms_poisson'
    results_path = Path("data") / f"gru_ode_{tag}_results.csv"
    df.to_csv(results_path, index=False)
    print(f"\nResults saved: {results_path}")

    # Print summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for _, row in df.iterrows():
        print(f"  S{row['session']} ({row['state']} {row['phase']}): "
              f"D2={row['d2']:.4f}, corr={row['mean_corr']:.4f}, "
              f"pred_rate={row['mean_pred_rate']:.2f} vs actual={row['mean_actual_count']:.2f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
