"""
GRU-ODE with Learnable Time Constants (tau) -- 10ms Poisson, RSP Fed + Fasted
===============================================================================
Trains With-tau models only. Compares against existing No-tau baselines
from gru_ode_10ms_poisson_results.csv.

  dh/dt = (1/tau) * (1 - z(h)) * (n(h) - h)

tau = exp(log_tau), one per hidden dimension, initialized at 1.0.
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

BIN_SIZE_MS = 10
PRED_WINDOW_MS = 100
PRED_BINS = PRED_WINDOW_MS // BIN_SIZE_MS
FS = 30000
BIN_SAMPLES = int(BIN_SIZE_MS * FS / 1000)

SEQ_LEN = 50
STRIDE = 50
D_SHARED = 32
HIDDEN_SIZE = 32
ODE_GATE_HIDDEN = 64
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
NUM_EPOCHS = 150
PATIENCE = 20
TRAIN_FRAC = 0.8
GRAD_CLIP = 1.0

ODE_SOLVER = 'rk4'
ODE_STEP_SIZE = 1.0
ODE_DT = 1.0

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

REGION = 'rsp'


# =============================================================================
# DATA LOADING
# =============================================================================

def get_good_units_rsp(sorted_path_obj):
    ci = sorted_path_obj / "cluster_info.tsv"
    if not ci.exists():
        return np.array([])
    df = pd.read_csv(ci, sep='\t')
    if 'depth' not in df.columns:
        return np.array([])
    label_col = None
    if 'group' in df.columns and df['group'].eq('good').any():
        label_col = 'group'
    elif 'KSLabel' in df.columns:
        label_col = 'KSLabel'
    if label_col is None:
        return np.array([])
    good = df[df[label_col] == 'good']
    return good[good['depth'] >= RSP_DEPTH_MIN]['cluster_id'].values


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
    raw_data = data.copy()
    means = data.mean(axis=0, keepdims=True)
    stds = data.std(axis=0, keepdims=True)
    stds[stds < 1e-8] = 1.0
    zscore_data = (data - means) / stds
    return zscore_data, raw_data, n_bins


def load_sessions():
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
        rsp_ids = get_good_units_rsp(sorted_path)
        if len(rsp_ids) < 3:
            print(f"  Session {sess_num}: too few RSP units ({len(rsp_ids)}), skipping")
            continue
        rsp_zscore, rsp_raw, rsp_bins = bin_spike_trains(sorting, rsp_ids)
        rsp_mean_rate = rsp_raw.mean(axis=0).mean() * (1000 / BIN_SIZE_MS)
        sessions_data[sess_num] = {
            'rsp': {
                'zscore': rsp_zscore, 'raw': rsp_raw,
                'n_neurons': len(rsp_ids), 'n_bins': rsp_bins,
            },
            'state': info['state'],
            'phase': info['phase'],
        }
        print(f"  Session {sess_num}: {info['state']} {info['phase']}, "
              f"RSP={len(rsp_ids)} ({rsp_bins} bins, {rsp_mean_rate:.1f} Hz)")
    return sessions_data


# =============================================================================
# DATASET
# =============================================================================

class MultiSessionDataset(Dataset):
    def __init__(self, sessions_data, session_nums, seq_len, pred_bins,
                 stride, split='train', train_frac=0.8):
        self.seq_len = seq_len
        self.pred_bins = pred_bins
        self.samples = []
        for sn in session_nums:
            data = sessions_data[sn]['rsp']['zscore']
            T = len(data)
            split_idx = int(T * train_frac)
            if split == 'train':
                start, end = 0, split_idx
            else:
                start, end = split_idx, T
            for i in range(0, end - start - seq_len - pred_bins, stride):
                self.samples.append((sn, start + i))
        self.sessions_data = sessions_data

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sn, start = self.samples[idx]
        zscore = self.sessions_data[sn]['rsp']['zscore']
        raw = self.sessions_data[sn]['rsp']['raw']
        x = torch.tensor(zscore[start:start + self.seq_len], dtype=torch.float32)
        y = torch.tensor(
            raw[start + self.seq_len:start + self.seq_len + self.pred_bins].sum(axis=0),
            dtype=torch.float32
        )
        return x, y, sn


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
# GRU-ODE WITH TAU
# =============================================================================

class GRUODEFuncTau(nn.Module):
    """dh/dt = (1/tau) * (1 - z(h)) * (n(h) - h)"""
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
        self.log_tau = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, t, h):
        z = self.update_gate(h)
        n = self.candidate(h)
        tau = torch.exp(self.log_tau)
        return (1.0 / tau) * (1 - z) * (n - h)


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

        self.ode_func = GRUODEFuncTau(hidden_size, gate_hidden)
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
        for k in range(x.shape[1]):
            h = self._ode_evolve(h)
            x_proj = self.input_projections[sn_key](x[:, k, :])
            h = self.obs_cell(x_proj, h)
        for _ in range(self.pred_steps):
            h = self._ode_evolve(h)
        shared_out = self.fc_shared(h)
        log_rate = self.output_projections[sn_key](shared_out)
        return log_rate

    def extract_hidden_states(self, x, session_num):
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
# TRAINING
# =============================================================================

def train_model(model, train_loader, val_loader, n_epochs, patience, lr, device, label=""):
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
                y = data['y'].to(device)
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
            print(f"      [{label}] Early stopping at epoch {epoch+1}")
            break

        if (epoch + 1) % 2 == 0 or epoch == 0:
            print(f"      [{label}] Epoch {epoch+1}: train={mean_train:.6f} val={mean_val:.6f} "
                  f"({epoch_time:.1f}s/epoch)")

    if best_state is not None:
        model.load_state_dict(best_state)
    return history, best_epoch, best_val_loss


# =============================================================================
# EVALUATION
# =============================================================================

def poisson_deviance_explained(y_true, log_rate_pred):
    rate_pred = np.exp(np.clip(log_rate_pred, -20, 20))
    eps = 1e-8
    with np.errstate(divide='ignore', invalid='ignore'):
        term1 = np.where(y_true > 0, y_true * np.log(y_true / (rate_pred + eps)), 0.0)
    dev_model = 2 * np.sum(term1 - (y_true - rate_pred))
    mean_rate = np.maximum(y_true.mean(axis=0, keepdims=True), eps)
    with np.errstate(divide='ignore', invalid='ignore'):
        term1_null = np.where(y_true > 0, y_true * np.log(y_true / mean_rate), 0.0)
    dev_null = 2 * np.sum(term1_null - (y_true - mean_rate))
    if dev_null <= 0:
        return 0.0
    return 1 - dev_model / dev_null


def evaluate_per_session(model, sessions_data, session_nums, device):
    results = {}
    criterion = nn.PoissonNLLLoss(log_input=True, reduction='mean')
    for sn in session_nums:
        zscore = sessions_data[sn]['rsp']['zscore']
        raw = sessions_data[sn]['rsp']['raw']
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
        log_rate_t = torch.tensor(log_rate_all, dtype=torch.float32)
        test_nll = criterion(log_rate_t, torch.tensor(test_y_np, dtype=torch.float32)).item()
        d2 = poisson_deviance_explained(test_y_np, log_rate_all)
        n_neurons = test_y_np.shape[1]
        corrs = []
        for j in range(n_neurons):
            if test_y_np[:, j].std() > 0 and rate_pred[:, j].std() > 0:
                r, _ = sp_stats.pearsonr(test_y_np[:, j], rate_pred[:, j])
                corrs.append(r)
        mean_corr = np.mean(corrs) if corrs else 0.0
        mean_pred_rate = rate_pred.mean()
        mean_actual_count = test_y_np.mean()
        results[sn] = {
            'test_nll': test_nll, 'd2': d2, 'mean_corr': mean_corr,
            'mean_pred_rate': mean_pred_rate, 'mean_actual_count': mean_actual_count,
        }
    return results


def extract_all_hidden_states(model, sessions_data, session_nums, device):
    hidden_states = {}
    for sn in session_nums:
        zscore = sessions_data[sn]['rsp']['zscore']
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
    print(f"GRU-ODE With-tau Only -- 10ms Poisson, RSP Fed + Fasted")
    print(f"Equation: dh/dt = (1/tau) * (1 - z(h)) * (n(h) - h)")
    print(f"Comparing against existing No-tau baselines from gru_ode_10ms_poisson_results.csv")
    print()

    # Load existing baselines
    baseline_df = pd.read_csv(Path("data") / "gru_ode_10ms_poisson_results.csv")
    baseline_rsp = baseline_df[
        (baseline_df['region'] == 'RSP') &
        (baseline_df['model_type'] == 'condition_specific')
    ]
    print(f"Loaded {len(baseline_rsp)} baseline RSP results\n")

    # Load data
    print("Loading sessions...")
    sessions_data = load_sessions()
    all_sessions = sorted(sessions_data.keys())
    print(f"Loaded {len(all_sessions)} sessions\n")

    fed_sessions = [sn for sn in all_sessions if sessions_data[sn]['state'] == 'Fed']
    fasted_sessions = [sn for sn in all_sessions if sessions_data[sn]['state'] == 'Fasted']

    all_tau_results = []

    for condition, session_nums in [('Fed', fed_sessions), ('Fasted', fasted_sessions)]:
        print(f"\n{'='*60}")
        print(f"  RSP {condition} -- With-tau")
        print(f"{'='*60}")

        neuron_counts = {sn: sessions_data[sn]['rsp']['n_neurons'] for sn in session_nums}
        print(f"  Neuron counts: {neuron_counts}")

        train_ds = MultiSessionDataset(sessions_data, session_nums, SEQ_LEN, PRED_BINS,
                                       STRIDE, 'train', TRAIN_FRAC)
        test_ds = MultiSessionDataset(sessions_data, session_nums, SEQ_LEN, PRED_BINS,
                                      STRIDE, 'test', TRAIN_FRAC)
        print(f"  Train: {len(train_ds)}, Test: {len(test_ds)} samples")

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                  collate_fn=collate_by_session)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                                 collate_fn=collate_by_session)

        model = PooledGRUODE(neuron_counts, D_SHARED, HIDDEN_SIZE, ODE_GATE_HIDDEN,
                             PRED_BINS).to(DEVICE)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {n_params:,}")

        label = f"RSP-{condition}"
        t0 = time.time()
        history, best_epoch, best_val = train_model(
            model, train_loader, test_loader, NUM_EPOCHS, PATIENCE, LEARNING_RATE,
            DEVICE, label
        )
        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.1f}s ({elapsed/60:.1f} min), "
              f"best epoch: {best_epoch+1}, val loss: {best_val:.6f}")

        # Evaluate
        per_session = evaluate_per_session(model, sessions_data, session_nums, DEVICE)
        hidden_states = extract_all_hidden_states(model, sessions_data, session_nums, DEVICE)

        for sn in session_nums:
            r = per_session[sn]
            metrics = compute_latent_metrics(hidden_states[sn])
            all_tau_results.append({
                'region': 'RSP',
                'condition': condition,
                'session': sn,
                'state': sessions_data[sn]['state'],
                'phase': sessions_data[sn]['phase'],
                'n_neurons': sessions_data[sn]['rsp']['n_neurons'],
                'd2': r['d2'],
                'mean_corr': r['mean_corr'],
                'test_nll': r['test_nll'],
                'mean_pred_rate': r['mean_pred_rate'],
                'mean_actual_count': r['mean_actual_count'],
                'pr': metrics['pr'],
                'variance': metrics['variance'],
                'speed': metrics['speed'],
            })
            print(f"  S{sn} ({sessions_data[sn]['phase']}): D2={r['d2']:.4f}, "
                  f"corr={r['mean_corr']:.4f}, PR={metrics['pr']:.2f}")

        # Save model
        model_path = Path("data") / f"gru_ode_10ms_tau_rsp_{condition.lower()}_model.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'neuron_counts': neuron_counts,
            'config': {
                'd_shared': D_SHARED, 'hidden_size': HIDDEN_SIZE,
                'gate_hidden': ODE_GATE_HIDDEN, 'seq_len': SEQ_LEN,
                'pred_bins': PRED_BINS, 'stride': STRIDE,
                'bin_size_ms': BIN_SIZE_MS, 'pred_window_ms': PRED_WINDOW_MS,
                'ode_solver': ODE_SOLVER, 'loss': 'PoissonNLL',
            },
            'history': history, 'best_epoch': best_epoch,
        }, model_path)
        print(f"  Saved: {model_path}")

        # Print tau values
        log_tau = model.ode_func.log_tau.detach().cpu().numpy()
        tau_vals = np.exp(log_tau)
        print(f"\n  Learned tau: min={tau_vals.min():.4f}, max={tau_vals.max():.4f}, "
              f"mean={tau_vals.mean():.4f}, std={tau_vals.std():.4f}")
        sorted_tau = np.sort(tau_vals)
        print(f"  5 smallest: {sorted_tau[:5]}")
        print(f"  5 largest:  {sorted_tau[-5:]}")

        # Save tau values
        tau_save = pd.DataFrame({
            'dimension': np.arange(HIDDEN_SIZE),
            'log_tau': log_tau,
            'tau': tau_vals,
        })
        tau_save.to_csv(
            Path("data") / f"gru_ode_10ms_tau_rsp_{condition.lower()}_values.csv",
            index=False
        )

    # =========================================================================
    # Combine with baselines and compare
    # =========================================================================

    tau_df = pd.DataFrame(all_tau_results)
    tau_df.to_csv(Path("data") / "gru_ode_10ms_tau_rsp_results.csv", index=False)

    print(f"\n{'='*60}")
    print("COMPARISON: No-tau (baseline) vs With-tau")
    print(f"{'='*60}")

    for condition in ['Fed', 'Fasted']:
        base = baseline_rsp[baseline_rsp['condition'] == condition]
        tau_c = tau_df[tau_df['condition'] == condition]
        print(f"\n  RSP {condition}:")
        for metric in ['d2', 'mean_corr', 'test_nll', 'pr', 'variance', 'speed']:
            bv = base[metric].values
            tv = tau_c[metric].values
            print(f"    {metric:16s}: No-tau={np.mean(bv):.4f}  "
                  f"With-tau={np.mean(tv):.4f}  "
                  f"diff={np.mean(tv) - np.mean(bv):+.4f}")

    # =========================================================================
    # Also load LHA tau results for cross-region comparison
    # =========================================================================

    print(f"\n{'='*60}")
    print("CROSS-REGION TAU COMPARISON (LHA vs RSP)")
    print(f"{'='*60}")

    for condition in ['Fed', 'Fasted']:
        lha_tau_path = Path("data") / f"gru_ode_10ms_tau_lha_{condition.lower()}_values.csv"
        rsp_tau_path = Path("data") / f"gru_ode_10ms_tau_rsp_{condition.lower()}_values.csv"

        if lha_tau_path.exists() and rsp_tau_path.exists():
            lha_tau = pd.read_csv(lha_tau_path)['tau'].values
            rsp_tau = pd.read_csv(rsp_tau_path)['tau'].values
            _, p = sp_stats.mannwhitneyu(lha_tau, rsp_tau, alternative='two-sided')
            print(f"\n  {condition}:")
            print(f"    LHA tau: mean={lha_tau.mean():.4f}, std={lha_tau.std():.4f}, "
                  f"range=[{lha_tau.min():.4f}, {lha_tau.max():.4f}]")
            print(f"    RSP tau: mean={rsp_tau.mean():.4f}, std={rsp_tau.std():.4f}, "
                  f"range=[{rsp_tau.min():.4f}, {rsp_tau.max():.4f}]")
            print(f"    Mann-Whitney U p={p:.4f} {'*' if p < 0.05 else 'ns'}")
        else:
            print(f"\n  {condition}: LHA tau file not found, skipping comparison")

    # =========================================================================
    # Figure
    # =========================================================================

    print("\nGenerating figures...")

    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    fig.suptitle("GRU-ODE With-tau vs Baseline -- 10ms Poisson, RSP Fed + Fasted",
                 fontsize=14, fontweight='bold')

    colors_model = {'No-tau': '#607D8B', 'With-tau': '#E91E63'}

    for row, condition in enumerate(['Fed', 'Fasted']):
        base = baseline_rsp[baseline_rsp['condition'] == condition]
        tau_c = tau_df[tau_df['condition'] == condition]
        session_nums = sorted(base['session'].values)

        # D2, Corr, PR bars
        for col, (metric, mlabel) in enumerate([
            ('d2', 'Poisson D2'), ('mean_corr', 'Correlation'), ('pr', 'Participation Ratio')
        ]):
            ax = axes[row, col]
            for mi, (mname, mdf, color) in enumerate([
                ('No-tau', base, colors_model['No-tau']),
                ('With-tau', tau_c, colors_model['With-tau'])
            ]):
                vals = mdf[metric].values
                ax.bar(mi, np.mean(vals), width=0.5, color=color, alpha=0.6, label=mname)
                ax.errorbar(mi, np.mean(vals), yerr=np.std(vals)/np.sqrt(len(vals)),
                            color='black', capsize=5)
                jitter = np.random.uniform(-0.1, 0.1, len(vals))
                ax.scatter([mi + j for j in jitter], vals, color=color,
                           edgecolors='black', s=60, zorder=5)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(['No-tau', 'With-tau'])
            ax.set_title(f"RSP {condition} -- {mlabel}", fontsize=11)
            if row == 0 and col == 0:
                ax.legend()

        # Tau distribution
        ax = axes[row, 3]
        rsp_tau_path = Path("data") / f"gru_ode_10ms_tau_rsp_{condition.lower()}_values.csv"
        tau_vals = pd.read_csv(rsp_tau_path)['tau'].values
        sorted_idx = np.argsort(tau_vals)
        bar_colors = ['#1976D2' if tau_vals[i] < 1.0 else '#E91E63' for i in sorted_idx]
        ax.barh(np.arange(HIDDEN_SIZE), tau_vals[sorted_idx], color=bar_colors, alpha=0.7)
        ax.axvline(x=1.0, color='black', linestyle='--', linewidth=1, label='tau=1.0')
        ax.set_xlabel('tau')
        ax.set_ylabel('Dim (sorted)')
        ax.set_title(f"RSP {condition} tau\n(mean={tau_vals.mean():.3f}, "
                     f"range=[{tau_vals.min():.3f}, {tau_vals.max():.3f}])", fontsize=10)
        ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(Path("figures") / "gru_ode_10ms_tau_rsp_comparison.png",
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: figures/gru_ode_10ms_tau_rsp_comparison.png")

    print("\nAll done!")


if __name__ == "__main__":
    main()
