"""
Linear Behavioral Decoding from GRU-ODE 10ms Latent States
============================================================
Decode 14 behavioral variables from the 32D latent states of the
combined GRU-ODE models (LHA and RSP).

- Continuous variables (4 distance measures): Ridge regression, R²
- Binary variables (3 states + 7 exploration subtypes): Logistic regression, ROC-AUC
- Per-session decoders, 80/20 temporal split
- Fed vs Fasted and LHA vs RSP statistical comparisons
"""

import yaml
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import spikeinterface.extractors as se
import torch
import torch.nn as nn
from torchdiffeq import odeint
from sklearn.linear_model import RidgeCV, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score
from scipy.stats import mannwhitneyu, wilcoxon, pearsonr
import warnings
import time

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

BIN_SIZE_MS = 10
FS = 30000
BIN_SAMPLES = int(BIN_SIZE_MS * FS / 1000)
SUBSAMPLE_RATIO = 10  # 10ms -> 100ms

TRAIN_FRAC = 0.8
N_SHUFFLES = 100

# ODE solver settings (must match training)
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

CONTINUOUS_TARGETS = [
    'Distance to Foraging arena',
    'Distance to Home',
    'Distance to Pot-2',
    'Distance to Pot-4',
]

BINARY_TARGETS = [
    'Feeding',
    'Digging',
    'Grooming',
    'Quick and hasty exploration at home',
    'Quick one loop at home',
    'Longer exploration at home',
    'Transition wall exploration',
    'Hiding in corners',
    'Incomplete home return',
    'Contemplation at T-zone',
]

ALL_TARGETS = CONTINUOUS_TARGETS + BINARY_TARGETS

# Short names for plots
SHORT_NAMES = {
    'Distance to Foraging arena': 'Dist Arena',
    'Distance to Home': 'Dist Home',
    'Distance to Pot-2': 'Dist Pot-2',
    'Distance to Pot-4': 'Dist Pot-4',
    'Feeding': 'Feeding',
    'Digging': 'Digging',
    'Grooming': 'Grooming',
    'Quick and hasty exploration at home': 'Quick Hasty Exp',
    'Quick one loop at home': 'Quick Loop',
    'Longer exploration at home': 'Long Exp Home',
    'Transition wall exploration': 'Trans Wall Exp',
    'Hiding in corners': 'Hiding Corners',
    'Incomplete home return': 'Incomplete Return',
    'Contemplation at T-zone': 'Contemp T-zone',
}

FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


# =============================================================================
# MODEL CLASSES (copied from gru_ode_10ms.py)
# =============================================================================

class GRUODEFunc(nn.Module):
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
    """Bin spike trains at 10ms. Returns z-scored data for model input."""
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
    zscore_data = (data - means) / stds
    return zscore_data, n_bins


def load_all_sessions():
    print("Loading session data...")
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

        lha_zscore, lha_bins = bin_spike_trains(sorting, lha_ids)
        rsp_zscore, rsp_bins = bin_spike_trains(sorting, rsp_ids)

        sessions_data[sess_num] = {
            'lha': {'zscore': lha_zscore, 'n_neurons': len(lha_ids), 'n_bins': lha_bins},
            'rsp': {'zscore': rsp_zscore, 'n_neurons': len(rsp_ids), 'n_bins': rsp_bins},
            'state': info['state'],
            'phase': info['phase'],
        }
        print(f"  Session {sess_num}: {info['state']} {info['phase']}, "
              f"LHA={len(lha_ids)}, RSP={len(rsp_ids)}")
    return sessions_data


def load_behavior(session_num):
    sp = paths_config['single_probe']['coordinates_1']['mouse01']['sessions']
    key = f"session_{session_num}"
    behav_path = Path(sp[key]['behavior'])
    behav_raw = pd.read_csv(behav_path, index_col=0)
    behav_df = behav_raw.T.reset_index(drop=True)
    behav_df.columns = behav_df.columns.str.strip()
    for col in behav_df.columns:
        behav_df[col] = pd.to_numeric(behav_df[col], errors='coerce')
    return behav_df


def load_model(region):
    model_path = DATA_DIR / f"gru_ode_10ms_poisson_{region}_combined_model.pt"
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    neuron_counts = checkpoint['neuron_counts']
    config = checkpoint['config']
    model = PooledGRUODE(
        neuron_counts, config['d_shared'], config['hidden_size'],
        config['gate_hidden'], config['pred_bins'],
    ).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"  Loaded {region.upper()} combined model ({len(neuron_counts)} sessions)")
    return model


# =============================================================================
# HIDDEN STATE EXTRACTION
# =============================================================================

def extract_full_session_hidden_states(model, zscore_data, session_num, device,
                                       report_every=50000):
    """Process entire session sequentially: one 32D hidden state per 10ms bin."""
    T, n_neurons = zscore_data.shape
    sn_key = str(session_num)
    h = torch.zeros(1, model.hidden_size, device=device)

    # Pre-allocate on CPU
    all_h = np.zeros((T, model.hidden_size), dtype=np.float32)

    with torch.no_grad():
        for k in range(T):
            h = model._ode_evolve(h)
            x_bin = torch.tensor(zscore_data[k:k+1], dtype=torch.float32, device=device)
            x_proj = model.input_projections[sn_key](x_bin)
            h = model.obs_cell(x_proj, h)
            all_h[k] = h.cpu().numpy()

            if (k + 1) % report_every == 0:
                print(f"      {k+1}/{T} bins ({(k+1)/T*100:.0f}%)")

    return all_h  # (T, 32)


def get_hidden_states(model, sessions_data, region, session_num, device):
    """Get hidden states with caching."""
    cache_path = DATA_DIR / f"gru_ode_10ms_hidden_{region}_s{session_num}.npy"
    if cache_path.exists():
        print(f"    S{session_num}: loading cached hidden states")
        return np.load(cache_path)

    print(f"    S{session_num}: extracting hidden states ({sessions_data[session_num][region]['n_bins']} bins)...")
    t0 = time.time()
    zscore = sessions_data[session_num][region]['zscore']
    h_all = extract_full_session_hidden_states(model, zscore, session_num, device)
    elapsed = time.time() - t0
    print(f"    S{session_num}: done in {elapsed:.0f}s, shape={h_all.shape}")

    np.save(cache_path, h_all)
    return h_all


# =============================================================================
# DECODING FUNCTIONS
# =============================================================================

def fit_ridge_decoder(H_train, y_train, H_test, y_test):
    """Ridge regression for continuous variables. Returns R², Pearson r, best alpha."""
    scaler_x = StandardScaler()
    H_tr = scaler_x.fit_transform(H_train)
    H_te = scaler_x.transform(H_test)

    scaler_y = StandardScaler()
    y_tr = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_te = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

    alphas = np.logspace(-3, 4, 20)
    ridge = RidgeCV(alphas=alphas)
    ridge.fit(H_tr, y_tr)

    y_pred = ridge.predict(H_te)
    ss_res = np.sum((y_te - y_pred) ** 2)
    ss_tot = np.sum((y_te - np.mean(y_tr)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    corr, _ = pearsonr(y_te, y_pred)

    return r2, corr, ridge.alpha_, ridge.coef_


def fit_logistic_decoder(H_train, y_train, H_test, y_test):
    """Logistic regression for binary variables. Returns AUC, F1, bal_acc, best C, status."""
    n_pos_train = int(y_train.sum())
    n_pos_test = int(y_test.sum())

    if n_pos_train < 5 or n_pos_test < 2:
        return np.nan, np.nan, np.nan, np.nan, 'insufficient_positives', None

    scaler_x = StandardScaler()
    H_tr = scaler_x.fit_transform(H_train)
    H_te = scaler_x.transform(H_test)

    clf = LogisticRegressionCV(
        Cs=10, cv=5, penalty='l2', solver='lbfgs',
        max_iter=1000, class_weight='balanced',
        scoring='roc_auc',
    )
    clf.fit(H_tr, y_train)

    y_prob = clf.predict_proba(H_te)[:, 1]
    y_pred = clf.predict(H_te)

    auc = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    bal_acc = balanced_accuracy_score(y_test, y_pred)

    return auc, f1, bal_acc, clf.C_[0], 'ok', clf.coef_[0]


def shuffle_baseline_r2(H_test, y_test, ridge_model, scaler_x, scaler_y, n_shuffles=N_SHUFFLES):
    """Compute chance-level R² by permuting test labels."""
    rng = np.random.default_rng(42)
    H_te = scaler_x.transform(H_test)
    shuffled = []
    for _ in range(n_shuffles):
        y_shuf = rng.permutation(y_test)
        y_shuf_s = scaler_y.transform(y_shuf.reshape(-1, 1)).ravel()
        y_pred = ridge_model.predict(H_te)
        ss_res = np.sum((y_shuf_s - y_pred) ** 2)
        ss_tot = np.sum((y_shuf_s - np.mean(y_shuf_s)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        shuffled.append(r2)
    return np.percentile(shuffled, 95)


def shuffle_baseline_auc(H_test, y_test, clf_model, scaler_x, n_shuffles=N_SHUFFLES):
    """Compute chance-level AUC by permuting test labels."""
    rng = np.random.default_rng(42)
    H_te = scaler_x.transform(H_test)
    y_prob = clf_model.predict_proba(H_te)[:, 1]
    shuffled = []
    for _ in range(n_shuffles):
        y_shuf = rng.permutation(y_test)
        if y_shuf.sum() < 2 or y_shuf.sum() > len(y_shuf) - 2:
            continue
        try:
            auc = roc_auc_score(y_shuf, y_prob)
            shuffled.append(auc)
        except ValueError:
            continue
    return np.percentile(shuffled, 95) if shuffled else np.nan


def decode_session(H_100ms, behav_df, session_num):
    """Run all decoders for one session. Returns list of result dicts."""
    info = SESSION_INFO[session_num]
    n_neural = len(H_100ms)
    n_behav = len(behav_df)
    n_use = min(n_neural, n_behav)

    H = H_100ms[:n_use]
    split_idx = int(n_use * TRAIN_FRAC)
    H_train, H_test = H[:split_idx], H[split_idx:]

    results = []

    # --- Continuous targets ---
    for var in CONTINUOUS_TARGETS:
        y = behav_df[var].values[:n_use].astype(np.float64)
        y = np.nan_to_num(y, nan=0.0)
        y_train, y_test = y[:split_idx], y[split_idx:]

        r2, corr, alpha, coef = fit_ridge_decoder(H_train, y_train, H_test, y_test)

        # Shuffle baseline
        scaler_x = StandardScaler().fit(H_train)
        scaler_y = StandardScaler().fit(y_train.reshape(-1, 1))
        ridge_tmp = RidgeCV(alphas=[alpha]).fit(
            scaler_x.transform(H_train),
            scaler_y.transform(y_train.reshape(-1, 1)).ravel()
        )
        shuf_r2 = shuffle_baseline_r2(H_test, y_test, ridge_tmp, scaler_x, scaler_y)

        results.append({
            'session': session_num, 'state': info['state'], 'phase': info['phase'],
            'variable': var, 'var_type': 'continuous',
            'r2': r2, 'pearson_r': corr, 'roc_auc': np.nan, 'f1': np.nan,
            'bal_acc': np.nan, 'shuffle_baseline': shuf_r2,
            'best_reg': alpha, 'status': 'ok',
            'n_positive': np.nan, 'n_test': len(y_test),
            'weights': coef,
        })

    # --- Binary targets ---
    for var in BINARY_TARGETS:
        y = behav_df[var].values[:n_use].astype(np.float64)
        y = np.nan_to_num(y, nan=0.0)
        y = (y > 0).astype(np.float64)
        y_train, y_test = y[:split_idx], y[split_idx:]
        n_pos = int(y.sum())
        n_pos_test = int(y_test.sum())

        auc, f1_val, bal_acc, best_C, status, coef = fit_logistic_decoder(
            H_train, y_train, H_test, y_test
        )

        shuf_auc = np.nan
        if status == 'ok':
            scaler_x = StandardScaler().fit(H_train)
            clf_tmp = LogisticRegressionCV(
                Cs=[best_C], cv=5, penalty='l2', solver='lbfgs',
                max_iter=1000, class_weight='balanced',
            ).fit(scaler_x.transform(H_train), y_train)
            shuf_auc = shuffle_baseline_auc(H_test, y_test, clf_tmp, scaler_x)

        results.append({
            'session': session_num, 'state': info['state'], 'phase': info['phase'],
            'variable': var, 'var_type': 'binary',
            'r2': np.nan, 'pearson_r': np.nan, 'roc_auc': auc, 'f1': f1_val,
            'bal_acc': bal_acc, 'shuffle_baseline': shuf_auc,
            'best_reg': best_C, 'status': status,
            'n_positive': n_pos, 'n_test': len(y_test),
            'weights': coef,
        })

    return results


# =============================================================================
# STATISTICAL COMPARISONS
# =============================================================================

def compare_groups(group1, group2):
    """Mann-Whitney U on two groups. Returns p, Cohen's d."""
    g1 = [v for v in group1 if np.isfinite(v)]
    g2 = [v for v in group2 if np.isfinite(v)]
    if len(g1) < 2 or len(g2) < 2:
        return np.nan, np.nan
    _, p = mannwhitneyu(g1, g2, alternative='two-sided')
    pooled = np.concatenate([g1, g2])
    d = (np.mean(g1) - np.mean(g2)) / (np.std(pooled) + 1e-10)
    return p, d


def compare_paired(vals1, vals2):
    """Wilcoxon signed-rank for paired comparison. Returns p, d."""
    paired = [(a, b) for a, b in zip(vals1, vals2)
              if np.isfinite(a) and np.isfinite(b)]
    if len(paired) < 4:
        return np.nan, np.nan
    v1, v2 = zip(*paired)
    v1, v2 = np.array(v1), np.array(v2)
    _, p = wilcoxon(v1, v2)
    diff = v1 - v2
    d = np.mean(diff) / (np.std(diff) + 1e-10)
    return p, d


# =============================================================================
# FIGURES
# =============================================================================

def plot_heatmap(all_results, region_results):
    """Figure 1: Decoding performance heatmap (LHA and RSP side by side)."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    for ax_idx, region in enumerate(['lha', 'rsp']):
        ax = axes[ax_idx]
        results = region_results[region]

        # Build matrix: rows=variables, columns=sessions
        matrix = np.full((len(ALL_TARGETS), 8), np.nan)
        for r in results:
            row = ALL_TARGETS.index(r['variable'])
            col = r['session'] - 1
            if r['var_type'] == 'continuous':
                matrix[row, col] = r['r2']
            else:
                matrix[row, col] = r['roc_auc']

        ylabels = [SHORT_NAMES[v] for v in ALL_TARGETS]
        xlabels = [f"S{s}\n{SESSION_INFO[s]['state'][:3]}\n{SESSION_INFO[s]['phase'][:3]}"
                   for s in range(1, 9)]

        sns.heatmap(matrix, ax=ax, cmap='RdYlGn', center=0.5,
                    vmin=-0.1, vmax=1.0,
                    xticklabels=xlabels, yticklabels=ylabels,
                    annot=True, fmt='.2f', annot_kws={'size': 7},
                    linewidths=0.5, cbar_kws={'shrink': 0.8})

        # Divider between continuous and binary
        ax.axhline(y=len(CONTINUOUS_TARGETS), color='black', linewidth=2)

        ax.set_title(f'{region.upper()} Decoding Performance\n(R² for continuous, AUC for binary)',
                     fontsize=12, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('')

    plt.tight_layout()
    path = FIGURES_DIR / 'gru_ode_10ms_decoding_heatmap.png'
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_fed_vs_fasted(all_results, region_results):
    """Figure 2: Fed vs Fasted grouped bar plots."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    fed_sessions = [1, 2, 3, 4]
    fasted_sessions = [5, 6, 7, 8]

    for row_idx, region in enumerate(['lha', 'rsp']):
        results = region_results[region]

        # Continuous variables
        ax = axes[row_idx, 0]
        vars_cont = CONTINUOUS_TARGETS
        x_pos = np.arange(len(vars_cont))
        width = 0.35

        fed_means, fed_sems, fasted_means, fasted_sems = [], [], [], []
        fed_pts, fasted_pts = [], []

        for var in vars_cont:
            fed_vals = [r['r2'] for r in results
                        if r['variable'] == var and r['session'] in fed_sessions]
            fas_vals = [r['r2'] for r in results
                        if r['variable'] == var and r['session'] in fasted_sessions]
            fed_means.append(np.nanmean(fed_vals))
            fed_sems.append(np.nanstd(fed_vals) / np.sqrt(max(len(fed_vals), 1)))
            fasted_means.append(np.nanmean(fas_vals))
            fasted_sems.append(np.nanstd(fas_vals) / np.sqrt(max(len(fas_vals), 1)))
            fed_pts.append(fed_vals)
            fasted_pts.append(fas_vals)

        bars1 = ax.bar(x_pos - width/2, fed_means, width, yerr=fed_sems,
                       label='Fed', color='#2196F3', alpha=0.7, capsize=3)
        bars2 = ax.bar(x_pos + width/2, fasted_means, width, yerr=fasted_sems,
                       label='Fasted', color='#F44336', alpha=0.7, capsize=3)

        for i in range(len(vars_cont)):
            ax.scatter([x_pos[i] - width/2] * len(fed_pts[i]), fed_pts[i],
                       color='#1565C0', s=25, zorder=5, alpha=0.8)
            ax.scatter([x_pos[i] + width/2] * len(fasted_pts[i]), fasted_pts[i],
                       color='#C62828', s=25, zorder=5, alpha=0.8)

            # p-value
            p, d = compare_groups(fed_pts[i], fasted_pts[i])
            if np.isfinite(p):
                sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
                ymax = max(max(fed_pts[i], default=0), max(fasted_pts[i], default=0))
                ax.text(x_pos[i], ymax + 0.05, sig, ha='center', fontsize=9)

        ax.set_xticks(x_pos)
        ax.set_xticklabels([SHORT_NAMES[v] for v in vars_cont], rotation=30, ha='right')
        ax.set_ylabel('R²')
        ax.set_title(f'{region.upper()} — Continuous Variables')
        ax.legend(loc='upper right')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

        # Binary variables
        ax = axes[row_idx, 1]
        vars_bin = BINARY_TARGETS
        x_pos = np.arange(len(vars_bin))

        fed_means, fed_sems, fasted_means, fasted_sems = [], [], [], []
        fed_pts, fasted_pts = [], []

        for var in vars_bin:
            fed_vals = [r['roc_auc'] for r in results
                        if r['variable'] == var and r['session'] in fed_sessions
                        and np.isfinite(r['roc_auc'])]
            fas_vals = [r['roc_auc'] for r in results
                        if r['variable'] == var and r['session'] in fasted_sessions
                        and np.isfinite(r['roc_auc'])]
            fed_means.append(np.nanmean(fed_vals) if fed_vals else np.nan)
            fed_sems.append(np.nanstd(fed_vals) / np.sqrt(max(len(fed_vals), 1))
                            if fed_vals else 0)
            fasted_means.append(np.nanmean(fas_vals) if fas_vals else np.nan)
            fasted_sems.append(np.nanstd(fas_vals) / np.sqrt(max(len(fas_vals), 1))
                               if fas_vals else 0)
            fed_pts.append(fed_vals)
            fasted_pts.append(fas_vals)

        bars1 = ax.bar(x_pos - width/2, fed_means, width, yerr=fed_sems,
                       label='Fed', color='#2196F3', alpha=0.7, capsize=3)
        bars2 = ax.bar(x_pos + width/2, fasted_means, width, yerr=fasted_sems,
                       label='Fasted', color='#F44336', alpha=0.7, capsize=3)

        for i in range(len(vars_bin)):
            ax.scatter([x_pos[i] - width/2] * len(fed_pts[i]), fed_pts[i],
                       color='#1565C0', s=25, zorder=5, alpha=0.8)
            ax.scatter([x_pos[i] + width/2] * len(fasted_pts[i]), fasted_pts[i],
                       color='#C62828', s=25, zorder=5, alpha=0.8)

            p, d = compare_groups(fed_pts[i], fasted_pts[i])
            if np.isfinite(p):
                sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
                ymax = max(max(fed_pts[i], default=0.5), max(fasted_pts[i], default=0.5))
                ax.text(x_pos[i], ymax + 0.02, sig, ha='center', fontsize=8)

        ax.set_xticks(x_pos)
        ax.set_xticklabels([SHORT_NAMES[v] for v in vars_bin], rotation=40, ha='right')
        ax.set_ylabel('ROC-AUC')
        ax.set_title(f'{region.upper()} — Binary Variables')
        ax.legend(loc='upper right')
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Chance')

    plt.suptitle('Behavioral Decoding from GRU-ODE Latent States: Fed vs Fasted',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    path = FIGURES_DIR / 'gru_ode_10ms_decoding_fed_vs_fasted.png'
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_lha_vs_rsp(region_results):
    """Figure 3: LHA vs RSP scatter comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Continuous: R²
    ax = axes[0]
    for var in CONTINUOUS_TARGETS:
        lha_vals = {r['session']: r['r2'] for r in region_results['lha']
                    if r['variable'] == var}
        rsp_vals = {r['session']: r['r2'] for r in region_results['rsp']
                    if r['variable'] == var}
        for sn in range(1, 9):
            if sn in lha_vals and sn in rsp_vals:
                color = '#2196F3' if SESSION_INFO[sn]['state'] == 'Fed' else '#F44336'
                marker = 'o' if SESSION_INFO[sn]['phase'] == 'Exploration' else 's'
                ax.scatter(lha_vals[sn], rsp_vals[sn], c=color, marker=marker,
                           s=60, alpha=0.8, edgecolors='k', linewidths=0.5)

    # Add variable labels at mean positions
    for var in CONTINUOUS_TARGETS:
        lha_mean = np.nanmean([r['r2'] for r in region_results['lha'] if r['variable'] == var])
        rsp_mean = np.nanmean([r['r2'] for r in region_results['rsp'] if r['variable'] == var])
        ax.annotate(SHORT_NAMES[var], (lha_mean, rsp_mean),
                    fontsize=8, ha='center', va='bottom', fontweight='bold')

    lims = ax.get_xlim() + ax.get_ylim()
    lo, hi = min(lims), max(lims)
    ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.3)
    ax.set_xlabel('LHA R²')
    ax.set_ylabel('RSP R²')
    ax.set_title('Continuous Variables')

    # Binary: AUC
    ax = axes[1]
    for var in BINARY_TARGETS:
        lha_vals = {r['session']: r['roc_auc'] for r in region_results['lha']
                    if r['variable'] == var and np.isfinite(r['roc_auc'])}
        rsp_vals = {r['session']: r['roc_auc'] for r in region_results['rsp']
                    if r['variable'] == var and np.isfinite(r['roc_auc'])}
        for sn in range(1, 9):
            if sn in lha_vals and sn in rsp_vals:
                color = '#2196F3' if SESSION_INFO[sn]['state'] == 'Fed' else '#F44336'
                marker = 'o' if SESSION_INFO[sn]['phase'] == 'Exploration' else 's'
                ax.scatter(lha_vals[sn], rsp_vals[sn], c=color, marker=marker,
                           s=60, alpha=0.8, edgecolors='k', linewidths=0.5)

    for var in BINARY_TARGETS:
        lha_mean = np.nanmean([r['roc_auc'] for r in region_results['lha']
                               if r['variable'] == var and np.isfinite(r['roc_auc'])])
        rsp_mean = np.nanmean([r['roc_auc'] for r in region_results['rsp']
                               if r['variable'] == var and np.isfinite(r['roc_auc'])])
        if np.isfinite(lha_mean) and np.isfinite(rsp_mean):
            ax.annotate(SHORT_NAMES[var], (lha_mean, rsp_mean),
                        fontsize=7, ha='center', va='bottom', fontweight='bold')

    lims = ax.get_xlim() + ax.get_ylim()
    lo, hi = min(lims), max(lims)
    ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.3)
    ax.set_xlabel('LHA AUC')
    ax.set_ylabel('RSP AUC')
    ax.set_title('Binary Variables')

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2196F3',
               markersize=8, label='Fed Exp'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#2196F3',
               markersize=8, label='Fed For'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#F44336',
               markersize=8, label='Fasted Exp'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#F44336',
               markersize=8, label='Fasted For'),
    ]
    axes[1].legend(handles=legend_elements, loc='lower right', fontsize=8)

    plt.suptitle('LHA vs RSP Decoding Quality', fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = FIGURES_DIR / 'gru_ode_10ms_decoding_lha_vs_rsp.png'
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_timecourse(region_results, all_hidden, all_behav):
    """Figure 4: Actual vs predicted timecourse for top decoded variables."""
    # Find top 3 continuous variables by mean R² across both regions
    var_r2 = {}
    for var in CONTINUOUS_TARGETS:
        vals = []
        for region in ['lha', 'rsp']:
            vals.extend([r['r2'] for r in region_results[region]
                         if r['variable'] == var and np.isfinite(r['r2'])])
        var_r2[var] = np.nanmean(vals) if vals else -1
    top_vars = sorted(var_r2.keys(), key=lambda v: var_r2[v], reverse=True)[:3]

    fig, axes = plt.subplots(len(top_vars), 2, figsize=(18, 4 * len(top_vars)))
    if len(top_vars) == 1:
        axes = axes.reshape(1, -1)

    # Use session 1 (fed) as example
    example_session = 1

    for row, var in enumerate(top_vars):
        for col, region in enumerate(['lha', 'rsp']):
            ax = axes[row, col]
            key = (region, example_session)
            if key not in all_hidden or example_session not in all_behav:
                ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center')
                continue

            H_100ms = all_hidden[key]
            behav_df = all_behav[example_session]
            n_use = min(len(H_100ms), len(behav_df))
            split_idx = int(n_use * TRAIN_FRAC)

            H_train = H_100ms[:split_idx]
            H_test = H_100ms[split_idx:n_use]
            y = behav_df[var].values[:n_use].astype(np.float64)
            y = np.nan_to_num(y, nan=0.0)
            y_train, y_test = y[:split_idx], y[split_idx:]

            # Fit decoder
            scaler_x = StandardScaler().fit(H_train)
            scaler_y = StandardScaler().fit(y_train.reshape(-1, 1))
            H_tr_s = scaler_x.transform(H_train)
            y_tr_s = scaler_y.transform(y_train.reshape(-1, 1)).ravel()
            ridge = RidgeCV(alphas=np.logspace(-3, 4, 20)).fit(H_tr_s, y_tr_s)

            H_te_s = scaler_x.transform(H_test)
            y_pred_s = ridge.predict(H_te_s)
            y_pred = scaler_y.inverse_transform(y_pred_s.reshape(-1, 1)).ravel()

            # Plot 60s window
            t = np.arange(len(y_test)) * 0.1  # seconds
            plot_len = min(600, len(y_test))  # 60 seconds
            ax.plot(t[:plot_len], y_test[:plot_len], 'k-', alpha=0.6, linewidth=0.8,
                    label='Actual')
            ax.plot(t[:plot_len], y_pred[:plot_len], 'r-', alpha=0.8, linewidth=0.8,
                    label='Predicted')

            r2_val = [r['r2'] for r in region_results[region]
                      if r['variable'] == var and r['session'] == example_session]
            r2_str = f"R²={r2_val[0]:.3f}" if r2_val else ""
            ax.set_title(f'{region.upper()} — {SHORT_NAMES[var]} (S{example_session}) {r2_str}')
            ax.set_xlabel('Time (s)' if row == len(top_vars) - 1 else '')
            ax.legend(loc='upper right', fontsize=8)

    plt.suptitle('Decoding Timecourse: Actual vs Predicted (Test Set, 60s window)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = FIGURES_DIR / 'gru_ode_10ms_decoding_timecourse.png'
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_decoder_weights(region_results):
    """Figure 5: Decoder weight heatmap showing which latent dims matter."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))

    for ax_idx, region in enumerate(['lha', 'rsp']):
        ax = axes[ax_idx]

        # Average weights across sessions for continuous variables
        weight_matrix = np.zeros((32, len(CONTINUOUS_TARGETS)))
        counts = np.zeros(len(CONTINUOUS_TARGETS))

        for r in region_results[region]:
            if r['var_type'] == 'continuous' and r['weights'] is not None:
                col = CONTINUOUS_TARGETS.index(r['variable'])
                weight_matrix[:, col] += np.abs(r['weights'])
                counts[col] += 1

        for j in range(len(CONTINUOUS_TARGETS)):
            if counts[j] > 0:
                weight_matrix[:, j] /= counts[j]

        # Normalize per variable for visualization
        for j in range(weight_matrix.shape[1]):
            wmax = weight_matrix[:, j].max()
            if wmax > 0:
                weight_matrix[:, j] /= wmax

        sns.heatmap(weight_matrix, ax=ax, cmap='viridis',
                    xticklabels=[SHORT_NAMES[v] for v in CONTINUOUS_TARGETS],
                    yticklabels=range(32),
                    cbar_kws={'shrink': 0.8, 'label': 'Normalized |weight|'})
        ax.set_title(f'{region.upper()} — Ridge Decoder Weights', fontweight='bold')
        ax.set_xlabel('Behavioral Variable')
        ax.set_ylabel('Latent Dimension')

    plt.tight_layout()
    path = FIGURES_DIR / 'gru_ode_10ms_decoding_weights.png'
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print(f"Device: {DEVICE}")

    # Load spike data
    sessions_data = load_all_sessions()

    # Load behavior for all sessions
    print("\nLoading behavior data...")
    all_behav = {}
    for sn in sorted(sessions_data.keys()):
        behav_df = load_behavior(sn)
        all_behav[sn] = behav_df
        print(f"  Session {sn}: {len(behav_df)} bins, "
              f"{len(behav_df.columns)} variables")

    # Run decoding for each region
    all_results = []
    region_results = {}
    all_hidden = {}  # (region, session) -> H_100ms

    for region in ['lha', 'rsp']:
        print(f"\n{'='*60}")
        print(f"  {region.upper()}")
        print(f"{'='*60}")

        # Load model
        model = load_model(region)

        # Extract hidden states for all sessions
        print(f"\n  Extracting hidden states...")
        for sn in sorted(sessions_data.keys()):
            h_10ms = get_hidden_states(model, sessions_data, region, sn, DEVICE)
            h_100ms = h_10ms[::SUBSAMPLE_RATIO]
            all_hidden[(region, sn)] = h_100ms
            print(f"    S{sn}: {h_10ms.shape[0]} -> {h_100ms.shape[0]} (100ms bins)")

        # Decode each session
        print(f"\n  Decoding behavioral variables...")
        region_res = []
        for sn in sorted(sessions_data.keys()):
            print(f"    Session {sn} ({SESSION_INFO[sn]['state']} "
                  f"{SESSION_INFO[sn]['phase']})...")
            H_100ms = all_hidden[(region, sn)]
            behav_df = all_behav[sn]
            results = decode_session(H_100ms, behav_df, sn)
            for r in results:
                r['region'] = region
            region_res.extend(results)

            # Print summary
            for r in results:
                if r['var_type'] == 'continuous':
                    print(f"      {SHORT_NAMES[r['variable']]:20s}: "
                          f"R²={r['r2']:.3f}, r={r['pearson_r']:.3f} "
                          f"(shuffle={r['shuffle_baseline']:.3f})")
                elif r['status'] == 'ok':
                    print(f"      {SHORT_NAMES[r['variable']]:20s}: "
                          f"AUC={r['roc_auc']:.3f}, F1={r['f1']:.3f} "
                          f"(shuffle={r['shuffle_baseline']:.3f})")
                else:
                    print(f"      {SHORT_NAMES[r['variable']]:20s}: "
                          f"skipped ({r['status']})")

        region_results[region] = region_res
        all_results.extend(region_res)

    # --- Statistical Comparisons ---
    print(f"\n{'='*60}")
    print(f"  Statistical Comparisons")
    print(f"{'='*60}")

    summary_rows = []
    fed_sessions = [1, 2, 3, 4]
    fasted_sessions = [5, 6, 7, 8]

    for region in ['lha', 'rsp']:
        print(f"\n  {region.upper()}:")
        for var in ALL_TARGETS:
            vtype = 'continuous' if var in CONTINUOUS_TARGETS else 'binary'
            metric_key = 'r2' if vtype == 'continuous' else 'roc_auc'

            fed_vals = [r[metric_key] for r in region_results[region]
                        if r['variable'] == var and r['session'] in fed_sessions
                        and np.isfinite(r[metric_key])]
            fas_vals = [r[metric_key] for r in region_results[region]
                        if r['variable'] == var and r['session'] in fasted_sessions
                        and np.isfinite(r[metric_key])]

            p_state, d_state = compare_groups(fed_vals, fas_vals)
            sig = '***' if p_state < 0.001 else '**' if p_state < 0.01 else \
                  '*' if p_state < 0.05 else 'ns' if np.isfinite(p_state) else 'n/a'

            metric_name = 'R²' if vtype == 'continuous' else 'AUC'
            print(f"    {SHORT_NAMES[var]:20s}: Fed={np.nanmean(fed_vals):.3f}, "
                  f"Fasted={np.nanmean(fas_vals):.3f}, p={p_state:.4f} {sig}")

            # LHA vs RSP comparison
            lha_vals = [r[metric_key] for r in region_results['lha']
                        if r['variable'] == var and np.isfinite(r[metric_key])]
            rsp_vals_all = [r[metric_key] for r in region_results['rsp']
                            if r['variable'] == var and np.isfinite(r[metric_key])]

            summary_rows.append({
                'region': region, 'variable': var, 'var_type': vtype,
                'fed_mean': np.nanmean(fed_vals) if fed_vals else np.nan,
                'fed_sem': np.nanstd(fed_vals) / np.sqrt(max(len(fed_vals), 1)) if fed_vals else np.nan,
                'fasted_mean': np.nanmean(fas_vals) if fas_vals else np.nan,
                'fasted_sem': np.nanstd(fas_vals) / np.sqrt(max(len(fas_vals), 1)) if fas_vals else np.nan,
                'fed_vs_fasted_p': p_state, 'fed_vs_fasted_d': d_state,
            })

    # LHA vs RSP paired comparison
    print(f"\n  LHA vs RSP (Wilcoxon signed-rank):")
    for var in ALL_TARGETS:
        vtype = 'continuous' if var in CONTINUOUS_TARGETS else 'binary'
        metric_key = 'r2' if vtype == 'continuous' else 'roc_auc'

        lha_by_session = {r['session']: r[metric_key] for r in region_results['lha']
                          if r['variable'] == var}
        rsp_by_session = {r['session']: r[metric_key] for r in region_results['rsp']
                          if r['variable'] == var}

        lha_v = [lha_by_session.get(s, np.nan) for s in range(1, 9)]
        rsp_v = [rsp_by_session.get(s, np.nan) for s in range(1, 9)]

        p_region, d_region = compare_paired(lha_v, rsp_v)
        sig = '***' if p_region < 0.001 else '**' if p_region < 0.01 else \
              '*' if p_region < 0.05 else 'ns' if np.isfinite(p_region) else 'n/a'
        print(f"    {SHORT_NAMES[var]:20s}: LHA={np.nanmean(lha_v):.3f}, "
              f"RSP={np.nanmean(rsp_v):.3f}, p={p_region:.4f} {sig}")

    # --- Save CSVs ---
    print(f"\n{'='*60}")
    print(f"  Saving results...")
    print(f"{'='*60}")

    # Per-session results (exclude weights column)
    results_df = pd.DataFrame([{k: v for k, v in r.items() if k != 'weights'}
                                for r in all_results])
    results_path = DATA_DIR / 'gru_ode_10ms_decoding_results.csv'
    results_df.to_csv(results_path, index=False, float_format='%.4f')
    print(f"  Saved: {results_path}")

    # Summary
    summary_df = pd.DataFrame(summary_rows)
    summary_path = DATA_DIR / 'gru_ode_10ms_decoding_summary.csv'
    summary_df.to_csv(summary_path, index=False, float_format='%.4f')
    print(f"  Saved: {summary_path}")

    # --- Generate Figures ---
    print(f"\n  Generating figures...")
    plot_heatmap(all_results, region_results)
    plot_fed_vs_fasted(all_results, region_results)
    plot_lha_vs_rsp(region_results)
    plot_timecourse(region_results, all_hidden, all_behav)
    plot_decoder_weights(region_results)

    print(f"\nDone!")


if __name__ == '__main__':
    main()
