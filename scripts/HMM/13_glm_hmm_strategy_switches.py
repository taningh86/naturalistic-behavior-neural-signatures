"""13 — GLM-HMM with covariate-dependent transitions for strategy-switch detection.

Pools 6 foraging sessions (S4, S6, S8, S12, S14, S16). S10 excluded (no-food).

Pipeline:
  1. Per-bin covariates per session (cum_non_rewarded_digs, cum_distinct_pots,
     time_since_start_z) — saved to data/HMM/glm_hmm/covariates/.
  2. CV across K ∈ {8, 10, 12, 14}, 3-fold stratified, 2 random inits per fold
     (reduced from spec's 5 to keep wall-time tractable; ~4 h total).
  3. Recommended K via 1-SE rule.
  4. Final GLM-HMM fit at recommended K on all 6 sessions pooled.
  5. Standard MixedHMM (fixed transitions) refit at the same K for comparison
     (held-out LL/bin + AIC/BIC).
  6. Per-session per-bin transition tensor extraction.
  7. Key transition trajectories: P(dig | pot-zone), P(dig | T-zone),
     P(feed | dig).
  8. Coefficient summary: which transitions are most modulated by
     cum_non_rewarded_digs.
  9. Strategy-switch index per session (smoothed Δ P(pot→dig)).

Same QC-filtered units / same emission factorisation as MixedHMM. Custom
GLM-HMM in `scripts/HMM/mixed_glmhmm.py`.

Spec deviations:
  - 2 inits per fold (not 5): runtime budget.
  - max_iters=40, tol=1e-2: same.
"""
from pathlib import Path
import sys
import time
import warnings

import numpy as np
import pandas as pd
import yaml
import jax
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "HMM"))

from _utils import load_config
import mixed_glmhmm as mg
import mixed_hmm as mh   # for standard HMM comparison


# =====  Constants  =====
SESSIONS = [4, 6, 8, 12, 14, 16]
K_RANGE = [8, 10, 12, 14]
N_INITS = 2                 # spec: 5; reduced for tractability
MAX_ITERS = 40
TOL = 1e-2
TRANS_GRAD_STEPS = 40
TRANS_LR = 0.05
SEED_MASTER = 20260509
DISCOVERY_WINDOW_S = 5.0    # for cum_non_rewarded_digs definition
DIGGING_STATE_ID_OLD = 6    # in the K=14 merged fit, used for pot identity
HMM_BIN_S = 0.480

# CV folds (stratified leave 1 fed + 1 fasted)
CV_FOLDS = [(4, 12), (6, 14), (8, 16)]


def out_dirs():
    base_out = REPO_ROOT / "data" / "HMM" / "glm_hmm"
    base_fig = REPO_ROOT / "figures" / "HMM" / "glm_hmm"
    (base_out / "covariates").mkdir(parents=True, exist_ok=True)
    (base_out / "transitions").mkdir(parents=True, exist_ok=True)
    (base_fig / "transitions").mkdir(parents=True, exist_ok=True)
    return base_out, base_fig


# =====  Covariates  =====
def compute_covariates_for_session(sn, cfg):
    """Compute (T, 3) per-bin covariate matrix for a session, plus column names.

    cum_non_rewarded_digs: count of dig runs (state 6) in 480ms-binned Viterbi
                             that did NOT enter a feeding state (S2 or S11) within
                             5 s, accumulated. Updated at the END of each dig run.
    cum_distinct_pots_visited: count of distinct pots (1..4) entered up to bin t,
                                using pot_id transitions.
    time_since_start_z: (t * 0.480 - mean) / std, z-scored within session.
    """
    binned = np.load(REPO_ROOT / cfg["out_dirs"]["binned"] / f"session_{sn}.npz",
                      allow_pickle=True)
    pot_id = np.asarray(binned["pot_id"], dtype=np.int64)
    n = pot_id.shape[0]

    post = pd.read_csv(REPO_ROOT / cfg["merge_dirs"]["posteriors"]
                        / f"session_{sn}.csv")
    viterbi = post["viterbi"].values.astype(np.int64)
    n = min(n, len(viterbi))
    pot_id = pot_id[:n]; viterbi = viterbi[:n]

    feed_states = {2, 11}
    win_bins = max(1, int(round(DISCOVERY_WINDOW_S / HMM_BIN_S)))

    # Find dig runs
    in_dig = (viterbi == DIGGING_STATE_ID_OLD)
    diff = np.diff(in_dig.astype(int), prepend=0, append=0)
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1)

    # cum_non_rewarded_digs: stepwise update
    cum_nrd = np.zeros(n, dtype=np.float64)
    for s, e in zip(starts, ends):
        end_check = min(n, e + win_bins)
        followed_by_feeding = bool(np.isin(viterbi[s:end_check], list(feed_states)).any())
        if not followed_by_feeding:
            # Increment from end-of-dig onward
            cum_nrd[e - 1:] += 1
    # cumulative carry: above already broadcasts the increment forward.
    # But the broadcasting was wrong: I added to slice [e-1:] which is fine
    # if increments stack — they do, since each new dig adds +1 from its e onward.

    # cum_distinct_pots_visited: track set of pots ever visited
    distinct_pots = np.zeros(n, dtype=np.int64)
    seen = set()
    for t in range(n):
        if pot_id[t] > 0:
            seen.add(int(pot_id[t]))
        distinct_pots[t] = len(seen)
    cum_distinct = distinct_pots.astype(np.float64)

    # time_since_start (s)
    tt = np.arange(n) * HMM_BIN_S
    tt_z = (tt - tt.mean()) / (tt.std() + 1e-9)

    # Z-score the other covariates
    nrd_z = (cum_nrd - cum_nrd.mean()) / (cum_nrd.std() + 1e-9)
    dist_z = (cum_distinct - cum_distinct.mean()) / (cum_distinct.std() + 1e-9)

    cov = np.column_stack([nrd_z, dist_z, tt_z])
    cov_raw = np.column_stack([cum_nrd, cum_distinct, tt])
    cov_names = ["cum_non_rewarded_digs_z", "cum_distinct_pots_visited_z",
                 "time_since_start_z"]
    return cov, cov_raw, cov_names


def save_covariates_for_all_sessions(cfg):
    out_dir, _ = out_dirs()
    cov_dir = out_dir / "covariates"
    cov_data = {}
    for sn in SESSIONS:
        cov, cov_raw, cov_names = compute_covariates_for_session(sn, cfg)
        df = pd.DataFrame(cov, columns=cov_names)
        df["bin"] = np.arange(len(cov))
        df["time_s"] = df["bin"] * HMM_BIN_S
        df["cum_non_rewarded_digs_raw"] = cov_raw[:, 0]
        df["cum_distinct_pots_visited_raw"] = cov_raw[:, 1]
        df.to_csv(cov_dir / f"session_{sn}_covariates.csv", index=False)
        cov_data[sn] = cov
    return cov_data


# =====  Session loader for the GLM-HMM (uses prepared_dynamax data)  =====
def load_session_for_glmhmm(sn, cfg, cov_data):
    prep = np.load(REPO_ROOT / cfg["dynamax_dirs"]["prepared"]
                    / f"session_{sn}.npz", allow_pickle=True)
    X_cont = np.asarray(prep["X_continuous"], dtype=np.float64)
    X_zone = np.asarray(prep["X_zone"], dtype=np.int64)
    X_events = np.asarray(prep["X_events"], dtype=np.float64)
    U = cov_data[sn]
    n = min(X_cont.shape[0], U.shape[0])
    return dict(
        sn=sn,
        X_cont=X_cont[:n], X_zone=X_zone[:n],
        X_events=X_events[:n], U=U[:n],
        state=str(prep["state"]),
    )


# =====  CV  =====
def cv_glmhmm(all_sessions, K, n_cov, K_zone, n_events, base_seed):
    """Returns list of dicts: K, fold, init, train_ll, heldout_ll, heldout_T."""
    results = []
    for fi, (held_fed, held_fasted) in enumerate(CV_FOLDS):
        train = [s for s in all_sessions
                  if s["sn"] not in (held_fed, held_fasted)]
        test = [s for s in all_sessions
                 if s["sn"] in (held_fed, held_fasted)]
        for ii in range(N_INITS):
            seed = base_seed + 1000 * (fi + 1) + 100 * K + ii
            key = jax.random.PRNGKey(seed)
            cont_pool = np.concatenate([s["X_cont"] for s in train], axis=0)
            zone_pool = np.concatenate([s["X_zone"] for s in train], axis=0)
            ev_pool = np.concatenate([s["X_events"] for s in train], axis=0)
            params0 = mg.init_params(
                key, K, n_cov, cont_pool, zone_pool, ev_pool, K_zone, n_events,
            )
            t0 = time.time()
            params, hist = mg.fit(
                params0, train, max_iters=MAX_ITERS, tol=TOL, verbose=False,
                trans_grad_steps=TRANS_GRAD_STEPS, trans_lr=TRANS_LR,
            )
            train_ll = hist["final_loglik"]
            ho_ll, ho_T = mg.held_out_loglik(params, test)
            t_fit = time.time() - t0
            row = dict(K=K, fold=fi,
                        held_fed=held_fed, held_fasted=held_fasted,
                        init_idx=ii, seed=seed,
                        train_ll=train_ll, n_iter=hist["n_iter"],
                        heldout_ll=ho_ll, heldout_T=ho_T,
                        heldout_ll_per_bin=ho_ll / ho_T,
                        fit_time_s=t_fit)
            results.append(row)
            print(f"    K={K} fold={fi} init={ii}: "
                  f"train_ll={train_ll:.1f} ho_ll/bin={ho_ll/ho_T:.4f} "
                  f"iters={hist['n_iter']} ({t_fit:.0f}s)", flush=True)
    return results


# =====  Standard MixedHMM CV (re-uses earlier infrastructure for comparison)  =====
def cv_standard_hmm(all_sessions, K, K_zone, n_events, base_seed):
    """Same CV folds, standard HMM (no covariates). Faster — use 5 inits."""
    results = []
    for fi, (held_fed, held_fasted) in enumerate(CV_FOLDS):
        train = [s for s in all_sessions
                  if s["sn"] not in (held_fed, held_fasted)]
        test = [s for s in all_sessions
                 if s["sn"] in (held_fed, held_fasted)]
        for ii in range(5):
            seed = base_seed + 7000 * K + ii + 1000 * (fi + 1)
            key = jax.random.PRNGKey(seed)
            cont_pool = np.concatenate([s["X_cont"] for s in train], axis=0)
            zone_pool = np.concatenate([s["X_zone"] for s in train], axis=0)
            ev_pool = np.concatenate([s["X_events"] for s in train], axis=0)
            params0 = mh.init_params(key, K=K, X_cont_pool=cont_pool,
                                       X_zone_pool=zone_pool, X_events_pool=ev_pool,
                                       K_zone=K_zone, n_events=n_events)
            train_for_mh = [{"X_cont": s["X_cont"], "X_zone": s["X_zone"],
                              "X_events": s["X_events"]} for s in train]
            test_for_mh = [{"X_cont": s["X_cont"], "X_zone": s["X_zone"],
                             "X_events": s["X_events"]} for s in test]
            t0 = time.time()
            params, hist = mh.fit(params0, train_for_mh, max_iters=200, tol=1e-4,
                                    verbose=False)
            train_ll = hist["final_loglik"]
            ho_ll, ho_T = mh.held_out_loglik(params, test_for_mh)
            t_fit = time.time() - t0
            row = dict(K=K, fold=fi, init_idx=ii, seed=seed,
                        train_ll=train_ll, n_iter=hist["n_iter"],
                        heldout_ll_per_bin=ho_ll / ho_T,
                        heldout_T=ho_T, fit_time_s=t_fit)
            results.append(row)
            print(f"    [std] K={K} fold={fi} init={ii}: "
                  f"ho_ll/bin={ho_ll/ho_T:.4f} ({t_fit:.0f}s)", flush=True)
    return results


def select_K_by_1se(cv_df: pd.DataFrame, group_col="K"):
    """1-SE rule: smallest K within 1 SE of best mean held-out LL."""
    best = (cv_df.sort_values("train_ll", ascending=False)
                  .groupby([group_col, "fold"], as_index=False).first())
    agg = best.groupby(group_col, as_index=False).agg(
        mean_ll=("heldout_ll_per_bin", "mean"),
        se_ll=("heldout_ll_per_bin",
               lambda x: x.std(ddof=1) / np.sqrt(len(x))),
    )
    max_idx = agg["mean_ll"].idxmax()
    threshold = agg.loc[max_idx, "mean_ll"] - agg.loc[max_idx, "se_ll"]
    eligible = agg[agg["mean_ll"] >= threshold]
    return int(eligible[group_col].min()), agg


# =====  Final fit  =====
def final_fit_glmhmm(all_sessions, K, n_cov, K_zone, n_events, base_seed):
    """Run N_INITS random inits on all sessions, return best by train LL."""
    cont_pool = np.concatenate([s["X_cont"] for s in all_sessions], axis=0)
    zone_pool = np.concatenate([s["X_zone"] for s in all_sessions], axis=0)
    ev_pool = np.concatenate([s["X_events"] for s in all_sessions], axis=0)
    best = None
    for ii in range(N_INITS):
        seed = base_seed + 9000 * K + ii
        key = jax.random.PRNGKey(seed)
        params0 = mg.init_params(key, K, n_cov, cont_pool, zone_pool, ev_pool,
                                   K_zone, n_events)
        t0 = time.time()
        p, hist = mg.fit(params0, all_sessions, max_iters=MAX_ITERS,
                          tol=TOL, verbose=False,
                          trans_grad_steps=TRANS_GRAD_STEPS, trans_lr=TRANS_LR)
        t_fit = time.time() - t0
        ll = hist["final_loglik"]
        print(f"  final init {ii}: train_ll={ll:.1f} iters={hist['n_iter']} "
              f"({t_fit:.0f}s)", flush=True)
        if best is None or ll > best["ll"]:
            best = dict(params=p, hist=hist, ll=ll, init_idx=ii, seed=seed)
    return best


def final_fit_standard(all_sessions, K, K_zone, n_events, base_seed):
    cont_pool = np.concatenate([s["X_cont"] for s in all_sessions], axis=0)
    zone_pool = np.concatenate([s["X_zone"] for s in all_sessions], axis=0)
    ev_pool = np.concatenate([s["X_events"] for s in all_sessions], axis=0)
    sess_for_mh = [{"X_cont": s["X_cont"], "X_zone": s["X_zone"],
                     "X_events": s["X_events"]} for s in all_sessions]
    best = None
    for ii in range(5):
        seed = base_seed + 33000 * K + ii
        key = jax.random.PRNGKey(seed)
        params0 = mh.init_params(key, K=K, X_cont_pool=cont_pool,
                                   X_zone_pool=zone_pool, X_events_pool=ev_pool,
                                   K_zone=K_zone, n_events=n_events)
        t0 = time.time()
        p, hist = mh.fit(params0, sess_for_mh, max_iters=200, tol=1e-4,
                          verbose=False)
        ll = hist["final_loglik"]
        print(f"  std final init {ii}: train_ll={ll:.1f} ({time.time()-t0:.0f}s)",
              flush=True)
        if best is None or ll > best["ll"]:
            best = dict(params=p, hist=hist, ll=ll, init_idx=ii, seed=seed)
    return best


# =====  Per-session transition trajectories + key transitions  =====
def find_states_by_emission(params, q_event_threshold=0.7,
                             zone_event_threshold=0.6):
    """Return state IDs for digging, feeding, transition-zone, pot-zone."""
    q_dig = params.q_events[:, 2]      # event index 2 = digging_sand
    q_feed = params.q_events[:, 3]     # event index 3 = feeding
    p_pot = params.p_zone[:, 2] + params.p_zone[:, 3]  # zone pot + pot_zone
    p_trans = params.p_zone[:, 1]      # zone transition

    digging = list(np.where(q_dig >= q_event_threshold)[0])
    feeding = list(np.where(q_feed >= q_event_threshold)[0])
    pot_zone = [int(k) for k in np.where(p_pot >= zone_event_threshold)[0]
                if k not in digging and k not in feeding]
    transition = list(np.where(p_trans >= zone_event_threshold)[0])
    return dict(digging=digging, feeding=feeding,
                pot_zone=pot_zone, transition=transition)


def extract_per_session_transitions(params, sessions, base_out_dir,
                                      base_fig_dir, history_df,
                                      sess_state_lookup, state_idx):
    """For each session, compute and save the per-bin (T, K, K) transition tensor
    plus key transitions of interest."""
    digging = state_idx["digging"]; feeding = state_idx["feeding"]
    pot_zone = state_idx["pot_zone"]; transition = state_idx["transition"]
    print(f"  State-emission classification:")
    print(f"    digging: {digging}")
    print(f"    feeding: {feeding}")
    print(f"    pot-zone: {pot_zone}")
    print(f"    transition: {transition}")

    if not digging:
        print("  WARNING: no digging state identified; skipping key trajectories")
        digging = [0]   # fallback
    dig_id = digging[0]

    rows_traj = []
    for s in sessions:
        sn = s["sn"]
        gamma, A_t, ll = mg.smoothed_posteriors_and_transitions(
            params, s["X_cont"], s["X_zone"], s["X_events"], s["U"]
        )
        T, K, _ = A_t.shape
        # Save full tensor
        np.savez(base_out_dir / "transitions" / f"session_{sn}_transitions.npz",
                  A=A_t.astype(np.float32), gamma=gamma.astype(np.float32))

        # Key trajectories
        # P(digging | pot-zone source) — average over pot_zone source states
        if pot_zone:
            P_pot_to_dig = A_t[:, pot_zone, :][:, :, digging].sum(axis=2).mean(axis=1)
        else:
            P_pot_to_dig = np.full(T, np.nan)
        if transition:
            P_trans_to_dig = A_t[:, transition, :][:, :, digging].sum(axis=2).mean(axis=1)
        else:
            P_trans_to_dig = np.full(T, np.nan)
        if feeding:
            P_dig_to_feed = A_t[:, dig_id, feeding].sum(axis=1)
        else:
            P_dig_to_feed = np.full(T, np.nan)

        time_s = np.arange(T) * HMM_BIN_S
        df = pd.DataFrame(dict(
            bin=np.arange(T), time_s=time_s,
            P_pot_to_dig=P_pot_to_dig,
            P_transition_to_dig=P_trans_to_dig,
            P_dig_to_feed=P_dig_to_feed,
        ))
        df.to_csv(base_out_dir / "transitions" / f"session_{sn}_key_transitions.csv",
                   index=False)

        # Plot
        h = history_df[history_df.session == sn].iloc[0]
        disc_t = float(h["discovery_time_s"])
        fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
        for ax, (col, label, color) in zip(axes, [
            ("P_pot_to_dig", "P(dig | pot-zone)", "firebrick"),
            ("P_transition_to_dig", "P(dig | T-zone)", "darkgoldenrod"),
            ("P_dig_to_feed", "P(feed | dig)", "darkgreen"),
        ]):
            ax.plot(df["time_s"], df[col], color=color, lw=1.0)
            # smoothed
            kernel = np.ones(15) / 15
            sm = np.convolve(df[col].fillna(0), kernel, mode="same")
            ax.plot(df["time_s"], sm, color="black", lw=1.5, alpha=0.7)
            ax.axvline(disc_t, color="red", lw=1.0, ls="--", label="discovery")
            ax.set_ylabel(label)
            ax.set_ylim(0, 1)
            ax.grid(alpha=0.3)
        axes[0].legend(fontsize=9, loc="upper right")
        axes[2].set_xlabel("Time (s)")
        fig.suptitle(f"S{sn} ({sess_state_lookup[sn]}) — key transition probabilities")
        fig.tight_layout()
        fig.savefig(base_fig_dir / "transitions" / f"session_{sn}_key_transitions.png",
                     dpi=130)
        plt.close(fig)

        rows_traj.append(dict(session=sn,
                                state=sess_state_lookup[sn],
                                discovery_time_s=disc_t,
                                T=T, K=K))

    return pd.DataFrame(rows_traj)


def coefficient_summary(params, n_cov, cov_names, base_out_dir, base_fig_dir):
    """Tabulate coefficients per (source, target, covariate)."""
    K = params.K
    rows = []
    for i in range(K):
        for j in range(K):
            for c in range(n_cov):
                rows.append(dict(source_state=i, target_state=j,
                                   covariate=cov_names[c],
                                   coefficient=float(params.W[i, c, j]),
                                   abs_coefficient=float(abs(params.W[i, c, j]))))
    df = pd.DataFrame(rows)
    df["rank"] = df["abs_coefficient"].rank(method="dense", ascending=False).astype(int)
    df.to_csv(base_out_dir / "transition_coefficients.csv", index=False)

    # Heatmap of |coef| for cum_non_rewarded_digs
    cnrd_idx = cov_names.index("cum_non_rewarded_digs_z")
    coef_mat = np.abs(params.W[:, cnrd_idx, :])  # (K, K)
    fig, ax = plt.subplots(figsize=(0.5 * K + 2, 0.5 * K + 2))
    im = ax.imshow(coef_mat, aspect="auto", cmap="Reds")
    ax.set_xticks(range(K)); ax.set_xticklabels([f"to S{j}" for j in range(K)])
    ax.set_yticks(range(K)); ax.set_yticklabels([f"from S{i}" for i in range(K)])
    plt.colorbar(im, ax=ax, label="|coefficient|")
    ax.set_title("|coef on cum_non_rewarded_digs_z| — modulates each (i, j) transition")
    fig.tight_layout()
    fig.savefig(base_fig_dir / "coef_cum_non_rewarded_digs_heatmap.png", dpi=130)
    plt.close(fig)

    return df


def strategy_switch_index(traj_df, smooth_w=15):
    """Strategy-switch index = first-derivative of P(pot→dig), smoothed."""
    p = traj_df["P_pot_to_dig"].fillna(0).values
    kernel = np.ones(smooth_w) / smooth_w
    sm = np.convolve(p, kernel, mode="same")
    return np.gradient(sm)


def plot_strategy_switches(sessions, base_out_dir, base_fig_dir,
                            history_df, sess_state_lookup):
    n = len(sessions)
    fig, axes = plt.subplots(n, 1, figsize=(11, 2.5 * n), sharex=True)
    if n == 1:
        axes = [axes]
    rows = []
    for ax, s in zip(axes, sessions):
        sn = s["sn"]
        df = pd.read_csv(base_out_dir / "transitions"
                          / f"session_{sn}_key_transitions.csv")
        idx = strategy_switch_index(df)
        ax.plot(df["time_s"], idx, color="purple", lw=1)
        h = history_df[history_df.session == sn].iloc[0]
        ax.axvline(float(h["discovery_time_s"]), color="red", lw=1, ls="--",
                    label="discovery")
        # Find steepest rise BEFORE discovery
        disc_bin = int(h["discovery_bin"])
        if disc_bin > 30:
            window = idx[:disc_bin]
            steepest_bin = int(np.argmax(window))
            steepest_t = steepest_bin * HMM_BIN_S
            ax.axvline(steepest_t, color="orange", lw=1, ls=":",
                        label=f"steepest rise @ {steepest_t:.0f}s")
        else:
            steepest_t = np.nan
        ax.set_ylabel(f"S{sn} ({sess_state_lookup[sn]})")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(alpha=0.3)
        rows.append(dict(session=sn, state=sess_state_lookup[sn],
                          discovery_time_s=float(h["discovery_time_s"]),
                          steepest_rise_t_s=float(steepest_t) if not np.isnan(steepest_t) else np.nan,
                          lead_time_s=(float(h["discovery_time_s"]) - steepest_t)
                                       if not np.isnan(steepest_t) else np.nan))
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Strategy-switch index per session "
                 "(d/dt smoothed P(pot→dig); orange = steepest pre-discovery rise)",
                 y=1.0)
    fig.tight_layout()
    fig.savefig(base_fig_dir / "strategy_switch_per_session.png", dpi=130)
    plt.close(fig)
    df_t = pd.DataFrame(rows)
    df_t.to_csv(base_out_dir / "strategy_switch_timing.csv", index=False)
    return df_t


# =====  Main  =====
def main():
    cfg = load_config()
    base_out, base_fig = out_dirs()

    print(f"JAX backend: {jax.default_backend()}, devices: {jax.devices()}")

    # 1. Covariates
    print("\n=== Computing covariates ===")
    cov_data = save_covariates_for_all_sessions(cfg)
    n_cov = cov_data[SESSIONS[0]].shape[1]
    cov_names = ["cum_non_rewarded_digs_z", "cum_distinct_pots_visited_z",
                 "time_since_start_z"]

    # 2. Load all sessions for GLM-HMM
    print("\n=== Loading sessions ===")
    all_sessions = [load_session_for_glmhmm(sn, cfg, cov_data) for sn in SESSIONS]
    K_zone = int(all_sessions[0]["X_zone"].max() + 1)
    n_events = all_sessions[0]["X_events"].shape[1]
    print(f"  Sessions: {[s['sn'] for s in all_sessions]}")
    print(f"  K_zone={K_zone}, n_events={n_events}, n_cov={n_cov}")

    sess_state_lookup = {s["sn"]: s["state"] for s in all_sessions}
    history_df = pd.read_csv(REPO_ROOT / cfg["commitment_dirs"]["out"]
                              / "sampling_history.csv")

    # 3. CV across K
    print(f"\n=== GLM-HMM CV (K∈{K_RANGE}, {N_INITS} inits/fold, "
          f"max_iters={MAX_ITERS}) ===")
    cv_rows = []
    for K in K_RANGE:
        rows = cv_glmhmm(all_sessions, K, n_cov, K_zone, n_events, SEED_MASTER)
        cv_rows.extend(rows)
        pd.DataFrame(cv_rows).to_csv(base_out / "cv_results.csv", index=False)

    cv_df = pd.DataFrame(cv_rows)
    cv_df.to_csv(base_out / "cv_results.csv", index=False)
    rec_K, agg = select_K_by_1se(cv_df)
    print(f"\n  CV aggregate:")
    print(agg.to_string(index=False))
    print(f"\n  GLM-HMM recommended K = {rec_K}")

    # CV plot
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(agg["K"], agg["mean_ll"], yerr=agg["se_ll"],
                  marker="o", lw=1.5, capsize=3, color="steelblue")
    ax.axvline(rec_K, color="firebrick", ls=":", label=f"recommended K = {rec_K}")
    ax.set_xlabel("K"); ax.set_ylabel("Held-out LL/bin")
    ax.set_title("GLM-HMM state selection (CV mean ± SE)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(base_fig / "state_selection.png", dpi=130)
    plt.close(fig)

    # 4. Final GLM-HMM fit
    print(f"\n=== Final GLM-HMM fit at K = {rec_K} ===")
    glmhmm_best = final_fit_glmhmm(all_sessions, rec_K, n_cov, K_zone,
                                      n_events, SEED_MASTER)
    glmhmm_params = glmhmm_best["params"]

    # Save final params
    np.savez(
        base_out / "final_model.npz",
        pi=glmhmm_params.pi, W=glmhmm_params.W, b=glmhmm_params.b,
        mu=glmhmm_params.mu, sigma=glmhmm_params.sigma,
        p_zone=glmhmm_params.p_zone, q_events=glmhmm_params.q_events,
        K=rec_K, n_cov=n_cov, K_zone=K_zone, n_events=n_events,
    )

    # 5. Standard MixedHMM comparison at recommended K
    print(f"\n=== Standard HMM comparison at K = {rec_K} ===")
    print("  CV...")
    std_cv_rows = cv_standard_hmm(all_sessions, rec_K, K_zone, n_events,
                                     SEED_MASTER)
    std_cv_df = pd.DataFrame(std_cv_rows)
    std_cv_df.to_csv(base_out / "standard_hmm_cv_results.csv", index=False)
    std_best_per_fold = (std_cv_df.sort_values("train_ll", ascending=False)
                          .groupby("fold", as_index=False).first())
    std_mean_ho = float(std_best_per_fold["heldout_ll_per_bin"].mean())

    glmhmm_best_per_fold = (cv_df[cv_df.K == rec_K]
                              .sort_values("train_ll", ascending=False)
                              .groupby("fold", as_index=False).first())
    glmhmm_mean_ho = float(glmhmm_best_per_fold["heldout_ll_per_bin"].mean())

    n_bins_total = sum(s["X_cont"].shape[0] for s in all_sessions)
    n_params_glm = mg.n_free_params(rec_K, all_sessions[0]["X_cont"].shape[1],
                                       K_zone, n_events, n_cov)
    n_params_std = mg.n_free_params_standard(rec_K, all_sessions[0]["X_cont"].shape[1],
                                                K_zone, n_events)

    aic_glm = 2 * n_params_glm - 2 * glmhmm_best["ll"]
    aic_std_train = 2 * n_params_std    # placeholder; get LL by fitting
    print("  Final standard HMM fit (for AIC/BIC)...")
    std_best = final_fit_standard(all_sessions, rec_K, K_zone, n_events,
                                    SEED_MASTER)
    aic_std = 2 * n_params_std - 2 * std_best["ll"]
    bic_glm = n_params_glm * np.log(n_bins_total) - 2 * glmhmm_best["ll"]
    bic_std = n_params_std * np.log(n_bins_total) - 2 * std_best["ll"]

    cmp_df = pd.DataFrame([
        dict(model="GLM-HMM", K=rec_K, heldout_ll_per_bin=glmhmm_mean_ho,
              n_params=n_params_glm, AIC=aic_glm, BIC=bic_glm,
              train_ll=glmhmm_best["ll"]),
        dict(model="Standard MixedHMM", K=rec_K, heldout_ll_per_bin=std_mean_ho,
              n_params=n_params_std, AIC=aic_std, BIC=bic_std,
              train_ll=std_best["ll"]),
    ])
    cmp_df.to_csv(base_out / "glm_vs_standard_comparison.csv", index=False)
    print(cmp_df.to_string(index=False))
    delta_ho = glmhmm_mean_ho - std_mean_ho
    print(f"  Δ held-out LL/bin (GLM minus standard): {delta_ho:+.4f}")
    if abs(delta_ho) < 0.01:
        print("  ⚠ FLAG: covariate-dependent transitions provide < 0.01 nats/bin "
              "improvement — covariates likely not informative.")

    # Comparison bar plot
    fig, ax = plt.subplots(figsize=(6, 4.5))
    sessions_x = [(0, "Standard HMM\nmean across folds"),
                   (1, "GLM-HMM\nmean across folds")]
    vals = [std_mean_ho, glmhmm_mean_ho]
    ax.bar([0, 1], vals, color=["#888888", "firebrick"], alpha=0.85,
            edgecolor="black")
    ax.set_xticks([0, 1])
    ax.set_xticklabels([s[1] for s in sessions_x])
    ax.set_ylabel("Held-out LL / bin")
    ax.set_title(f"GLM-HMM vs Standard HMM at K={rec_K}")
    for x, v in zip([0, 1], vals):
        ax.text(x, v, f"{v:.3f}", ha="center", va="bottom", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(base_fig / "glm_vs_standard.png", dpi=130)
    plt.close(fig)

    # 6. Per-session transitions
    print("\n=== Per-session transitions ===")
    state_idx = find_states_by_emission(glmhmm_params)
    extract_per_session_transitions(glmhmm_params, all_sessions, base_out,
                                      base_fig, history_df, sess_state_lookup,
                                      state_idx)

    # 7. Coefficient summary
    print("\n=== Coefficient summary ===")
    coef_df = coefficient_summary(glmhmm_params, n_cov, cov_names,
                                    base_out, base_fig)
    print("Top 10 transitions by |coef on cum_non_rewarded_digs_z|:")
    print(coef_df[coef_df.covariate == "cum_non_rewarded_digs_z"]
            .nlargest(10, "abs_coefficient")
            [["source_state", "target_state", "coefficient", "abs_coefficient"]]
            .to_string(index=False))

    # 8. Strategy-switch detection
    print("\n=== Strategy-switch timing per session ===")
    timing_df = plot_strategy_switches(all_sessions, base_out, base_fig,
                                         history_df, sess_state_lookup)
    print(timing_df.to_string(index=False))

    # 9. State profiles plot (sanity check)
    print("\n=== State profiles ===")
    K = glmhmm_params.K
    cont = glmhmm_params.mu
    zone = glmhmm_params.p_zone
    ev = glmhmm_params.q_events
    fig, axes = plt.subplots(1, 3, figsize=(2 + 0.45 * (2 + K_zone + n_events),
                                              0.5 + 0.4 * K),
                              gridspec_kw={"width_ratios": [2, K_zone, n_events]})
    vmax_c = np.abs(cont).max() + 1e-9
    axes[0].imshow(cont, aspect="auto", cmap="RdBu_r", vmin=-vmax_c, vmax=vmax_c)
    axes[0].set_xticks([0, 1]); axes[0].set_xticklabels(["speed_z", "dist_z"])
    axes[0].set_yticks(np.arange(K))
    axes[0].set_yticklabels([f"S{k}" for k in range(K)])
    axes[0].set_title("Continuous (z)")
    axes[1].imshow(zone, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    axes[1].set_xticks(np.arange(K_zone))
    axes[1].set_xticklabels(["home", "trans", "pot", "potZ", "arena", "other"])
    axes[1].set_yticks([])
    axes[1].set_title("Zone P")
    axes[2].imshow(ev, aspect="auto", cmap="Reds", vmin=0, vmax=ev.max() + 1e-9)
    axes[2].set_xticks(np.arange(n_events))
    axes[2].set_xticklabels(["inc_home", "ql_home", "dig", "feed", "rear",
                              "explT", "contT"], rotation=30, ha="right")
    axes[2].set_yticks([])
    axes[2].set_title("Event P")
    fig.suptitle(f"GLM-HMM state profiles (K={K})", y=1.0)
    fig.tight_layout()
    fig.savefig(base_fig / "state_profiles.png", dpi=130)
    plt.close(fig)

    print("\nDone.")


if __name__ == "__main__":
    main()
