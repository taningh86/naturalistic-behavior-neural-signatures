"""03 (dynamax) — CV-based state-count selection for the mixed-emission HMM.

3 stratified CV folds, each holding out 1 fed + 1 fasted session (matches the
ssm pipeline). For each N in cfg["dynamax_N_range"]:
  - For each fold:
      - Run cfg["dynamax_n_inits"] random initializations on the training sessions
      - Fit until convergence (max cfg["dynamax_em_max_iters"], tol cfg["dynamax_em_tol"])
      - Keep the init with highest training log-likelihood
      - Compute held-out per-bin log-likelihood on test sessions using the best-init model
  - Aggregate mean and SE across folds.
Plot held-out LL vs N (mean ± SE), apply 1-SE rule, save CV table.
"""
from pathlib import Path
import sys
import time
import argparse

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import jax

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _utils import load_config, ensure_dir, REPO_ROOT
import mixed_hmm as mh


CV_FOLDS = [
    # (held_out_fed_session, held_out_fasted_session)
    (4, 12),
    (6, 14),
    (8, 16),
]


def load_prepared_session(prepared_dir: Path, session_num: int):
    z = np.load(prepared_dir / f"session_{session_num}.npz", allow_pickle=True)
    return dict(
        X_cont=np.asarray(z["X_continuous"], dtype=np.float64),
        X_zone=np.asarray(z["X_zone"], dtype=np.int64),
        X_events=np.asarray(z["X_events"], dtype=np.float64),
        session_num=int(z["session_num"]),
        state=str(z["state"]),
    )


def fit_one_init(seed, K, train_sessions, K_zone, n_events, cfg):
    """Random init + EM until convergence. Returns (params, history)."""
    key = jax.random.PRNGKey(seed)
    # pool training data for init stats
    cont_pool = np.concatenate([s["X_cont"] for s in train_sessions], axis=0)
    zone_pool = np.concatenate([s["X_zone"] for s in train_sessions], axis=0)
    ev_pool = np.concatenate([s["X_events"] for s in train_sessions], axis=0)

    params = mh.init_params(
        key, K=K,
        X_cont_pool=cont_pool, X_zone_pool=zone_pool, X_events_pool=ev_pool,
        K_zone=K_zone, n_events=n_events,
    )
    params, history = mh.fit(
        params, train_sessions,
        max_iters=cfg["dynamax_em_max_iters"],
        tol=cfg["dynamax_em_tol"],
        var_floor=cfg["dynamax_var_floor"],
        zone_dirichlet=cfg["dynamax_zone_dirichlet"],
        event_beta=cfg["dynamax_event_beta"],
        verbose=False,
    )
    return params, history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N_range", nargs="*", type=int, default=None)
    parser.add_argument("--n_inits", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config()
    print(f"JAX backend: {jax.default_backend()}  devices: {jax.devices()}")
    prepared_dir = REPO_ROOT / cfg["dynamax_dirs"]["prepared"]
    cv_csv = REPO_ROOT / cfg["dynamax_dirs"]["cv_results"]
    fig_path = REPO_ROOT / cfg["dynamax_dirs"]["state_selection_fig"]
    ensure_dir(cv_csv.parent)
    ensure_dir(fig_path.parent)

    N_range = args.N_range or cfg["dynamax_N_range"]
    n_inits = args.n_inits or cfg["dynamax_n_inits"]
    base_seed = int(cfg["random_seed"])

    print(f"N_range: {N_range}")
    print(f"n_inits per fold: {n_inits}")
    print(f"CV folds: {CV_FOLDS}")

    fed_sessions = cfg["sessions"]["fed"]
    fasted_sessions = cfg["sessions"]["fasted"]
    all_sessions = fed_sessions + fasted_sessions

    # Pre-load all prepared sessions
    cache = {sn: load_prepared_session(prepared_dir, sn) for sn in all_sessions}
    K_zone = int(cache[all_sessions[0]]["X_zone"].max() + 1)
    n_events = cache[all_sessions[0]]["X_events"].shape[1]
    print(f"K_zone={K_zone}, n_events={n_events}")

    rows = []
    for N in N_range:
        print(f"\n=== N = {N} ===")
        for fi, (held_fed, held_fasted) in enumerate(CV_FOLDS):
            train_ids = [s for s in all_sessions if s not in (held_fed, held_fasted)]
            test_ids = [held_fed, held_fasted]
            train_sessions = [cache[s] for s in train_ids]
            test_sessions = [cache[s] for s in test_ids]
            test_T = sum(s["X_cont"].shape[0] for s in test_sessions)

            best_train_ll = -np.inf
            best_held_ll_per_bin = None
            best_init = None
            for ii in range(n_inits):
                seed = base_seed + 1000 * (fi + 1) + 100 * N + ii
                t0 = time.time()
                params, hist = fit_one_init(
                    seed, N, train_sessions, K_zone, n_events, cfg,
                )
                t_fit = time.time() - t0

                train_ll = hist["final_loglik"]
                ho_ll, ho_T = mh.held_out_loglik(params, test_sessions)
                ho_per_bin = ho_ll / ho_T

                rows.append(dict(
                    N=N, fold=fi,
                    held_fed=held_fed, held_fasted=held_fasted,
                    init_idx=ii, seed=seed,
                    train_ll=train_ll, n_iter=hist["n_iter"],
                    heldout_ll=ho_ll, heldout_T=ho_T,
                    heldout_ll_per_bin=ho_per_bin,
                    fit_time_s=t_fit,
                ))
                print(f"  fold {fi} init {ii:>2}: train_ll={train_ll:>10.2f}  "
                      f"iters={hist['n_iter']:>3}  ho_ll/bin={ho_per_bin:>7.4f}  "
                      f"({t_fit:.1f}s)")

                if train_ll > best_train_ll:
                    best_train_ll = train_ll
                    best_held_ll_per_bin = ho_per_bin
                    best_init = ii
            print(f"  fold {fi} BEST init={best_init}  "
                  f"train_ll={best_train_ll:.2f}  ho_ll/bin={best_held_ll_per_bin:.4f}")

        # Save partial after each N
        df = pd.DataFrame(rows)
        df.to_csv(cv_csv, index=False)

    df = pd.DataFrame(rows)
    df.to_csv(cv_csv, index=False)
    print(f"\nCV results → {cv_csv}")

    # Aggregate: per (N, fold) pick best init by train_ll, then mean/SE across folds.
    best_per_fold = (
        df.sort_values("train_ll", ascending=False)
        .groupby(["N", "fold"], as_index=False)
        .first()
    )
    agg = best_per_fold.groupby("N", as_index=False).agg(
        mean_ll_per_bin=("heldout_ll_per_bin", "mean"),
        se_ll_per_bin=("heldout_ll_per_bin", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
        n_folds=("fold", "count"),
    )
    print("\nPer-N (best init per fold) aggregate:")
    print(agg.to_string(index=False))

    # 1-SE rule: smallest N within 1 SE of max mean
    max_mean = agg["mean_ll_per_bin"].max()
    threshold = max_mean - agg.loc[agg["mean_ll_per_bin"].idxmax(), "se_ll_per_bin"]
    eligible = agg[agg["mean_ll_per_bin"] >= threshold]
    recommended_N = int(eligible["N"].min())
    print(f"\nMax mean ll/bin: {max_mean:.4f} at N={int(agg.loc[agg['mean_ll_per_bin'].idxmax(),'N'])}")
    print(f"1-SE threshold: {threshold:.4f}")
    print(f"Recommended N (smallest within 1 SE of max): {recommended_N}")

    # Plot
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(
        agg["N"], agg["mean_ll_per_bin"],
        yerr=agg["se_ll_per_bin"], marker="o", lw=1.5, capsize=3, color="steelblue",
        label="held-out ll/bin (mean ± SE)",
    )
    ax.axhline(threshold, color="grey", lw=0.8, ls="--",
               label=f"1-SE threshold ({threshold:.3f})")
    ax.axvline(recommended_N, color="firebrick", lw=1.0, ls=":",
               label=f"recommended N = {recommended_N}")
    ax.set_xlabel("N states")
    ax.set_ylabel("Held-out log-likelihood per bin")
    ax.set_title("dynamax mixed-emission HMM — CV state selection")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    print(f"Plot → {fig_path}")


if __name__ == "__main__":
    main()
