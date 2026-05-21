"""03 — Cross-validated state selection for the HMM.

Strategy: 3 stratified folds (each holds out 1 fed + 1 fasted session) since 7
sessions does not divide cleanly into k=4 leave-2-out. For each N in N_range and
each fold, fit on training and report held-out log-likelihood per bin.

Output:
  data/HMM/cv_results.csv
  figures/HMM/state_selection.png
  Recommended N (smallest N within 1 SE of max) printed and written back to
  config.yaml as `selected_N`.
"""
from itertools import product
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _utils import load_config, session_list, ensure_dir, REPO_ROOT, CONFIG_PATH

import ssm  # noqa: E402


def make_folds(sessions, k):
    """Stratified k-fold; each fold holds out 1 fed + 1 fasted (or as close as possible).

    sessions = [(num, state), ...]
    Returns list of (train_idx, test_idx) tuples (indices into sessions).
    """
    fed = [i for i, (_, s) in enumerate(sessions) if s == "fed"]
    fas = [i for i, (_, s) in enumerate(sessions) if s == "fasted"]
    rng = np.random.default_rng(20260429)
    rng.shuffle(fed)
    rng.shuffle(fas)
    folds = []
    for f in range(k):
        held = []
        if f < len(fed):
            held.append(fed[f])
        if f < len(fas):
            held.append(fas[f])
        if not held:
            continue
        train = [i for i in range(len(sessions)) if i not in held]
        folds.append((train, held))
    return folds


def fit_and_score(X_train_list, X_test_list, K, D, cfg, seed):
    """Fit HMM on training, return mean held-out log-lik per bin."""
    np.random.seed(seed)
    hmm = ssm.HMM(K, D, observations=cfg["hmm_observations"])
    hmm.fit(
        X_train_list,
        method=cfg["hmm_method"],
        num_iters=cfg["hmm_n_iters"],
        tolerance=cfg["hmm_tolerance"],
        verbose=0,
    )
    total_ll = 0.0
    total_T = 0
    for X in X_test_list:
        total_ll += float(hmm.log_likelihood(X))
        total_T += X.shape[0]
    return total_ll / total_T, total_T


def main():
    cfg = load_config()
    prepared_dir = REPO_ROOT / cfg["out_dirs"]["prepared"]
    sess = session_list(cfg)

    # Load all prepared X
    all_data = []
    for session_num, state in sess:
        p = prepared_dir / f"session_{session_num}.npz"
        if not p.exists():
            print(f"  SKIP S{session_num}: missing {p}")
            continue
        z = np.load(p, allow_pickle=True)
        all_data.append((session_num, state, z["X"]))
    if not all_data:
        raise SystemExit("No prepared sessions; run 02 first.")

    sessions = [(s, st) for s, st, _ in all_data]
    Xs = [x for _, _, x in all_data]
    D = Xs[0].shape[1]
    print(f"Loaded {len(Xs)} sessions, D={D}")
    print(f"Bin counts: {[x.shape[0] for x in Xs]}")

    folds = make_folds(sessions, cfg["cv_folds"])
    print(f"\n{len(folds)} folds:")
    for i, (tr, te) in enumerate(folds):
        tr_names = [f"S{sessions[j][0]}({sessions[j][1][:3]})" for j in tr]
        te_names = [f"S{sessions[j][0]}({sessions[j][1][:3]})" for j in te]
        print(f"  fold {i}: train={tr_names} test={te_names}")

    rows = []
    for N in cfg["N_range"]:
        print(f"\n=== N = {N} ===")
        fold_ll = []
        for f, (tr, te) in enumerate(folds):
            X_tr = [Xs[j] for j in tr]
            X_te = [Xs[j] for j in te]
            seed = cfg["random_seed"] + 13 * N + f
            ll_per_bin, T_te = fit_and_score(X_tr, X_te, N, D, cfg, seed)
            print(f"  fold {f}: held-out ll/bin = {ll_per_bin:.4f}  (T_te={T_te})")
            fold_ll.append(ll_per_bin)
            rows.append(
                dict(N=N, fold=f, ll_per_bin=ll_per_bin, T_test=T_te)
            )
        mean = float(np.mean(fold_ll))
        sd = float(np.std(fold_ll, ddof=1)) if len(fold_ll) > 1 else 0.0
        print(f"  N={N}: mean={mean:.4f}, sd={sd:.4f}")

    df = pd.DataFrame(rows)
    csv_path = REPO_ROOT / cfg["out_dirs"]["cv_results"]
    ensure_dir(csv_path.parent)
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {csv_path}")

    # Aggregate
    agg = df.groupby("N")["ll_per_bin"].agg(["mean", "std", "count"]).reset_index()
    agg["se"] = agg["std"] / np.sqrt(agg["count"])
    print("\nAggregate:")
    print(agg.to_string(index=False))

    # Recommend N: smallest N within 1 SE of max-mean
    best = agg.loc[agg["mean"].idxmax()]
    threshold = best["mean"] - best["se"]
    eligible = agg[agg["mean"] >= threshold].sort_values("N")
    recommended = int(eligible.iloc[0]["N"])
    print(f"\nMax mean ll/bin = {best['mean']:.4f} at N={int(best['N'])} "
          f"(SE={best['se']:.4f})")
    print(f"Recommended N (smallest within 1 SE of max): {recommended}")

    # Plot
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.errorbar(
        agg["N"], agg["mean"], yerr=agg["se"], fmt="o-",
        color="steelblue", capsize=4, markersize=7,
    )
    ax.axhline(threshold, ls="--", color="grey", alpha=0.6,
               label=f"1 SE below max (N={int(best['N'])})")
    ax.axvline(recommended, ls=":", color="firebrick", alpha=0.7,
               label=f"Recommended N={recommended}")
    ax.set_xlabel("Number of states (N)")
    ax.set_ylabel("Held-out log-likelihood / bin")
    ax.set_title("HMM state-count selection (cross-validated)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig_path = REPO_ROOT / cfg["out_dirs"]["state_selection_fig"]
    ensure_dir(fig_path.parent)
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"Saved {fig_path}")

    # Write back to config
    with open(CONFIG_PATH) as f:
        c = yaml.safe_load(f)
    c["selected_N"] = recommended
    with open(CONFIG_PATH, "w") as f:
        yaml.safe_dump(c, f, default_flow_style=False, sort_keys=False)
    print(f"Updated config.yaml: selected_N = {recommended}")


if __name__ == "__main__":
    main()
