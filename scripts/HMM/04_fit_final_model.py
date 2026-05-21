"""04 — Fit final HMM with selected N on all sessions pooled.

Saves: data/HMM/final_model.pkl
        data/HMM/final_model_params/initial_distribution.csv
        data/HMM/final_model_params/transition_matrix.csv
        data/HMM/final_model_params/emissions.csv  (per-state, per-feature mean & sigma)
        data/HMM/final_model_params/meta.json
"""
import argparse
import json
import pickle
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _utils import load_config, session_list, ensure_dir, REPO_ROOT

import ssm  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=None,
                        help="override selected_N from config")
    args = parser.parse_args()

    cfg = load_config()
    N = args.N if args.N is not None else cfg.get("selected_N")
    if N is None:
        raise SystemExit("No N specified: pass --N or run 03 first")

    prepared_dir = REPO_ROOT / cfg["out_dirs"]["prepared"]
    sess = session_list(cfg)

    Xs = []
    sids = []
    states = []
    feature_names = None
    for session_num, state in sess:
        p = prepared_dir / f"session_{session_num}.npz"
        if not p.exists():
            print(f"  SKIP S{session_num}: missing {p}")
            continue
        z = np.load(p, allow_pickle=True)
        Xs.append(z["X"])
        sids.append(int(z["session_num"]))
        states.append(str(z["state"]))
        if feature_names is None:
            feature_names = list(z["feature_names"])

    if not Xs:
        raise SystemExit("No prepared data; run 02 first")

    D = Xs[0].shape[1]
    print(f"Fitting final HMM: N={N}, D={D}, sessions={sids}")
    print(f"Total bins across sessions: {sum(x.shape[0] for x in Xs)}")

    np.random.seed(cfg["random_seed"])
    hmm = ssm.HMM(N, D, observations=cfg["hmm_observations"])
    log_likes = hmm.fit(
        Xs,
        method=cfg["hmm_method"],
        num_iters=cfg["hmm_n_iters"],
        tolerance=cfg["hmm_tolerance"],
        verbose=2,
    )
    final_ll = float(np.sum([hmm.log_likelihood(X) for X in Xs]))
    total_T = int(sum(x.shape[0] for x in Xs))
    print(f"Final total ll = {final_ll:.2f} ({final_ll/total_T:.4f} per bin)")

    # Save model
    model_path = REPO_ROOT / cfg["out_dirs"]["final_model"]
    ensure_dir(model_path.parent)
    with open(model_path, "wb") as f:
        pickle.dump(
            dict(
                hmm=hmm,
                N=N,
                D=D,
                feature_names=feature_names,
                session_nums=sids,
                states=states,
                final_log_likelihood=final_ll,
                fit_log_likes=list(map(float, log_likes)),
            ),
            f,
        )
    print(f"Saved {model_path}")

    # Human-readable params
    params_dir = ensure_dir(REPO_ROOT / cfg["out_dirs"]["final_params"])

    # Initial distribution
    init = np.exp(hmm.init_state_distn.log_pi0 - np.max(hmm.init_state_distn.log_pi0))
    init /= init.sum()
    pd.DataFrame({"state": np.arange(N), "p_init": init}).to_csv(
        params_dir / "initial_distribution.csv", index=False
    )

    # Transition matrix
    P = hmm.transitions.transition_matrix
    pd.DataFrame(P, columns=[f"to_state_{i}" for i in range(N)]).to_csv(
        params_dir / "transition_matrix.csv", index_label="from_state"
    )

    # Emissions: per-state per-feature mean and sigma (Gaussian observations)
    obs = hmm.observations
    mus = obs.mus  # (N, D)
    sigmasq = np.exp(obs.log_sigmasq) if hasattr(obs, "log_sigmasq") else None
    if sigmasq is None and hasattr(obs, "Sigmas"):
        sigmasq = np.array([np.diag(s) for s in obs.Sigmas])
    rows = []
    for k in range(N):
        for j, fname in enumerate(feature_names):
            rows.append(dict(
                state=k, feature=fname,
                mean=float(mus[k, j]),
                sigma=float(np.sqrt(sigmasq[k, j])) if sigmasq is not None else float("nan"),
            ))
    pd.DataFrame(rows).to_csv(params_dir / "emissions.csv", index=False)

    # Meta
    meta = dict(
        N=N,
        D=D,
        feature_names=feature_names,
        sessions=[dict(session_num=s, state=st) for s, st in zip(sids, states)],
        final_log_likelihood=final_ll,
        per_bin_log_likelihood=final_ll / total_T,
        n_iter_actual=len(log_likes),
    )
    with open(params_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved params to {params_dir}")


if __name__ == "__main__":
    main()
