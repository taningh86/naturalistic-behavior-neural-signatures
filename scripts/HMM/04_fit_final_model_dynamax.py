"""04 (dynamax) — Fit the final mixed-emission HMM at the chosen N.

Selects N from CLI --N flag, else from cfg["dynamax_selected_N"]. Runs
cfg["dynamax_n_inits"] random initializations on all 7 sessions pooled, picks
the init with highest training LL, saves params + meta.json.

Persistence: dynamax HMMs aren't trivially picklable, so we save plain numpy
parameter arrays + a meta.json. Reload via mixed_hmm.MixedHMMParams(...).
"""
from pathlib import Path
import sys
import time
import json
import argparse

import numpy as np
import pandas as pd

import jax

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _utils import load_config, ensure_dir, REPO_ROOT
import mixed_hmm as mh


def load_prepared_session(prepared_dir: Path, session_num: int):
    z = np.load(prepared_dir / f"session_{session_num}.npz", allow_pickle=True)
    return dict(
        X_cont=np.asarray(z["X_continuous"], dtype=np.float64),
        X_zone=np.asarray(z["X_zone"], dtype=np.int64),
        X_events=np.asarray(z["X_events"], dtype=np.float64),
        session_num=int(z["session_num"]),
        state=str(z["state"]),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=None)
    parser.add_argument("--n_inits", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config()
    print(f"JAX backend: {jax.default_backend()}  devices: {jax.devices()}")
    prepared_dir = REPO_ROOT / cfg["dynamax_dirs"]["prepared"]
    out_npz = REPO_ROOT / cfg["dynamax_dirs"]["final_model"]
    out_params_dir = ensure_dir(REPO_ROOT / cfg["dynamax_dirs"]["final_params"])
    ensure_dir(out_npz.parent)

    N = args.N if args.N is not None else cfg.get("dynamax_selected_N")
    if N is None:
        raise SystemExit("Must pass --N or set cfg['dynamax_selected_N']")
    n_inits = args.n_inits or cfg["dynamax_n_inits"]
    base_seed = int(cfg["random_seed"])

    fed = cfg["sessions"]["fed"]
    fasted = cfg["sessions"]["fasted"]
    all_sessions = fed + fasted

    sessions = [load_prepared_session(prepared_dir, sn) for sn in all_sessions]
    K_zone = int(sessions[0]["X_zone"].max() + 1)
    n_events = sessions[0]["X_events"].shape[1]
    total_T = sum(s["X_cont"].shape[0] for s in sessions)
    print(f"N = {N}, n_inits = {n_inits}, total_bins = {total_T}, "
          f"K_zone = {K_zone}, n_events = {n_events}")

    cont_pool = np.concatenate([s["X_cont"] for s in sessions], axis=0)
    zone_pool = np.concatenate([s["X_zone"] for s in sessions], axis=0)
    ev_pool = np.concatenate([s["X_events"] for s in sessions], axis=0)

    best = None
    init_records = []
    for ii in range(n_inits):
        seed = base_seed + 7000 * N + ii
        key = jax.random.PRNGKey(seed)
        params = mh.init_params(key, N, cont_pool, zone_pool, ev_pool, K_zone, n_events)
        t0 = time.time()
        fit_p, hist = mh.fit(
            params, sessions,
            max_iters=cfg["dynamax_em_max_iters"],
            tol=cfg["dynamax_em_tol"],
            var_floor=cfg["dynamax_var_floor"],
            zone_dirichlet=cfg["dynamax_zone_dirichlet"],
            event_beta=cfg["dynamax_event_beta"],
            verbose=False,
        )
        t = time.time() - t0
        rec = dict(init_idx=ii, seed=seed, ll=hist["final_loglik"],
                   n_iter=hist["n_iter"], time_s=t)
        init_records.append(rec)
        print(f"  init {ii}: ll={hist['final_loglik']:.2f}  iters={hist['n_iter']}  "
              f"({t:.1f}s)  seed={seed}")
        if best is None or hist["final_loglik"] > best["ll"]:
            best = dict(params=fit_p, hist=hist, seed=seed, init_idx=ii,
                        ll=hist["final_loglik"])

    fit_p = best["params"]
    print(f"\nBest init = {best['init_idx']}, ll = {best['ll']:.3f}, "
          f"per-bin ll = {best['ll']/total_T:.4f}")

    # Save numpy parameter arrays
    np.savez(
        out_npz,
        pi=fit_p.pi, A=fit_p.A,
        mu=fit_p.mu, sigma=fit_p.sigma,
        p_zone=fit_p.p_zone, q_events=fit_p.q_events,
        K=N, K_zone=K_zone, n_events=n_events,
    )
    print(f"Saved final model → {out_npz}")

    # Human-readable parameter CSVs
    pd.DataFrame({"state": np.arange(N), "pi": fit_p.pi}).to_csv(
        out_params_dir / "initial_distribution.csv", index=False
    )
    pd.DataFrame(fit_p.A,
                 index=[f"from_state_{i}" for i in range(N)],
                 columns=[f"to_state_{j}" for j in range(N)]).to_csv(
        out_params_dir / "transition_matrix.csv"
    )
    cont_names = ["speed_z", "distance_to_pot_z"]
    cont_rows = []
    for k in range(N):
        for d, nm in enumerate(cont_names):
            cont_rows.append(dict(state=k, feature=nm,
                                  mu=float(fit_p.mu[k, d]),
                                  sigma=float(fit_p.sigma[k, d])))
    pd.DataFrame(cont_rows).to_csv(out_params_dir / "emissions_continuous.csv", index=False)

    # Zone categorical
    z = np.load(prepared_dir / f"session_{all_sessions[0]}.npz", allow_pickle=True)
    zone_labels = list(z["zone_labels"])
    event_names = list(z["event_names"])
    zone_df = pd.DataFrame(fit_p.p_zone, columns=zone_labels)
    zone_df.insert(0, "state", np.arange(N))
    zone_df.to_csv(out_params_dir / "emissions_zone.csv", index=False)

    # Bernoulli events
    ev_df = pd.DataFrame(fit_p.q_events, columns=event_names)
    ev_df.insert(0, "state", np.arange(N))
    ev_df.to_csv(out_params_dir / "emissions_events.csv", index=False)

    meta = dict(
        N=int(N),
        K_zone=int(K_zone),
        n_events=int(n_events),
        total_bins=int(total_T),
        sessions=list(map(int, all_sessions)),
        fed_sessions=list(map(int, fed)),
        fasted_sessions=list(map(int, fasted)),
        n_inits_tried=int(n_inits),
        chosen_init_idx=int(best["init_idx"]),
        chosen_seed=int(best["seed"]),
        n_iters_to_convergence=int(best["hist"]["n_iter"]),
        final_log_likelihood=float(best["ll"]),
        per_bin_log_likelihood=float(best["ll"] / total_T),
        all_init_records=init_records,
        var_floor=float(cfg["dynamax_var_floor"]),
        zone_dirichlet=float(cfg["dynamax_zone_dirichlet"]),
        event_beta=float(cfg["dynamax_event_beta"]),
        em_max_iters=int(cfg["dynamax_em_max_iters"]),
        em_tol=float(cfg["dynamax_em_tol"]),
        jax_backend=str(jax.default_backend()),
        zone_labels=zone_labels,
        event_names=event_names,
    )
    with open(out_params_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Meta → {out_params_dir / 'meta.json'}")


if __name__ == "__main__":
    main()
