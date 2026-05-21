"""Main pipeline for fitting Poisson-GLM-HMMs to dual-region spike data.

Modes:
  fit-one   — Fit one (region, phase, K) on all sessions with N random restarts.
              Saves best model + diagnostics.
  cv         — Leave-one-session-out CV across a K range for one (region, phase).
              Saves per-K held-out LL.
  final      — After CV, fit final model at chosen K on all sessions.
              Saves model + transitions + emissions + occupancy + plots.
  all        — Run cv then final for all 4 (region, phase) combinations.

Output layout (per the spec):
  results/hmm/{region}/{phase}/K{k}/
    model.pkl, config.yaml, state_assignments.npz,
    transitions_by_condition.npz, emissions.npz, occupancy.csv,
    plots/{occupancy.png, transitions.png, emissions.png, timeline_*.png}
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "HMM"))
sys.path.insert(0, str(REPO / "scripts" / "HMM_glm"))

from _utils import load_config
from _data import (load_grouped, prepare_for_fit, METABOLIC_STATES,
                     SESSIONS_BY_PHASE_STATE, BIN_S_DEFAULT)
from _model import (fit_one, evaluate_ll, state_assignments,
                      effective_transitions, emission_log_rates, save_model)
from _plots import (plot_state_occupancy_heatmap, plot_transition_matrices,
                      plot_emission_heatmap, plot_state_timeline,
                      plot_cv_curve)


RESULTS_ROOT = REPO / "results" / "hmm"
LOG_ROOT = REPO / "data" / "HMM_glm" / "logs"


def out_dir(region: str, phase: str, K: int) -> Path:
    p = RESULTS_ROOT / region / phase / f"K{K:02d}"
    (p / "plots").mkdir(parents=True, exist_ok=True)
    return p


def write_config(out: Path, **kwargs) -> None:
    with open(out / "config.yaml", "w") as f:
        yaml.safe_dump(kwargs, f, sort_keys=False)


# ============================================================================
# fit-one mode: one (region, phase, K), all sessions, N restarts
# ============================================================================
def fit_one_mode(region: str, phase: str, K: int, n_restarts: int,
                  num_iters: int, bin_s: float, seed_base: int,
                  cfg) -> dict:
    print(f"\n=== fit-one: {region} {phase} K={K}, "
          f"restarts={n_restarts}, bin_s={bin_s} ===", flush=True)
    t0 = time.time()
    grouped = load_grouped(region, phase, cfg, bin_s=bin_s)
    if not grouped.sequences:
        print("  No sessions loaded; aborting.")
        return None
    datasets, D, M, pooled_ids, sess2cols = prepare_for_fit(grouped)
    counts_list = [d["counts"] for d in datasets]
    inputs_list = [d["input"] for d in datasets]
    print(f"  D_total={D}, M={M}, sessions={len(datasets)}, "
          f"total_bins={sum(c.shape[0] for c in counts_list)}", flush=True)

    best_hmm = None
    best_ll = -np.inf
    best_init = -1
    init_lls = []
    for r in range(n_restarts):
        t_init = time.time()
        seed = seed_base + r
        hmm, ll_history = fit_one(counts_list, inputs_list,
                                     K=K, D=D, M=M,
                                     num_iters=num_iters,
                                     seed=seed, verbose=False)
        final_ll = float(ll_history[-1]) if ll_history else -np.inf
        init_lls.append(dict(restart=r, seed=seed, train_ll=final_ll,
                              dt_s=time.time() - t_init,
                              n_iters=len(ll_history)))
        print(f"    restart {r} seed={seed}: train_ll={final_ll:.2f}, "
              f"iters={len(ll_history)}, dt={time.time()-t_init:.1f}s",
              flush=True)
        if final_ll > best_ll:
            best_ll = final_ll
            best_hmm = hmm
            best_init = r

    # Save best
    out = out_dir(region, phase, K)
    save_model(best_hmm, out / "model.pkl")
    pd.DataFrame(init_lls).to_csv(out / "restart_log.csv", index=False)
    np.savez(out / "pooled_ids.npz", pooled_ids=pooled_ids,
              session_nums=[d["session_num"] for d in datasets],
              metabolic_states=[d["metabolic_state"] for d in datasets])
    write_config(out, region=region, phase=phase, K=K, M=M, D=D,
                  bin_s=bin_s, n_restarts=n_restarts, best_init=best_init,
                  best_train_ll=best_ll, num_iters=num_iters,
                  sessions=[d["session_num"] for d in datasets],
                  metabolic_states=[d["metabolic_state"] for d in datasets])

    # Per-session state assignments + occupancy
    z_by_session = {}
    occupancy_rows = []
    for d in datasets:
        z = state_assignments(best_hmm, d["counts"], d["input"])
        z_by_session[d["session_num"]] = z
        # occupancy
        occ = np.bincount(z, minlength=K) / max(1, len(z))
        for k in range(K):
            occupancy_rows.append(dict(
                session=d["session_num"],
                metabolic_state=d["metabolic_state"],
                state=k, occupancy=float(occ[k]),
                n_bins=int(len(z)),
            ))
    np.savez(out / "state_assignments.npz",
              **{f"session_{sn}": z for sn, z in z_by_session.items()})
    occ_df = pd.DataFrame(occupancy_rows)
    occ_df.to_csv(out / "occupancy.csv", index=False)

    # Transitions per metabolic state
    trans = effective_transitions(best_hmm, one_hot_dim=M)
    np.savez(out / "transitions_by_condition.npz",
              metabolic_states=METABOLIC_STATES,
              transitions=trans)

    # Emissions
    emissions = emission_log_rates(best_hmm)
    np.savez(out / "emissions.npz",
              log_rates=emissions, pooled_ids=pooled_ids)

    # Plots
    occ_pivot = occ_df.pivot_table(
        index="state", columns="session", values="occupancy", aggfunc="first",
    ).fillna(0).sort_index()
    plot_state_occupancy_heatmap(
        occ_pivot, K,
        out / "plots" / "occupancy.png",
        title=f"{region} {phase} K={K}: state occupancy per session",
    )
    plot_transition_matrices(
        trans, METABOLIC_STATES,
        out / "plots" / "transitions.png",
        title=f"{region} {phase} K={K}: effective transitions per metabolic state",
    )
    plot_emission_heatmap(
        emissions, pooled_ids,
        out / "plots" / "emissions.png",
        title=f"{region} {phase} K={K}: emission log-rates",
    )
    # one timeline per state
    seen = set()
    for d in datasets:
        if d["metabolic_state"] in seen:
            continue
        seen.add(d["metabolic_state"])
        z = z_by_session[d["session_num"]]
        plot_state_timeline(
            z, K,
            out / "plots" / f"timeline_S{d['session_num']}_{d['metabolic_state']}.png",
            title=f"S{d['session_num']} ({d['metabolic_state']}) Viterbi states",
        )

    print(f"  Done {region} {phase} K={K} [{time.time()-t0:.0f}s]. "
          f"Saved to {out}", flush=True)
    return dict(K=K, train_ll=best_ll, out=str(out))


# ============================================================================
# CV mode
# ============================================================================
def cv_mode(region: str, phase: str, k_range: list[int], n_restarts: int,
             num_iters: int, bin_s: float, seed_base: int, cfg) -> pd.DataFrame:
    print(f"\n=== CV: {region} {phase} K∈{k_range} ===", flush=True)
    grouped = load_grouped(region, phase, cfg, bin_s=bin_s)
    if not grouped.sequences:
        print("  No sessions loaded; aborting.")
        return None
    datasets, D, M, pooled_ids, sess2cols = prepare_for_fit(grouped)
    counts_list = [d["counts"] for d in datasets]
    inputs_list = [d["input"] for d in datasets]
    n_sess = len(datasets)
    sess_nums = [d["session_num"] for d in datasets]

    rows = []
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    cv_log = LOG_ROOT / f"cv_{region}_{phase}.csv"
    for K in k_range:
        for fold, held_out_idx in enumerate(range(n_sess)):
            train_idx = [i for i in range(n_sess) if i != held_out_idx]
            tr_counts = [counts_list[i] for i in train_idx]
            tr_inputs = [inputs_list[i] for i in train_idx]
            ho_counts = [counts_list[held_out_idx]]
            ho_inputs = [inputs_list[held_out_idx]]
            # Fit with N restarts on training; pick best by train LL
            best_hmm = None; best_train_ll = -np.inf
            for r in range(n_restarts):
                seed = seed_base + 1000 * K + 10 * fold + r
                t0 = time.time()
                hmm, hist = fit_one(tr_counts, tr_inputs,
                                       K=K, D=D, M=M,
                                       num_iters=num_iters, seed=seed)
                train_ll = float(hist[-1]) if hist else -np.inf
                dt = time.time() - t0
                if train_ll > best_train_ll:
                    best_train_ll = train_ll
                    best_hmm = hmm
                rows.append(dict(K=K, fold=fold,
                                  held_out_session=sess_nums[held_out_idx],
                                  restart=r, train_ll=train_ll,
                                  ho_ll=np.nan, ho_bins=np.nan,
                                  is_best=False, dt_s=dt))
                print(f"  K={K} fold={fold} ho=S{sess_nums[held_out_idx]} "
                      f"r={r}: train_ll={train_ll:.1f} [{dt:.1f}s]", flush=True)
            # Eval on held-out with best model
            ho_ll, ho_bins = evaluate_ll(best_hmm, ho_counts, ho_inputs)
            # mark the best restart's row
            for row in rows[-n_restarts:]:
                if row["train_ll"] == best_train_ll:
                    row["is_best"] = True
                    row["ho_ll"] = ho_ll
                    row["ho_bins"] = ho_bins
                    break
            print(f"    K={K} fold={fold} BEST: ho_ll/bin="
                  f"{ho_ll/max(1,ho_bins):.4f}", flush=True)
            pd.DataFrame(rows).to_csv(cv_log, index=False)
    df = pd.DataFrame(rows)
    df.to_csv(cv_log, index=False)
    return df


def aggregate_cv(cv_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-fold held-out LL into per-K summary."""
    best = cv_df[cv_df.is_best & cv_df.ho_ll.notna()].copy()
    best["ho_ll_per_bin"] = best["ho_ll"] / best["ho_bins"]
    summary = best.groupby("K").agg(
        n_folds=("ho_ll_per_bin", "size"),
        mean_ho_ll_per_bin=("ho_ll_per_bin", "mean"),
        sem_ho_ll_per_bin=("ho_ll_per_bin", "sem"),
        mean_train_ll=("train_ll", "mean"),
    ).reset_index()
    return summary


# ============================================================================
# CLI
# ============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("fit-one", "cv", "all"), default="fit-one")
    ap.add_argument("--region", choices=("ACA", "LHA"), default="ACA")
    ap.add_argument("--phase", choices=("exploration", "foraging"), default="foraging")
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--k-min", type=int, default=2)
    ap.add_argument("--k-max", type=int, default=10)
    ap.add_argument("--n-restarts", type=int, default=5)
    ap.add_argument("--num-iters", type=int, default=100)
    ap.add_argument("--bin-s", type=float, default=BIN_S_DEFAULT)
    ap.add_argument("--seed", type=int, default=20260513)
    args = ap.parse_args()

    cfg = load_config()
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)

    if args.mode == "fit-one":
        fit_one_mode(args.region, args.phase, args.K, args.n_restarts,
                      args.num_iters, args.bin_s, args.seed, cfg)
    elif args.mode == "cv":
        k_range = list(range(args.k_min, args.k_max + 1))
        df = cv_mode(args.region, args.phase, k_range, args.n_restarts,
                       args.num_iters, args.bin_s, args.seed, cfg)
        summary = aggregate_cv(df)
        out = RESULTS_ROOT / args.region / args.phase
        out.mkdir(parents=True, exist_ok=True)
        summary.to_csv(out / "cv_summary.csv", index=False)
        plot_cv_curve(summary["K"].tolist(),
                        summary["mean_ho_ll_per_bin"].values,
                        summary["sem_ho_ll_per_bin"].fillna(0).values,
                        (summary["mean_train_ll"]
                          / summary["n_folds"].clip(lower=1)).values * 0,  # placeholder
                        out / "cv_curve.png",
                        title=f"{args.region} {args.phase}: CV-curve")
        print(f"Saved CV summary: {out / 'cv_summary.csv'}")
        print(summary.to_string(index=False))
    elif args.mode == "all":
        k_range = list(range(args.k_min, args.k_max + 1))
        for region in ("ACA", "LHA"):
            for phase in ("exploration", "foraging"):
                df = cv_mode(region, phase, k_range, args.n_restarts,
                               args.num_iters, args.bin_s, args.seed, cfg)
                summary = aggregate_cv(df)
                out = RESULTS_ROOT / region / phase
                out.mkdir(parents=True, exist_ok=True)
                summary.to_csv(out / "cv_summary.csv", index=False)
                print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
