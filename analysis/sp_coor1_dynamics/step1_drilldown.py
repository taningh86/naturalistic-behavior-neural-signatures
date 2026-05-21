"""Stage 1 dynamics drill-down for single-probe Mouse01-Coordinates1.

Mirror of analysis/dynamics_stage1/stage1_batch.py for the single-probe data:
- Regions: LHA (depth < 1300 um) and RSP (depth >= 1300 um), same probe
- 8 sessions: 1-4 fed, 5-8 fasted (alternating exploration/foraging)
- Same neural preprocessing, entropy pipeline, and phase-summary outputs

Outputs per session:
  data/sp_coor1_dynamics/session_{N}_speed.npy
  data/sp_coor1_dynamics/session_{N}_curvature.npy
  data/sp_coor1_dynamics/session_{N}_phases.json
  data/sp_coor1_dynamics/session_{N}_phase_data.npz
  data/sp_coor1_dynamics/session_{N}_summary.csv
  figures/sp_coor1_dynamics/session_{N}_diagnostic.png
Aggregated:
  data/sp_coor1_dynamics/all_sessions_summary.csv
  data/sp_coor1_dynamics/batch_log.txt
"""
import sys
import time
import traceback
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "analysis" / "sp_coor1_dynamics"))

from sp_lib import (
    BIN_S, list_sessions,
    load_neural, load_behavior, filter_behavior,
    load_zones_and_velocity, compute_entropy_series, detect_inflections,
    define_phases, compute_speed, compute_curvature,
    phase_summary_row, interp_entropy_to_bins,
)

OUTDIR = REPO / "data" / "sp_coor1_dynamics"
FIGDIR = REPO / "figures" / "sp_coor1_dynamics"
OUTDIR.mkdir(parents=True, exist_ok=True)
FIGDIR.mkdir(parents=True, exist_ok=True)


def run_session(sess_meta, log):
    sn = sess_meta['session']
    state, phase = sess_meta['state'], sess_meta['phase']
    log(f"\n=== S{sn}  {state} / {phase} ===")
    t0 = time.time()

    lha, bin_centers, n_lha = load_neural(sn, 'LHA')
    rsp, _, n_rsp = load_neural(sn, 'RSP')
    n_bins = min(lha.shape[0], rsp.shape[0])
    lha, rsp = lha[:n_bins], rsp[:n_bins]
    bin_centers = bin_centers[:n_bins]
    log(f"  neural: LHA {lha.shape} ({n_lha} units)  RSP {rsp.shape} ({n_rsp} units)  ({n_bins} bins)")

    behav = filter_behavior(load_behavior(sn, bin_centers))
    log(f"  behavior keys: {len(behav)}")

    time_vals, zones, vel = load_zones_and_velocity(sess_meta['behavior'])
    ent_t, ent_v, _ = compute_entropy_series(zones, time_vals, vel)
    peaks, troughs, smoothed = detect_inflections(ent_t, ent_v)
    log(f"  entropy pts={len(ent_t)}  peaks={len(peaks)}  troughs={len(troughs)}")

    phases = define_phases(ent_t, peaks, troughs, n_bins)
    log(f"  phases: {len(phases)}")

    sp_lha = compute_speed(lha)
    sp_rsp = compute_speed(rsp)
    cv_lha = compute_curvature(lha)
    cv_rsp = compute_curvature(rsp)
    log(f"  LHA spd {sp_lha.mean():.3f}+/-{sp_lha.std():.3f}, "
        f"RSP spd {sp_rsp.mean():.3f}+/-{sp_rsp.std():.3f}")

    fr_lha = lha.mean(axis=1)
    fr_rsp = rsp.mean(axis=1)
    pc1_lha = PCA(n_components=1).fit_transform(lha).ravel()
    pc1_rsp = PCA(n_components=1).fit_transform(rsp).ravel()
    ent_interp = interp_entropy_to_bins(ent_t, smoothed, bin_centers)

    rs = {'LHA': sp_lha, 'RSP': sp_rsp}
    rc = {'LHA': cv_lha, 'RSP': cv_rsp}
    rfr = {'LHA': fr_lha, 'RSP': fr_rsp}
    rpc = {'LHA': pc1_lha, 'RSP': pc1_rsp}

    rows = []
    for ph in phases:
        row = phase_summary_row(ph, None, behav, rs, rc, rfr, rpc, ent_interp)
        if row is not None:
            row['session'] = sn
            row['state'] = state
            row['exp_phase'] = phase
            row['n_units_LHA'] = n_lha
            row['n_units_RSP'] = n_rsp
            rows.append(row)
    df = pd.DataFrame(rows)

    np.save(OUTDIR / f"session_{sn}_speed.npy",
            dict(LHA=sp_lha, RSP=sp_rsp), allow_pickle=True)
    np.save(OUTDIR / f"session_{sn}_curvature.npy",
            dict(LHA=cv_lha, RSP=cv_rsp), allow_pickle=True)
    with open(OUTDIR / f"session_{sn}_phases.json", 'w') as f:
        json.dump([{k: (v.item() if isinstance(v, np.generic) else v)
                    for k, v in p.items()} for p in phases], f, indent=2)
    np.savez(OUTDIR / f"session_{sn}_phase_data.npz",
             ent_t=ent_t, ent_v=ent_v, ent_smoothed=smoothed,
             peaks=peaks, troughs=troughs, bin_centers=bin_centers,
             fr_lha=fr_lha, fr_rsp=fr_rsp,
             pc1_lha=pc1_lha, pc1_rsp=pc1_rsp,
             entropy_interp=ent_interp)
    df.to_csv(OUTDIR / f"session_{sn}_summary.csv", index=False)

    bin_t = bin_centers
    fig, axes = plt.subplots(5, 1, figsize=(15, 12), sharex=True)
    ax = axes[0]
    ax.plot(ent_t, ent_v, color='lightgray', lw=0.7, label='raw')
    ax.plot(ent_t, smoothed, color='black', lw=1.3, label='smoothed')
    ax.scatter(ent_t[peaks], smoothed[peaks], color='red', s=35, zorder=5,
               label=f'peaks ({len(peaks)})')
    ax.scatter(ent_t[troughs], smoothed[troughs], color='blue', s=35, zorder=5,
               label=f'troughs ({len(troughs)})')
    ax.set_ylabel('entropy (bits)')
    ax.set_title(f'S{sn} ({state}, {phase}) — entropy with inflections')
    ax.legend(loc='upper right', fontsize=8); ax.grid(alpha=0.3)
    ax = axes[1]
    ax.plot(bin_t[1:1 + len(sp_lha)], sp_lha, color='C3', lw=0.7, label='LHA speed')
    ax.set_ylabel('||dN/dt||'); ax.legend(loc='upper right', fontsize=8); ax.grid(alpha=0.3)
    ax = axes[2]
    ax.plot(bin_t[1:1 + len(sp_rsp)], sp_rsp, color='C2', lw=0.7, label='RSP speed')
    ax.set_ylabel('||dN/dt||'); ax.legend(loc='upper right', fontsize=8); ax.grid(alpha=0.3)
    ax = axes[3]
    ax.plot(bin_t[1:1 + len(cv_lha)], cv_lha, color='C3', lw=0.7, label='LHA curv')
    ax.plot(bin_t[1:1 + len(cv_rsp)], cv_rsp, color='C2', lw=0.7, alpha=0.7, label='RSP curv')
    ax.set_ylabel('1-cos(theta)'); ax.legend(loc='upper right', fontsize=8); ax.grid(alpha=0.3)
    ax = axes[4]
    comp_classes = behav['compartment']['classes']
    palette = {'Home': '#2ca02c', 'Ladder': '#ff7f0e',
               'Arena': '#1f77b4', 'AtPot': '#d62728'}
    comp_vals = behav['compartment']['values']
    code = np.array([comp_classes.index(v) for v in comp_vals])
    ax.scatter(bin_t[:len(code)], code, c=[palette[v] for v in comp_vals],
               s=2, alpha=0.6)
    ax.set_yticks(range(len(comp_classes)))
    ax.set_yticklabels(comp_classes)
    ax.set_xlabel('time (s)'); ax.set_ylabel('compartment')
    handles = [mpatches.Patch(color=palette[c], label=c) for c in comp_classes]
    ax.legend(handles=handles, loc='upper right', fontsize=8); ax.grid(alpha=0.3)
    for axx in axes:
        for t in ent_t[peaks]:
            axx.axvline(t, color='red', alpha=0.15, lw=0.8)
        for t in ent_t[troughs]:
            axx.axvline(t, color='blue', alpha=0.15, lw=0.8)
    plt.tight_layout()
    plt.savefig(FIGDIR / f"session_{sn}_diagnostic.png", dpi=110)
    plt.close()

    log(f"  done in {(time.time() - t0):.1f}s  rows={len(df)}")
    return df


def main():
    log_path = OUTDIR / "batch_log.txt"
    log_lines = []
    def log(msg):
        print(msg)
        log_lines.append(msg)

    sess_list = list_sessions()
    log(f"Found {len(sess_list)} valid sessions.")
    for s in sess_list:
        log(f"  S{s['session']}  {s['state']}  {s['phase']}")

    all_rows = []
    failures = []
    for s in sess_list:
        try:
            df = run_session(s, log)
            all_rows.append(df)
        except Exception as e:
            log(f"  !! S{s['session']} FAILED: {e}")
            log(traceback.format_exc())
            failures.append(s['session'])

    if all_rows:
        combo = pd.concat(all_rows, ignore_index=True)
        combo.to_csv(OUTDIR / "all_sessions_summary.csv", index=False)
        log(f"\nWrote all_sessions_summary.csv ({len(combo)} rows, "
            f"{combo['session'].nunique()} sessions)")

    log(f"\nFailures: {failures}")
    with open(log_path, 'w') as f:
        f.write("\n".join(log_lines))
    log(f"Wrote {log_path}")


if __name__ == '__main__':
    main()
