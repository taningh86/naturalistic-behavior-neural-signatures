"""
Stage 1 dynamics batch: run the pilot pipeline on every valid session.

Outputs per-session:
  data/dynamics_stage1/session_{N}_speed.npy
  data/dynamics_stage1/session_{N}_curvature.npy
  data/dynamics_stage1/session_{N}_phases.json
  data/dynamics_stage1/session_{N}_phase_data.npz
  data/dynamics_stage1/session_{N}_summary.csv
  figures/dynamics_stage1/session_{N}_diagnostic.png

Plus aggregated:
  data/dynamics_stage1/all_sessions_summary.csv
  data/dynamics_stage1/batch_log.txt

Lever-zone columns are filtered out at behavior load time. Binary behaviors are
the fixed five: feeding, digging_sand, incomplete_home_returns,
quick_one_loop_at_home, transition_wall_exploration. Absent behaviors emit
frac=0.0 so columns remain aligned across sessions.
"""
import sys
import time
import traceback
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "analysis" / "dynamics_stage1"))

from stage1_lib import (
    BIN_S, list_sessions,
    load_neural, load_behavior, filter_behavior,
    load_zones_and_velocity, compute_entropy_series, detect_inflections,
    define_phases, compute_speed, compute_curvature,
    phase_summary_row, interp_entropy_to_bins,
)

OUTDIR = REPO / "data" / "dynamics_stage1"
FIGDIR = REPO / "figures" / "dynamics_stage1"
OUTDIR.mkdir(parents=True, exist_ok=True)
FIGDIR.mkdir(parents=True, exist_ok=True)


def run_session(sess_meta, log):
    sn = sess_meta['session']
    state, phase = sess_meta['state'], sess_meta['phase']
    log(f"\n=== S{sn}  {state} / {phase} ===")
    t0 = time.time()

    aca, bin_centers, _ = load_neural(sn, 'ACA')
    lha, _, _ = load_neural(sn, 'LHA')
    n_bins = min(aca.shape[0], lha.shape[0])
    aca, lha = aca[:n_bins], lha[:n_bins]
    bin_centers = bin_centers[:n_bins]
    log(f"  neural: ACA {aca.shape}  LHA {lha.shape}  ({n_bins} bins)")

    behav = filter_behavior(load_behavior(sn, bin_centers))
    log(f"  behavior keys (post-lever-filter): {len(behav)}")

    time_vals, zones, vel = load_zones_and_velocity(sess_meta['behavior'])
    ent_t, ent_v, _ = compute_entropy_series(zones, time_vals, vel)
    peaks, troughs, smoothed = detect_inflections(ent_t, ent_v)
    log(f"  entropy pts={len(ent_t)}  peaks={len(peaks)}  troughs={len(troughs)}")

    phases = define_phases(ent_t, peaks, troughs, n_bins)
    log(f"  phases: {len(phases)}")

    sp_aca = compute_speed(aca)
    sp_lha = compute_speed(lha)
    cv_aca = compute_curvature(aca)
    cv_lha = compute_curvature(lha)
    log(f"  ACA spd {sp_aca.mean():.3f}+/-{sp_aca.std():.3f}, "
        f"LHA spd {sp_lha.mean():.3f}+/-{sp_lha.std():.3f}")

    fr_aca = aca.mean(axis=1)
    fr_lha = lha.mean(axis=1)
    pc1_aca = PCA(n_components=1).fit_transform(aca).ravel()
    pc1_lha = PCA(n_components=1).fit_transform(lha).ravel()
    ent_interp = interp_entropy_to_bins(ent_t, smoothed, bin_centers)

    rs = {'ACA': sp_aca, 'LHA': sp_lha}
    rc = {'ACA': cv_aca, 'LHA': cv_lha}
    rfr = {'ACA': fr_aca, 'LHA': fr_lha}
    rpc = {'ACA': pc1_aca, 'LHA': pc1_lha}

    rows = []
    for ph in phases:
        row = phase_summary_row(ph, None, behav, rs, rc, rfr, rpc, ent_interp)
        if row is not None:
            row['session'] = sn
            row['state'] = state
            row['exp_phase'] = phase
            rows.append(row)
    df = pd.DataFrame(rows)

    np.save(OUTDIR / f"session_{sn}_speed.npy",
            dict(ACA=sp_aca, LHA=sp_lha), allow_pickle=True)
    np.save(OUTDIR / f"session_{sn}_curvature.npy",
            dict(ACA=cv_aca, LHA=cv_lha), allow_pickle=True)
    with open(OUTDIR / f"session_{sn}_phases.json", 'w') as f:
        json.dump([{k: (v.item() if isinstance(v, np.generic) else v)
                    for k, v in p.items()} for p in phases], f, indent=2)
    np.savez(OUTDIR / f"session_{sn}_phase_data.npz",
             ent_t=ent_t, ent_v=ent_v, ent_smoothed=smoothed,
             peaks=peaks, troughs=troughs, bin_centers=bin_centers,
             fr_aca=fr_aca, fr_lha=fr_lha,
             pc1_aca=pc1_aca, pc1_lha=pc1_lha,
             entropy_interp=ent_interp)
    df.to_csv(OUTDIR / f"session_{sn}_summary.csv", index=False)

    # Diagnostic figure
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
    ax.plot(bin_t[1:1 + len(sp_aca)], sp_aca, color='C0', lw=0.7, label='ACA speed')
    ax.set_ylabel('||dN/dt||'); ax.legend(loc='upper right', fontsize=8); ax.grid(alpha=0.3)
    ax = axes[2]
    ax.plot(bin_t[1:1 + len(sp_lha)], sp_lha, color='C3', lw=0.7, label='LHA speed')
    ax.set_ylabel('||dN/dt||'); ax.legend(loc='upper right', fontsize=8); ax.grid(alpha=0.3)
    ax = axes[3]
    ax.plot(bin_t[1:1 + len(cv_aca)], cv_aca, color='C0', lw=0.7, label='ACA curv')
    ax.plot(bin_t[1:1 + len(cv_lha)], cv_lha, color='C3', lw=0.7, alpha=0.7, label='LHA curv')
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
