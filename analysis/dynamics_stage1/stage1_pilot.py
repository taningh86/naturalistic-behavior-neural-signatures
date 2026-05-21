"""
Stage 1 dynamics pilot on session 3.

Run order (8 steps from plan):
  1. Load neural (ACA, LHA) and behavior
  2. Behavioral entropy series + inflections
  3. Define phases (rising / falling / peak / trough)
  4. Trajectory speed (per region)
  5. Trajectory curvature (per region)
  6. Per-phase summary rows
  7. Save artifacts
  8. Diagnostic figure

STOP after this pilot. Inspect diagnostic figure before batching.

Usage: python stage1_pilot.py [session_num]
"""
import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "analysis" / "dynamics_stage1"))
sys.path.insert(0, str(REPO / "analysis" / "cycles_partD"))

from stage1_lib import (
    BIN_S, K_PCS, list_sessions,
    load_neural, load_behavior, filter_behavior,
    load_zones_and_velocity, compute_entropy_series, detect_inflections,
    define_phases, compute_speed, compute_curvature,
    phase_summary_row, interp_entropy_to_bins,
)

OUTDIR = REPO / "data" / "dynamics_stage1"
FIGDIR = REPO / "figures" / "dynamics_stage1"
OUTDIR.mkdir(parents=True, exist_ok=True)
FIGDIR.mkdir(parents=True, exist_ok=True)


def main():
    session_num = int(sys.argv[1]) if len(sys.argv) > 1 else 3

    print(f"\n=== Stage 1 dynamics pilot: S{session_num} ===")

    sessions = list_sessions()
    sess_meta = next((s for s in sessions if s['session'] == session_num), None)
    if sess_meta is None:
        print(f"Session {session_num} has no usable behavior+sorted; aborting.")
        return

    print(f"  state={sess_meta['state']}  phase={sess_meta['phase']}")
    print(f"  behavior xlsx: {sess_meta['behavior']}")

    # ---------- Step 1: Load neural + behavior ----------
    print("\n[1] Loading neural (ACA, LHA) and behavior...")
    aca, bin_centers_aca, n_aca = load_neural(session_num, 'ACA')
    lha, bin_centers_lha, n_lha = load_neural(session_num, 'LHA')
    n_bins = min(aca.shape[0], lha.shape[0])
    aca = aca[:n_bins]
    lha = lha[:n_bins]
    bin_centers = bin_centers_aca[:n_bins]
    print(f"  ACA: {aca.shape}  ({n_aca} units)")
    print(f"  LHA: {lha.shape}  ({n_lha} units)")
    print(f"  total {n_bins} bins = {n_bins * BIN_S:.1f} s")

    behav = filter_behavior(load_behavior(session_num, bin_centers))
    print(f"  behavior keys (post-lever-filter): {len(behav)}")

    # ---------- Step 2: Behavioral entropy + inflections ----------
    print("\n[2] Computing behavioral entropy series and inflections...")
    time_vals, zones, vel = load_zones_and_velocity(sess_meta['behavior'])
    ent_t, ent_v, vel_m = compute_entropy_series(zones, time_vals, vel)
    peaks, troughs, smoothed = detect_inflections(ent_t, ent_v)
    print(f"  entropy series: {len(ent_t)} timepoints "
          f"(t={ent_t[0]:.1f}-{ent_t[-1]:.1f} s)")
    print(f"  peaks={len(peaks)}  troughs={len(troughs)}")

    # ---------- Step 3: Phase definition ----------
    print("\n[3] Defining phases...")
    phases = define_phases(ent_t, peaks, troughs, n_bins)
    type_counts = pd.Series([p['phase_type'] for p in phases]).value_counts().to_dict()
    print(f"  total phases: {len(phases)}")
    print(f"  by type: {type_counts}")
    if len(phases):
        durs = [p['duration_s'] for p in phases]
        print(f"  duration: median {np.median(durs):.1f}s, "
              f"min {np.min(durs):.1f}s, max {np.max(durs):.1f}s")

    # ---------- Step 4: Trajectory speed ----------
    print("\n[4] Trajectory speed (full neural state space)...")
    sp_aca = compute_speed(aca)
    sp_lha = compute_speed(lha)
    print(f"  ACA speed: mean {sp_aca.mean():.3f}  std {sp_aca.std():.3f}")
    print(f"  LHA speed: mean {sp_lha.mean():.3f}  std {sp_lha.std():.3f}")

    # ---------- Step 5: Trajectory curvature ----------
    print("\n[5] Trajectory curvature (1 - cos(theta))...")
    cv_aca = compute_curvature(aca)
    cv_lha = compute_curvature(lha)
    print(f"  ACA curv: mean {cv_aca.mean():.3f}  std {cv_aca.std():.3f}")
    print(f"  LHA curv: mean {cv_lha.mean():.3f}  std {cv_lha.std():.3f}")

    # ---------- Step 6: Per-phase summary ----------
    print("\n[6] Computing per-phase summary rows...")
    fr_aca = aca.mean(axis=1)
    fr_lha = lha.mean(axis=1)
    pca_aca = PCA(n_components=1).fit_transform(aca).ravel()
    pca_lha = PCA(n_components=1).fit_transform(lha).ravel()
    ent_interp = interp_entropy_to_bins(ent_t, smoothed, bin_centers)

    region_speeds = {'ACA': sp_aca, 'LHA': sp_lha}
    region_curvs = {'ACA': cv_aca, 'LHA': cv_lha}
    region_fr = {'ACA': fr_aca, 'LHA': fr_lha}
    region_pc1 = {'ACA': pca_aca, 'LHA': pca_lha}

    rows = []
    for ph in phases:
        row = phase_summary_row(ph, None, behav,
                                 region_speeds, region_curvs,
                                 region_fr, region_pc1, ent_interp)
        if row is not None:
            row['session'] = session_num
            row['state'] = sess_meta['state']
            row['exp_phase'] = sess_meta['phase']
            rows.append(row)
    df = pd.DataFrame(rows)
    print(f"  {len(df)} phase rows")
    print(f"  phase_type counts in df: {df['phase_type'].value_counts().to_dict()}")

    # ---------- Step 7: Save artifacts ----------
    print("\n[7] Saving artifacts...")
    np.save(OUTDIR / f"session_{session_num}_speed.npy",
            dict(ACA=sp_aca, LHA=sp_lha), allow_pickle=True)
    np.save(OUTDIR / f"session_{session_num}_curvature.npy",
            dict(ACA=cv_aca, LHA=cv_lha), allow_pickle=True)
    with open(OUTDIR / f"session_{session_num}_phases.json", 'w') as f:
        json.dump([{k: (v if not isinstance(v, np.generic) else v.item())
                    for k, v in p.items()} for p in phases], f, indent=2)
    np.savez(OUTDIR / f"session_{session_num}_phase_data.npz",
             ent_t=ent_t, ent_v=ent_v, ent_smoothed=smoothed,
             peaks=peaks, troughs=troughs,
             bin_centers=bin_centers,
             fr_aca=fr_aca, fr_lha=fr_lha,
             pc1_aca=pca_aca, pc1_lha=pca_lha,
             entropy_interp=ent_interp)
    df.to_csv(OUTDIR / f"session_{session_num}_summary.csv", index=False)
    print(f"  wrote 5 files in {OUTDIR}")

    # ---------- Step 8: Diagnostic figure ----------
    print("\n[8] Diagnostic figure...")
    bin_t = bin_centers
    fig, axes = plt.subplots(5, 1, figsize=(15, 12), sharex=True)

    ax = axes[0]
    ax.plot(ent_t, ent_v, color='lightgray', lw=0.7, label='raw entropy')
    ax.plot(ent_t, smoothed, color='black', lw=1.4, label='smoothed')
    ax.scatter(ent_t[peaks], smoothed[peaks], color='red', s=40, zorder=5,
               label=f'peaks ({len(peaks)})')
    ax.scatter(ent_t[troughs], smoothed[troughs], color='blue', s=40, zorder=5,
               label=f'troughs ({len(troughs)})')
    ax.set_ylabel('entropy (bits)')
    ax.set_title(f'S{session_num} ({sess_meta["state"]}, {sess_meta["phase"]}) — '
                 f'entropy with inflections')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(bin_t[1:1 + len(sp_aca)], sp_aca, color='C0', lw=0.7, label='ACA speed')
    ax.set_ylabel('||dN/dt||')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(bin_t[1:1 + len(sp_lha)], sp_lha, color='C3', lw=0.7, label='LHA speed')
    ax.set_ylabel('||dN/dt||')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[3]
    ax.plot(bin_t[1:1 + len(cv_aca)], cv_aca, color='C0', lw=0.7, label='ACA curv')
    ax.plot(bin_t[1:1 + len(cv_lha)], cv_lha, color='C3', lw=0.7, alpha=0.7,
            label='LHA curv')
    ax.set_ylabel('1-cos(theta)')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(alpha=0.3)

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
    ax.set_xlabel('time (s)')
    ax.set_ylabel('compartment')
    handles = [mpatches.Patch(color=palette[c], label=c) for c in comp_classes]
    ax.legend(handles=handles, loc='upper right', fontsize=8)
    ax.grid(alpha=0.3)

    # Vertical lines for inflections on all panels
    for axx in axes:
        for t in ent_t[peaks]:
            axx.axvline(t, color='red', alpha=0.15, lw=0.8)
        for t in ent_t[troughs]:
            axx.axvline(t, color='blue', alpha=0.15, lw=0.8)

    plt.tight_layout()
    figpath = FIGDIR / f"session_{session_num}_diagnostic.png"
    plt.savefig(figpath, dpi=110)
    plt.close()
    print(f"  wrote {figpath}")

    # ---------- Print phase summary table head ----------
    print("\n[summary head]")
    show_cols = ['phase_type', 'duration_s', 'mean_speed_ACA', 'mean_speed_LHA',
                 'mean_curv_ACA', 'mean_curv_LHA', 'mean_entropy',
                 'dominant_compartment', 'dominant_action']
    show_cols = [c for c in show_cols if c in df.columns]
    print(df[show_cols].head(10).to_string(index=False))

    print("\nDone. STOP per plan — inspect diagnostic before batching.")


if __name__ == '__main__':
    main()
