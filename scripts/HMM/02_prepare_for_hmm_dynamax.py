"""02 (dynamax) — Prepare HMM observation matrices with FACTORIZED groups.

Reads data/HMM/binned/session_{N}.npz and writes per-session npz files with three
separate arrays (no concatenation, no jitter):

  X_continuous  (T, 2)   — speed_z, distance_to_pot_z (pooled-z-scored)
  X_zone        (T,)     — integer zone index in [0, K_zone)
  X_events      (T, 7)   — 0/1 binary events

Plus a session_metadata.csv table summarising the prepared sessions and a
zone_label_mapping.csv for downstream interpretation.
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _utils import load_config, session_list, ensure_dir, REPO_ROOT


def main():
    cfg = load_config()
    binned_dir = REPO_ROOT / cfg["out_dirs"]["binned"]
    out_dir = ensure_dir(REPO_ROOT / cfg["dynamax_dirs"]["prepared"])
    np.random.seed(cfg["random_seed"])

    sess = session_list(cfg)

    # First pass: pool continuous values for z-score stats.
    speed_pool, dist_pool = [], []
    sessions_data = []
    for session_num, state in sess:
        path = binned_dir / f"session_{session_num}.npz"
        if not path.exists():
            print(f"  SKIP S{session_num}: no binned file at {path}")
            continue
        z = np.load(path, allow_pickle=True)
        sessions_data.append((session_num, state, z))
        speed_pool.append(z["speed"])
        dist_pool.append(z["distance_to_pot"])

    if not sessions_data:
        raise SystemExit("No binned sessions found; run 01 first.")

    speed_all = np.concatenate(speed_pool)
    dist_all = np.concatenate(dist_pool)
    speed_mean, speed_std = float(speed_all.mean()), float(speed_all.std() + 1e-12)
    dist_mean, dist_std = float(dist_all.mean()), float(dist_all.std() + 1e-12)
    print(f"Pooled stats: speed mean={speed_mean:.3f} std={speed_std:.3f};  "
          f"distance_to_pot mean={dist_mean:.3f} std={dist_std:.3f}")

    # Verify zone label and event name consistency across sessions.
    zone_labels = list(sessions_data[0][2]["zone_labels"])
    event_names = list(sessions_data[0][2]["event_names"])
    for sn, st, z in sessions_data:
        if list(z["zone_labels"]) != zone_labels:
            raise RuntimeError(f"Zone labels mismatch in S{sn}")
        if list(z["event_names"]) != event_names:
            raise RuntimeError(f"Event names mismatch in S{sn}")

    K_zone = len(zone_labels)
    n_events = len(event_names)
    print(f"Zone labels (K_zone={K_zone}): {zone_labels}")
    print(f"Event names (n_events={n_events}): {event_names}")

    # Save zone-label mapping for downstream.
    zmap = pd.DataFrame({"zone_index": np.arange(K_zone), "zone_label": zone_labels})
    zmap_path = out_dir / "zone_label_mapping.csv"
    zmap.to_csv(zmap_path, index=False)
    print(f"Zone label mapping → {zmap_path}")

    meta_rows = []
    for session_num, state, z in sessions_data:
        T = int(z["meta"].item()["n_bins"])

        speed_z = (z["speed"] - speed_mean) / speed_std            # (T,)
        dist_z = (z["distance_to_pot"] - dist_mean) / dist_std     # (T,)
        X_cont = np.stack([speed_z, dist_z], axis=1).astype(np.float64)  # (T, 2)

        X_zone = z["zone"].astype(np.int64)                        # (T,)
        X_events = z["events"].astype(np.float64)                  # (T, 7) — keep as float for matmul

        out_path = out_dir / f"session_{session_num}.npz"
        np.savez(
            out_path,
            X_continuous=X_cont,
            X_zone=X_zone,
            X_events=X_events,
            continuous_names=np.array(["speed_z", "distance_to_pot_z"]),
            zone_labels=np.array(zone_labels),
            event_names=np.array(event_names),
            speed_mean=speed_mean,
            speed_std=speed_std,
            dist_mean=dist_mean,
            dist_std=dist_std,
            session_num=session_num,
            state=state,
            trial_time=z["trial_time"],
        )
        print(f"  S{session_num} ({state}): X_cont {X_cont.shape}, "
              f"X_zone {X_zone.shape}, X_events {X_events.shape} → {out_path.name}")
        meta_rows.append(
            dict(
                session_num=session_num,
                state=state,
                n_bins=T,
                D_cont=X_cont.shape[1],
                K_zone=K_zone,
                n_events=n_events,
            )
        )

    meta_df = pd.DataFrame(meta_rows)
    meta_csv = out_dir / "session_metadata.csv"
    meta_df.to_csv(meta_csv, index=False)
    print(f"\nSession metadata → {meta_csv}")
    print(meta_df.to_string(index=False))
    print("Done.")


if __name__ == "__main__":
    main()
