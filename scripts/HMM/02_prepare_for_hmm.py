"""02 — Prepare HMM observation matrices.

Loads binned sessions, packages observations for ssm.HMM.

WORKAROUND for mixed emissions: ssm does not natively support a single HMM with
factorized Gaussian + Bernoulli + Categorical emissions. We use the standard
behavioral-HMM workaround (cf. Wiltschko 2015 / MoSeq) — concatenate all
features into a single Gaussian observation vector:

  - Continuous (speed, distance_to_pot): z-scored across pooled dataset.
  - Bernoulli events: kept as 0/1 (modeled as Gaussian on {0,1}).
  - Zone categorical: one-hot encoded (each class becomes its own 0/1 column).

The means of the Gaussian fit on 0/1 columns approximate Bernoulli probabilities,
and likelihoods are tractable. This is theoretically suboptimal but practically
robust and is the convention for behavioral-state HMMs in the literature.

To swap in a proper factorized emission later, replace the assembled X array
with three separate matrices and write a custom ssm Observations subclass.

Output:
  data/HMM/prepared/session_{N}.npz with arrays
    X            (T, D)   — z-scored continuous + binary events + one-hot zones
    feature_names list[str] — col names in X
    state, session_num
  data/HMM/prepared/session_metadata.csv — table of all sessions
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
    out_dir = ensure_dir(REPO_ROOT / cfg["out_dirs"]["prepared"])
    np.random.seed(cfg["random_seed"])

    sess = session_list(cfg)

    # First pass: pool continuous values for z-score stats.
    speed_pool = []
    dist_pool = []
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
    speed_mean, speed_std = float(speed_all.mean()), float(speed_all.std())
    dist_mean, dist_std = float(dist_all.mean()), float(dist_all.std())
    print(f"Pooled stats: speed mean={speed_mean:.3f} std={speed_std:.3f};  "
          f"distance_to_pot mean={dist_mean:.3f} std={dist_std:.3f}")

    # Determine union of zone labels (should be identical across sessions, but be safe)
    zone_labels = list(sessions_data[0][2]["zone_labels"])
    for sn, st, z in sessions_data:
        zl = list(z["zone_labels"])
        if zl != zone_labels:
            raise RuntimeError(f"Zone labels mismatch in S{sn}: {zl} vs {zone_labels}")
    event_names = list(sessions_data[0][2]["event_names"])
    for sn, st, z in sessions_data:
        en = list(z["event_names"])
        if en != event_names:
            raise RuntimeError(f"Event names mismatch in S{sn}: {en} vs {event_names}")

    print(f"Zone labels: {zone_labels}")
    print(f"Event names: {event_names}")

    n_zones = len(zone_labels)
    n_events = len(event_names)
    feature_names = ["speed_z", "distance_to_pot_z"]
    feature_names += [f"event_{n}" for n in event_names]
    feature_names += [f"zone_{n}" for n in zone_labels]

    binary_jitter = float(cfg.get("binary_jitter_sigma", 0.05))
    cont_jitter = float(cfg.get("continuous_jitter_sigma", 0.0))
    rng = np.random.default_rng(cfg["random_seed"])

    meta_rows = []
    for session_num, state, z in sessions_data:
        T = int(z["meta"].item()["n_bins"])
        speed_z = (z["speed"] - speed_mean) / speed_std
        dist_z = (z["distance_to_pot"] - dist_mean) / dist_std

        # one-hot zone
        zone = z["zone"]
        zone_oh = np.zeros((T, n_zones), dtype=np.float64)
        zone_oh[np.arange(T), zone] = 1.0

        events = z["events"].astype(np.float64)

        # Add small Gaussian jitter to binary/one-hot columns to prevent
        # variance collapse in Gaussian-on-binary HMM (MoSeq-style workaround).
        if binary_jitter > 0:
            events = events + rng.normal(0, binary_jitter, size=events.shape)
            zone_oh = zone_oh + rng.normal(0, binary_jitter, size=zone_oh.shape)
        if cont_jitter > 0:
            speed_z = speed_z + rng.normal(0, cont_jitter, size=speed_z.shape)
            dist_z = dist_z + rng.normal(0, cont_jitter, size=dist_z.shape)

        X = np.concatenate(
            [
                speed_z[:, None],
                dist_z[:, None],
                events,
                zone_oh,
            ],
            axis=1,
        )

        out_path = out_dir / f"session_{session_num}.npz"
        np.savez(
            out_path,
            X=X,
            feature_names=np.array(feature_names),
            state=state,
            session_num=session_num,
            zone_labels=np.array(zone_labels),
            event_names=np.array(event_names),
            speed_mean=speed_mean,
            speed_std=speed_std,
            dist_mean=dist_mean,
            dist_std=dist_std,
            trial_time=z["trial_time"],
        )
        print(f"  S{session_num} ({state}): X shape {X.shape} → {out_path.name}")
        meta_rows.append(
            dict(
                session_num=session_num,
                state=state,
                n_bins=T,
                D=X.shape[1],
            )
        )

    meta_df = pd.DataFrame(meta_rows)
    meta_csv = out_dir / "session_metadata.csv"
    meta_df.to_csv(meta_csv, index=False)
    print(f"\nSession metadata → {meta_csv}")
    print(meta_df.to_string(index=False))
    print(f"\nFeature dimension D = {len(feature_names)}")
    print(f"Done.")


if __name__ == "__main__":
    main()
