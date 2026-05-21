"""05 (dynamax) — Per-bin smoothed posteriors + Viterbi for each session.

Loads data/HMM/final_model_dynamax.npz; for each prepared session computes
forward-backward smoothed posteriors and Viterbi-decoded states. Writes a CSV
per session with columns: bin, time_s, p_state_0..N-1, viterbi.
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

import jax

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _utils import load_config, ensure_dir, REPO_ROOT
import mixed_hmm as mh


def load_final_params(path: Path) -> mh.MixedHMMParams:
    z = np.load(path, allow_pickle=True)
    return mh.MixedHMMParams(
        pi=np.asarray(z["pi"], dtype=np.float64),
        A=np.asarray(z["A"], dtype=np.float64),
        mu=np.asarray(z["mu"], dtype=np.float64),
        sigma=np.asarray(z["sigma"], dtype=np.float64),
        p_zone=np.asarray(z["p_zone"], dtype=np.float64),
        q_events=np.asarray(z["q_events"], dtype=np.float64),
    )


def main():
    cfg = load_config()
    print(f"JAX backend: {jax.default_backend()}  devices: {jax.devices()}")
    prepared_dir = REPO_ROOT / cfg["dynamax_dirs"]["prepared"]
    final_npz = REPO_ROOT / cfg["dynamax_dirs"]["final_model"]
    out_dir = ensure_dir(REPO_ROOT / cfg["dynamax_dirs"]["posteriors"])

    params = load_final_params(final_npz)
    N = params.K
    print(f"Loaded final model: N={N}")

    fed = cfg["sessions"]["fed"]
    fasted = cfg["sessions"]["fasted"]
    all_sessions = fed + fasted

    state_cols = [f"p_state_{k}" for k in range(N)]

    for sn in all_sessions:
        z = np.load(prepared_dir / f"session_{sn}.npz", allow_pickle=True)
        x_cont = np.asarray(z["X_continuous"], dtype=np.float64)
        x_zone = np.asarray(z["X_zone"], dtype=np.int64)
        x_events = np.asarray(z["X_events"], dtype=np.float64)
        T = x_cont.shape[0]

        gamma, ll = mh.smoothed_posteriors(params, x_cont, x_zone, x_events)
        viterbi = mh.viterbi_states(params, x_cont, x_zone, x_events)

        time_s = np.asarray(z["trial_time"], dtype=np.float64)
        if time_s.shape[0] != T:
            time_s = np.arange(T) * 0.480

        df = pd.DataFrame(gamma, columns=state_cols)
        df.insert(0, "viterbi", viterbi)
        df.insert(0, "time_s", time_s)
        df.insert(0, "bin", np.arange(T))
        out_path = out_dir / f"session_{sn}.csv"
        df.to_csv(out_path, index=False)
        print(f"  S{sn}: T={T}  ll={ll:.2f}  "
              f"viterbi state distribution: {np.bincount(viterbi, minlength=N)} "
              f"→ {out_path.name}")


if __name__ == "__main__":
    main()
