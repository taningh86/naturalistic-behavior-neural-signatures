"""05 — Compute per-bin state posteriors and Viterbi decode for each session.

Output: data/HMM/posteriors/session_{N}.csv
  Columns: bin, time_s, p_state_0..p_state_{N-1}, viterbi
"""
import pickle
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _utils import load_config, session_list, ensure_dir, REPO_ROOT


def main():
    cfg = load_config()
    prepared_dir = REPO_ROOT / cfg["out_dirs"]["prepared"]
    out_dir = ensure_dir(REPO_ROOT / cfg["out_dirs"]["posteriors"])

    model_path = REPO_ROOT / cfg["out_dirs"]["final_model"]
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)
    hmm = bundle["hmm"]
    N = bundle["N"]
    target_bin_ms = cfg["target_bin_ms"]
    bin_s = target_bin_ms / 1000.0
    print(f"Loaded model: N={N}, D={bundle['D']}")

    sess = session_list(cfg)
    for session_num, state in sess:
        p = prepared_dir / f"session_{session_num}.npz"
        if not p.exists():
            print(f"  SKIP S{session_num}: missing {p}")
            continue
        z = np.load(p, allow_pickle=True)
        X = z["X"]
        T = X.shape[0]
        post = hmm.expected_states(X)[0]  # (T, N)
        viterbi = hmm.most_likely_states(X)
        time_s = np.arange(T) * bin_s

        df = pd.DataFrame(
            {
                "bin": np.arange(T),
                "time_s": time_s,
                **{f"p_state_{k}": post[:, k] for k in range(N)},
                "viterbi": viterbi,
            }
        )
        out_path = out_dir / f"session_{session_num}.csv"
        df.to_csv(out_path, index=False)
        print(f"  S{session_num} ({state}): T={T} → {out_path.name}")

    print(f"\nDone. Posteriors in {out_dir}")


if __name__ == "__main__":
    main()
