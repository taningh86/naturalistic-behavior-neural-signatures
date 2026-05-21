"""Driver: run GLM-HMM fit-one mode for all 4 (region, phase) × K∈{6, 8}."""
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PY = r"C:\Users\Gregg\anaconda3\envs\si_env\python.exe"
PIPELINE = REPO / "scripts" / "HMM_glm" / "fit_pipeline.py"

CELLS = [("ACA", "exploration"),
          ("ACA", "foraging"),
          ("LHA", "exploration"),
          ("LHA", "foraging")]
K_VALUES = [6]
N_RESTARTS = 3
NUM_ITERS = 100
BIN_S = 0.05
SEED = 20260513

import os
env = os.environ.copy()
env["PYTHONIOENCODING"] = "utf-8"

t_master = time.time()
total = len(CELLS) * len(K_VALUES)
done = 0
for region, phase in CELLS:
    for K in K_VALUES:
        done += 1
        print(f"\n[{done}/{total}] {region} {phase} K={K} starting at "
              f"+{(time.time()-t_master)/60:.0f} min", flush=True)
        t0 = time.time()
        result = subprocess.run(
            [PY, "-u", str(PIPELINE),
             "--mode", "fit-one",
             "--region", region,
             "--phase", phase,
             "--K", str(K),
             "--n-restarts", str(N_RESTARTS),
             "--num-iters", str(NUM_ITERS),
             "--bin-s", str(BIN_S),
             "--seed", str(SEED)],
            cwd=str(REPO), env=env,
        )
        dt = time.time() - t0
        if result.returncode != 0:
            print(f"  FAILED {region} {phase} K={K} (code {result.returncode}) "
                  f"after {dt:.0f}s", flush=True)
        else:
            print(f"  OK {region} {phase} K={K} in {dt/60:.1f} min", flush=True)
print(f"\nAll done in {(time.time()-t_master)/60:.1f} min")
