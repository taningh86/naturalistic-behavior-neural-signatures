"""ssm Poisson-GLM-HMM wrapper.

Wraps `ssm.HMM` with Poisson observations and `inputdriven` transitions.

The transition GLM has parameters:
  - `log_Ps`: (K, K) base log-transition matrix
  - `Ws`: (K, K, M) input-dependent perturbation
The effective transition prob from i to j given input u_t is
  P(z_t=j | z_{t-1}=i, u_t) ∝ exp(log_Ps[i, j] + Ws[i, j, :] @ u_t)
which is the softmax over j.
"""
from __future__ import annotations

import numpy as np
import pickle
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from ssm import HMM


def fit_one(counts_list: list[np.ndarray],
            inputs_list: list[np.ndarray],
            K: int, D: int, M: int,
            num_iters: int = 100,
            tolerance: float = 1e-4,
            init_method: str = "random",
            seed: int = 0,
            verbose: bool = False) -> tuple[HMM, list[float]]:
    """Fit a single random-init Poisson-GLM-HMM.

    counts_list: list of (T_n, D) int arrays (one per session)
    inputs_list: list of (T_n, M) one-hot input arrays
    Returns (fitted_model, list_of_train_LL_per_iter).
    """
    rng = np.random.RandomState(seed)
    np.random.seed(seed)
    hmm = HMM(K, D, M=M,
              observations="poisson",
              transitions="inputdriven")
    # ssm.HMM.fit returns elbo / ll history depending on method
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lls = hmm.fit(counts_list, inputs=inputs_list,
                      method="em", num_iters=num_iters,
                      tolerance=tolerance, verbose=0 if not verbose else 2)
    return hmm, list(lls)


def evaluate_ll(hmm: HMM, counts_list: list[np.ndarray],
                  inputs_list: list[np.ndarray]) -> tuple[float, int]:
    """Compute total log-likelihood and total bin count across the given dataset."""
    total_ll = 0.0
    total_bins = 0
    for c, u in zip(counts_list, inputs_list):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            total_ll += float(hmm.log_likelihood(c, input=u))
        total_bins += int(c.shape[0])
    return total_ll, total_bins


def state_assignments(hmm: HMM, counts: np.ndarray, inp: np.ndarray) -> np.ndarray:
    """Viterbi most-likely state path."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        z = hmm.most_likely_states(counts, input=inp)
    return np.asarray(z, dtype=np.int64)


def effective_transitions(hmm: HMM, one_hot_dim: int = 3) -> np.ndarray:
    """For each one-hot input u (dim M), return the (K, K) softmax-normalized
    transition matrix. Returns (M, K, K) array.

    Uses ssm's `transition_matrices` to compute properly under the `inputdriven`
    parameterization (where Ws is (K, M) — per-target-state input bias —
    rather than the full (K, K, M) form). The resulting per-input (K, K)
    matrix is the effective Markov chain for that metabolic state.
    """
    K = hmm.K
    M = one_hot_dim
    out = np.zeros((M, K, K), dtype=np.float64)
    D = hmm.D
    # ssm requires data + input of matching shape to compute transitions
    fake_obs = np.zeros((2, D), dtype=np.float32)
    for m in range(M):
        u = np.zeros((2, M), dtype=np.float32); u[:, m] = 1.0
        tm = hmm.transitions.transition_matrices(fake_obs, u, mask=None, tag=None)
        # tm shape is (T-1, K, K); take any time slice since input is constant
        out[m] = np.asarray(tm[0])
    return out


def emission_log_rates(hmm: HMM) -> np.ndarray:
    """Return (K, D) emission log-rate matrix."""
    return np.asarray(hmm.observations.log_lambdas)


def save_model(hmm: HMM, path) -> None:
    with open(path, "wb") as f:
        pickle.dump(hmm, f)


def load_model(path) -> HMM:
    with open(path, "rb") as f:
        return pickle.load(f)
