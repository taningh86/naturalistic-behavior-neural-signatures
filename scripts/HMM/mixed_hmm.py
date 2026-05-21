"""Custom mixed-emission HMM for behavioral data (dynamax inference + JAX M-step).

Emission model (factorized, conditional on hidden state z_t = k):
    log P(x_t | z_t = k) = sum_d log N(x_cont_td | mu_kd, sigma_kd)
                         + log Cat(x_zone_t | p_k_zone)
                         + sum_e [x_event_te * log q_ke + (1 - x_event_te) * log(1 - q_ke)]

Conditional independence between the three groups given the state. All M-step
updates are closed-form weighted MLEs with smoothing priors:
  - Gaussian: variance floor (var_floor)
  - Categorical zone: Dirichlet pseudocount alpha
  - Bernoulli events: Beta(beta_prior, beta_prior) pseudocount

Forward-backward / Viterbi delegated to dynamax.hidden_markov_model.inference.

Usage:
    params = init_params(key, K, X_cont_pool, X_zone_pool, X_events_pool, ...)
    params, history = fit(params, sessions, max_iters=500, tol=1e-4)
    held_ll, n_bins = held_out_loglik(params, test_sessions)
    posteriors = smoothed_posteriors(params, x_cont, x_zone, x_events)
    states = viterbi_states(params, x_cont, x_zone, x_events)
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, random

from dynamax.hidden_markov_model.inference import (
    hmm_filter,
    hmm_posterior_mode,
)
from jax.scipy.special import logsumexp


@dataclass
class MixedHMMParams:
    """Container for HMM parameters (numpy arrays for serialization, JAX inside fit)."""

    pi: np.ndarray              # (K,)         initial distribution
    A: np.ndarray               # (K, K)       transition matrix
    mu: np.ndarray              # (K, D_cont)  Gaussian means
    sigma: np.ndarray           # (K, D_cont)  Gaussian std devs (diagonal)
    p_zone: np.ndarray          # (K, K_zone)  zone categorical probs
    q_events: np.ndarray        # (K, n_event) per-event Bernoulli probs

    @property
    def K(self) -> int:
        return self.pi.shape[0]

    @property
    def D_cont(self) -> int:
        return self.mu.shape[1]

    @property
    def K_zone(self) -> int:
        return self.p_zone.shape[1]

    @property
    def n_events(self) -> int:
        return self.q_events.shape[1]

    def to_jax(self):
        return _JaxParams(
            pi=jnp.asarray(self.pi),
            A=jnp.asarray(self.A),
            mu=jnp.asarray(self.mu),
            sigma=jnp.asarray(self.sigma),
            p_zone=jnp.asarray(self.p_zone),
            q_events=jnp.asarray(self.q_events),
        )


@dataclass
class _JaxParams:
    pi: jax.Array
    A: jax.Array
    mu: jax.Array
    sigma: jax.Array
    p_zone: jax.Array
    q_events: jax.Array


# -----------------------------------------------------------------------------
# Initialization
# -----------------------------------------------------------------------------
def init_params(
    key: jax.Array,
    K: int,
    X_cont_pool: np.ndarray,         # (N_total, D_cont) pooled across sessions
    X_zone_pool: np.ndarray,         # (N_total,)        integer zone labels
    X_events_pool: np.ndarray,       # (N_total, n_ev)   0/1 events
    K_zone: int,
    n_events: int,
    sigma_init_scale: float = 1.0,
    mu_jitter_sigma: float = 0.5,
    zone_dirichlet_init: float = 1.0,
    event_beta_init: float = 1.0,
) -> MixedHMMParams:
    """Random initialization seeded by pooled-data statistics.

    - mu_k ~ pooled_mean + N(0, mu_jitter_sigma * pooled_std) per dim
    - sigma_k = sigma_init_scale * pooled_std (broadcast across states)
    - p_zone_k ~ Dirichlet(empirical_zone_counts + zone_dirichlet_init)
    - q_events_k ~ Beta(empirical_freq + event_beta_init, ...)
    - pi uniform; A near-uniform with small Dirichlet jitter
    """
    D_cont = X_cont_pool.shape[1]
    rng = np.random.default_rng(int(jax.random.randint(key, (), 0, 2**31 - 1)))

    mu_pool = X_cont_pool.mean(axis=0)
    sigma_pool = X_cont_pool.std(axis=0) + 1e-6

    mu = mu_pool[None, :] + rng.normal(
        0, mu_jitter_sigma * sigma_pool[None, :], size=(K, D_cont)
    )
    sigma = np.tile(sigma_init_scale * sigma_pool[None, :], (K, 1))
    sigma = np.maximum(sigma, 1e-3)

    # zone empirical with smoothing
    zone_counts = np.bincount(X_zone_pool, minlength=K_zone).astype(float)
    p_zone = np.zeros((K, K_zone))
    for k in range(K):
        sample = rng.dirichlet(zone_counts + zone_dirichlet_init)
        # mix with empirical (so states differ but all reasonable)
        p_zone[k] = 0.5 * sample + 0.5 * (zone_counts + 1) / (zone_counts + 1).sum()

    # bernoulli empirical with smoothing
    event_freq = X_events_pool.mean(axis=0)
    q_events = np.zeros((K, n_events))
    for k in range(K):
        for e in range(n_events):
            a = event_freq[e] * 50 + event_beta_init
            b = (1 - event_freq[e]) * 50 + event_beta_init
            q_events[k, e] = rng.beta(a, b)

    pi = np.ones(K) / K
    A = rng.dirichlet(np.ones(K) * 5.0, size=K)

    return MixedHMMParams(
        pi=pi.astype(np.float64),
        A=A.astype(np.float64),
        mu=mu.astype(np.float64),
        sigma=sigma.astype(np.float64),
        p_zone=p_zone.astype(np.float64),
        q_events=q_events.astype(np.float64),
    )


# -----------------------------------------------------------------------------
# Emission log-probabilities
# -----------------------------------------------------------------------------
def _emit_log_probs_jax(
    params: _JaxParams,
    x_cont: jax.Array,    # (T, D_cont)
    x_zone: jax.Array,    # (T,)
    x_events: jax.Array,  # (T, n_ev)
) -> jax.Array:           # (T, K)
    """Log P(x_t | z_t = k) for all t, k."""
    # Gaussian, diagonal: shape (T, K)
    # log p = -0.5 [log(2π) + 2*log(sigma) + ((x - mu)/sigma)^2]
    LOG2PI = jnp.log(2 * jnp.pi)
    diff = x_cont[:, None, :] - params.mu[None, :, :]              # (T, K, D)
    z2 = (diff / params.sigma[None, :, :]) ** 2                     # (T, K, D)
    gauss_log = -0.5 * (
        params.D_cont if False else 0
    )  # placeholder: we'll just sum the per-dim term
    gauss_log = -0.5 * jnp.sum(
        LOG2PI + 2 * jnp.log(params.sigma)[None, :, :] + z2, axis=-1
    )  # (T, K)

    # Categorical zone: log p_k_zone[x_zone_t]
    # gather: take params.p_zone[:, x_zone_t] for each t -> (T, K)
    zone_log = jnp.log(params.p_zone[:, x_zone].T + 1e-12)          # (T, K)

    # Bernoulli events: sum over events of [x*log q + (1-x)*log(1-q)]
    log_q = jnp.log(params.q_events + 1e-12)                        # (K, n_ev)
    log_1mq = jnp.log(1 - params.q_events + 1e-12)                  # (K, n_ev)
    # x_events: (T, n_ev). Need (T, K).
    bern_log = x_events @ log_q.T + (1 - x_events) @ log_1mq.T      # (T, K)

    return gauss_log + zone_log + bern_log


# Property used inside _emit_log_probs_jax
def _attach_dim(params, D_cont):
    # noop helper for clarity; D_cont must be a concrete int passed via jit static_argnames
    return params


@dataclass
class _EStepStats:
    """Sufficient statistics aggregated across sequences."""

    n_seq: int = 0
    total_loglik: float = 0.0
    total_T: int = 0
    init_count: np.ndarray = None   # (K,)         sum of gamma[0]
    trans_count: np.ndarray = None  # (K, K)       sum of two-slice marginals
    gamma_sum: np.ndarray = None    # (K,)         total responsibility per state
    cont_sum: np.ndarray = None     # (K, D_cont)  weighted sum
    cont_sumsq: np.ndarray = None   # (K, D_cont)  weighted sum of squares (around current mu)
    zone_count: np.ndarray = None   # (K, K_zone)  weighted zone occupancy
    event_count: np.ndarray = None  # (K, n_ev)    weighted event counts (sum_t gamma * x)


def _zero_stats(K, D_cont, K_zone, n_events) -> _EStepStats:
    return _EStepStats(
        init_count=np.zeros(K),
        trans_count=np.zeros((K, K)),
        gamma_sum=np.zeros(K),
        cont_sum=np.zeros((K, D_cont)),
        cont_sumsq=np.zeros((K, D_cont)),
        zone_count=np.zeros((K, K_zone)),
        event_count=np.zeros((K, n_events)),
    )


@jit
def _emit_log_probs_jit(mu, sigma, p_zone, q_events, x_cont, x_zone, x_events):
    LOG2PI = jnp.log(2 * jnp.pi)
    diff = x_cont[:, None, :] - mu[None, :, :]
    z2 = (diff / sigma[None, :, :]) ** 2
    gauss_log = -0.5 * jnp.sum(
        LOG2PI + 2 * jnp.log(sigma)[None, :, :] + z2, axis=-1
    )
    zone_log = jnp.log(p_zone[:, x_zone].T + 1e-12)
    log_q = jnp.log(q_events + 1e-12)
    log_1mq = jnp.log(1 - q_events + 1e-12)
    bern_log = x_events @ log_q.T + (1 - x_events) @ log_1mq.T
    return gauss_log + zone_log + bern_log


@jit
def _backward_and_trans(log_A, log_emit):
    """Log-space backward pass + per-pair sum of two-slice marginals.

    Returns:
      log_beta: (T, K)
      log_xi_sum: (K, K)  — log of sum_t P(z_t=i, z_{t+1}=j | y_1:T) up to a constant
                            that will cancel after exp + normalization across (i,j).
    """
    T, K = log_emit.shape

    # Backward messages: log_beta[T-1] = 0; log_beta[t] = logsumexp_j(log_A[k,j] + log_emit[t+1,j] + log_beta[t+1,j])
    def _step(carry, args):
        log_beta_next = carry          # (K,)
        log_emit_next = args            # (K,) at t+1
        log_msg = log_A + (log_emit_next + log_beta_next)[None, :]  # (K, K) where row=k, col=j
        log_beta_t = logsumexp(log_msg, axis=1)
        return log_beta_t, log_beta_t

    log_beta_final = jnp.zeros(K)
    _, log_beta_rev = jax.lax.scan(
        _step, log_beta_final, log_emit[1:], reverse=True
    )
    # log_beta_rev has length T-1; append the final zeros
    log_beta = jnp.vstack([log_beta_rev, log_beta_final[None, :]])
    return log_beta


def _per_session_e_step(
    pi, A, mu, sigma, p_zone, q_events,
    x_cont, x_zone, x_events,
    K_zone,
):
    """Compute log-lik, smoothed posteriors, summed two-slice marginals (trans_sum)."""
    log_emit = _emit_log_probs_jit(mu, sigma, p_zone, q_events,
                                    x_cont, x_zone, x_events)
    # Forward via dynamax (no buggy kwargs)
    fpost = hmm_filter(pi, A, log_emit)
    ll = fpost.marginal_loglik
    filtered_probs = fpost.filtered_probs   # (T, K)

    # Backward in log-space
    log_A = jnp.log(A + 1e-30)
    log_beta = _backward_and_trans(log_A, log_emit)  # (T, K)

    # Smoothed posteriors gamma[t, k] ∝ filtered_probs[t, k] * exp(log_beta[t, k])
    # In log space: log_gamma_unnorm = log filtered + log_beta; renormalize per t
    log_alpha_norm = jnp.log(filtered_probs + 1e-30)
    log_gamma_un = log_alpha_norm + log_beta
    log_gamma = log_gamma_un - logsumexp(log_gamma_un, axis=1, keepdims=True)
    gamma = jnp.exp(log_gamma)

    # Two-slice marginals: xi[t, i, j] ∝ alpha_t(i) * A[i, j] * emit_t+1(j) * beta_t+1(j)
    # In log space (using filtered_probs as alpha_t up to normalization):
    # log_xi_unnorm = log alpha_t(i) + log A(i,j) + log emit_t+1(j) + log beta_t+1(j)
    # Sum over t, then exponentiate at each (i, j) using a per-t logsumexp sum trick.
    # We compute trans_sum[i,j] = sum_t exp(log_xi_t[i,j] - logZ) where logZ is per-t normalizer.
    # Since gamma sums to 1 each t, normalize per-t.
    log_a_t = log_alpha_norm[:-1]                               # (T-1, K) -- alpha at t
    log_emit_tp1 = log_emit[1:]                                 # (T-1, K) -- emit at t+1
    log_beta_tp1 = log_beta[1:]                                 # (T-1, K)
    # log_xi_t[t, i, j] = log_a_t[t, i] + log_A[i, j] + log_emit_tp1[t, j] + log_beta_tp1[t, j]
    log_xi = (log_a_t[:, :, None]
              + log_A[None, :, :]
              + (log_emit_tp1 + log_beta_tp1)[:, None, :])      # (T-1, K, K)
    # per-t normalizer
    log_xi_norm = logsumexp(log_xi.reshape(log_xi.shape[0], -1), axis=1)  # (T-1,)
    log_xi = log_xi - log_xi_norm[:, None, None]
    # sum over t: trans_sum[i, j] = sum_t exp(log_xi[t, i, j])
    trans_sum = jnp.sum(jnp.exp(log_xi), axis=0)                # (K, K)

    return ll, gamma, trans_sum, log_emit


def e_step(params: MixedHMMParams, sessions) -> _EStepStats:
    """Aggregate sufficient statistics across sessions.

    sessions: list of dicts with keys X_cont, X_zone, X_events (numpy arrays).
    """
    K = params.K
    D_cont = params.D_cont
    K_zone = params.K_zone
    n_events = params.n_events
    stats = _zero_stats(K, D_cont, K_zone, n_events)
    stats.n_seq = len(sessions)

    pi_j = jnp.asarray(params.pi)
    A_j = jnp.asarray(params.A)
    mu_j = jnp.asarray(params.mu)
    sigma_j = jnp.asarray(params.sigma)
    p_zone_j = jnp.asarray(params.p_zone)
    q_events_j = jnp.asarray(params.q_events)

    for s in sessions:
        x_cont = jnp.asarray(s["X_cont"])
        x_zone = jnp.asarray(s["X_zone"])
        x_events = jnp.asarray(s["X_events"])

        ll, gamma, xi, _ = _per_session_e_step(
            pi_j, A_j, mu_j, sigma_j, p_zone_j, q_events_j,
            x_cont, x_zone, x_events,
            K_zone,
        )
        gamma_np = np.asarray(gamma)        # (T, K)
        trans_sum_np = np.asarray(xi)       # (K, K) — already summed over t
        x_cont_np = np.asarray(x_cont)
        x_zone_np = np.asarray(x_zone)
        x_events_np = np.asarray(x_events)
        T = gamma_np.shape[0]

        stats.total_loglik += float(ll)
        stats.total_T += T
        stats.init_count += gamma_np[0]
        stats.trans_count += trans_sum_np
        stats.gamma_sum += gamma_np.sum(axis=0)
        # weighted sums for Gaussian — accumulate around 0 (M-step computes mu, then var)
        stats.cont_sum += gamma_np.T @ x_cont_np                 # (K, D)
        stats.cont_sumsq += gamma_np.T @ (x_cont_np ** 2)        # (K, D)
        # zone weighted counts: sum_{t: x_zone_t = z} gamma[t, k]
        # one-hot zone, then matmul
        T_idx = np.arange(T)
        zone_oh = np.zeros((T, K_zone))
        zone_oh[T_idx, x_zone_np] = 1.0
        stats.zone_count += gamma_np.T @ zone_oh                 # (K, K_zone)
        # event weighted counts
        stats.event_count += gamma_np.T @ x_events_np            # (K, n_ev)

    return stats


def m_step(
    stats: _EStepStats,
    K: int,
    D_cont: int,
    K_zone: int,
    n_events: int,
    var_floor: float = 1e-3,
    zone_dirichlet: float = 1.0,
    event_beta: float = 0.5,
) -> MixedHMMParams:
    """Closed-form weighted MLE updates with smoothing priors."""
    # Initial distribution
    pi = stats.init_count / max(stats.n_seq, 1)
    pi = pi / pi.sum()

    # Transition matrix (row-normalize)
    A = stats.trans_count + 1e-6
    A = A / A.sum(axis=1, keepdims=True)

    gs = stats.gamma_sum + 1e-12   # (K,)

    # Gaussian
    mu = stats.cont_sum / gs[:, None]
    var = stats.cont_sumsq / gs[:, None] - mu ** 2
    var = np.maximum(var, var_floor)
    sigma = np.sqrt(var)

    # Categorical zone with Dirichlet smoothing
    p_zone = stats.zone_count + zone_dirichlet
    p_zone = p_zone / p_zone.sum(axis=1, keepdims=True)

    # Bernoulli events with Beta smoothing
    q_events = (stats.event_count + event_beta) / (gs[:, None] + 2 * event_beta)
    q_events = np.clip(q_events, 1e-6, 1 - 1e-6)

    return MixedHMMParams(pi=pi, A=A, mu=mu, sigma=sigma, p_zone=p_zone, q_events=q_events)


def fit(
    params: MixedHMMParams,
    sessions: List[dict],
    max_iters: int = 500,
    tol: float = 1e-4,
    var_floor: float = 1e-3,
    zone_dirichlet: float = 1.0,
    event_beta: float = 0.5,
    verbose: bool = False,
) -> Tuple[MixedHMMParams, dict]:
    """EM loop. Returns (final_params, history) with history['loglik'] per iter."""
    history = {"loglik": [], "delta": []}
    prev_ll = -np.inf
    for it in range(max_iters):
        stats = e_step(params, sessions)
        ll = stats.total_loglik
        history["loglik"].append(ll)
        delta = ll - prev_ll
        history["delta"].append(delta)
        if verbose:
            print(f"  EM iter {it+1:3d}  loglik={ll:.3f}  Δ={delta:+.5f}")
        params = m_step(
            stats, params.K, params.D_cont, params.K_zone, params.n_events,
            var_floor=var_floor, zone_dirichlet=zone_dirichlet, event_beta=event_beta,
        )
        if it > 0 and abs(delta) < tol:
            if verbose:
                print(f"  converged at iter {it+1}")
            break
        prev_ll = ll
    history["n_iter"] = len(history["loglik"])
    history["final_loglik"] = ll
    return params, history


def held_out_loglik(params: MixedHMMParams, sessions: List[dict]) -> Tuple[float, int]:
    """Total log-likelihood and total number of bins across held-out sessions."""
    total_ll = 0.0
    total_T = 0
    pi_j = jnp.asarray(params.pi)
    A_j = jnp.asarray(params.A)
    mu_j = jnp.asarray(params.mu)
    sigma_j = jnp.asarray(params.sigma)
    p_zone_j = jnp.asarray(params.p_zone)
    q_events_j = jnp.asarray(params.q_events)
    for s in sessions:
        x_cont = jnp.asarray(s["X_cont"])
        x_zone = jnp.asarray(s["X_zone"])
        x_events = jnp.asarray(s["X_events"])
        ll, _, _, _ = _per_session_e_step(
            pi_j, A_j, mu_j, sigma_j, p_zone_j, q_events_j,
            x_cont, x_zone, x_events,
            params.K_zone,
        )
        total_ll += float(ll)
        total_T += s["X_cont"].shape[0]
    return total_ll, total_T


def viterbi_states(params: MixedHMMParams, x_cont, x_zone, x_events) -> np.ndarray:
    log_emit = _emit_log_probs_jit(
        jnp.asarray(params.mu),
        jnp.asarray(params.sigma),
        jnp.asarray(params.p_zone),
        jnp.asarray(params.q_events),
        jnp.asarray(x_cont),
        jnp.asarray(x_zone),
        jnp.asarray(x_events),
    )
    states = hmm_posterior_mode(jnp.asarray(params.pi), jnp.asarray(params.A), log_emit)
    return np.asarray(states)


def smoothed_posteriors(params: MixedHMMParams, x_cont, x_zone, x_events):
    """Returns (gamma (T, K), marginal_loglik)."""
    ll, gamma, _, _ = _per_session_e_step(
        jnp.asarray(params.pi),
        jnp.asarray(params.A),
        jnp.asarray(params.mu),
        jnp.asarray(params.sigma),
        jnp.asarray(params.p_zone),
        jnp.asarray(params.q_events),
        jnp.asarray(x_cont), jnp.asarray(x_zone), jnp.asarray(x_events),
        params.K_zone,
    )
    return np.asarray(gamma), float(ll)
