"""Custom GLM-HMM: covariate-dependent transitions + factorized mixed emissions.

Emissions: same as MixedHMM (diagonal Gaussian + Categorical + Bernoulli).

Transitions are now bin-dependent through covariates u_t (n_cov-vector):
  log A_t[i, j] = (W[i, :, j]^T @ u_t + b[i, j])
                  - logsumexp_j' (W[i, :, j']^T @ u_t + b[i, j'])

Parameters:
  pi: (K,) initial distribution (probability simplex)
  W:  (K, n_cov, K) softmax-regression weights (source i × covariate × target j)
  b:  (K, K)        softmax-regression biases  (source i × target j)
  mu, sigma, p_zone, q_events: same shapes/semantics as MixedHMM

Inference: forward pass via dynamax.hmm_filter with time-varying transition_matrix.
Backward and two-slice marginals computed in log-space to handle time-varying A.
M-step:
  pi: gamma[0] averaged across sequences
  emissions: closed-form weighted MLE with smoothing priors (same as MixedHMM)
  transitions: gradient ascent on the soft-target softmax-regression objective
               per source state, fitting W[i] and b[i].

Hard caveat: this is a substantially heavier EM iter than MixedHMM. Expect
~5-10× slower per iter and 50-200 iters to converge.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, grad
from jax.scipy.special import logsumexp

from dynamax.hidden_markov_model.inference import (
    hmm_filter,
    hmm_posterior_mode,
)


# =====  Parameters  =====
@dataclass
class GLMHMMParams:
    pi: np.ndarray                  # (K,)
    W: np.ndarray                   # (K, n_cov, K)
    b: np.ndarray                   # (K, K)
    mu: np.ndarray                  # (K, D_cont)
    sigma: np.ndarray               # (K, D_cont)
    p_zone: np.ndarray              # (K, K_zone)
    q_events: np.ndarray            # (K, n_events)

    @property
    def K(self) -> int:
        return self.pi.shape[0]

    @property
    def n_cov(self) -> int:
        return self.W.shape[1]

    @property
    def D_cont(self) -> int:
        return self.mu.shape[1]

    @property
    def K_zone(self) -> int:
        return self.p_zone.shape[1]

    @property
    def n_events(self) -> int:
        return self.q_events.shape[1]


# =====  Initialization  =====
def init_params(
    key,
    K: int,
    n_cov: int,
    X_cont_pool: np.ndarray,
    X_zone_pool: np.ndarray,
    X_events_pool: np.ndarray,
    K_zone: int,
    n_events: int,
    sigma_init_scale: float = 1.0,
    mu_jitter_sigma: float = 0.5,
    W_init_scale: float = 0.05,
    b_init_scale: float = 0.5,
) -> GLMHMMParams:
    rng = np.random.default_rng(int(jax.random.randint(key, (), 0, 2**31 - 1)))
    D_cont = X_cont_pool.shape[1]

    mu_pool = X_cont_pool.mean(axis=0)
    sigma_pool = X_cont_pool.std(axis=0) + 1e-6

    mu = mu_pool[None, :] + rng.normal(0, mu_jitter_sigma * sigma_pool[None, :],
                                          size=(K, D_cont))
    sigma = np.maximum(sigma_init_scale * np.tile(sigma_pool, (K, 1)), 1e-3)

    zone_counts = np.bincount(X_zone_pool, minlength=K_zone).astype(float)
    p_zone = np.zeros((K, K_zone))
    for k in range(K):
        sample = rng.dirichlet(zone_counts + 1.0)
        p_zone[k] = 0.5 * sample + 0.5 * (zone_counts + 1) / (zone_counts + 1).sum()

    event_freq = X_events_pool.mean(axis=0)
    q_events = np.zeros((K, n_events))
    for k in range(K):
        for e in range(n_events):
            a = event_freq[e] * 50 + 1.0
            bb = (1 - event_freq[e]) * 50 + 1.0
            q_events[k, e] = rng.beta(a, bb)

    pi = np.ones(K) / K
    # Initialize W small so initial transitions are near-uniform with diagonal bias
    W = rng.normal(0, W_init_scale, size=(K, n_cov, K))
    b = rng.normal(0, b_init_scale, size=(K, K))
    # bias self-transitions (sticky) to encourage identifiability
    for i in range(K):
        b[i, i] += 1.5

    return GLMHMMParams(
        pi=pi.astype(np.float64), W=W.astype(np.float64), b=b.astype(np.float64),
        mu=mu.astype(np.float64), sigma=sigma.astype(np.float64),
        p_zone=p_zone.astype(np.float64), q_events=q_events.astype(np.float64),
    )


# =====  Emission log-probs (same as MixedHMM)  =====
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


# =====  Transition log-probs (covariate-dependent)  =====
@jit
def _trans_log_probs_jit(W, b, covariates):
    """Return log_A: (T, K, K) where log_A[t, i, j] = log P(z_t = j | z_{t-1} = i, u_t).

    NOTE: We use covariates at time t to predict transition INTO state at time t.
    So log_A is indexed by time of the destination bin. The first row (t=0) is
    irrelevant (initial distribution handles that).

    W: (K, n_cov, K)
    b: (K, K)
    covariates: (T, n_cov)
    """
    # logits[t, i, j] = covariates[t] @ W[i, :, j] + b[i, j]
    # Easier: einsum
    logits = jnp.einsum("tc,ick->tik", covariates, W) + b[None, :, :]
    log_A = logits - logsumexp(logits, axis=2, keepdims=True)
    return log_A


# =====  Forward + backward (time-varying transitions)  =====
def _per_session_e_step(params: GLMHMMParams, x_cont, x_zone, x_events,
                          covariates):
    """Compute log-lik, gamma (T, K), xi (T-1, K, K), log_emit (T, K)
    for one session under covariate-dependent transitions."""
    pi = jnp.asarray(params.pi)
    W = jnp.asarray(params.W); bb = jnp.asarray(params.b)
    mu = jnp.asarray(params.mu); sigma = jnp.asarray(params.sigma)
    p_zone = jnp.asarray(params.p_zone)
    q_events = jnp.asarray(params.q_events)

    log_emit = _emit_log_probs_jit(mu, sigma, p_zone, q_events,
                                     jnp.asarray(x_cont),
                                     jnp.asarray(x_zone),
                                     jnp.asarray(x_events))    # (T, K)
    log_A_full = _trans_log_probs_jit(W, bb, jnp.asarray(covariates))  # (T, K, K)
    # transition_matrix (3D) for hmm_filter expects probs A[t, i, j] = P(z_t=j | z_{t-1}=i, u_t)
    # dynamax expects shape (num_timesteps, K, K) per docs — yes 3D path.
    # CAUTION: dynamax indexing convention. Let me defer to dynamax: the 3D form
    # represents A_t for each t; hmm_filter applies A_t at step t. Good.
    A_full = jnp.exp(log_A_full)

    # Forward via dynamax (3D transition path). Use t=1..T-1 transitions.
    # dynamax expects T-1 entries for predict; we'll pass full and let it index.
    fpost = hmm_filter(pi, A_full, log_emit)
    ll = float(fpost.marginal_loglik)
    filtered = jnp.asarray(fpost.filtered_probs)

    # Backward in log-space with time-varying transitions
    log_A = log_A_full
    T = log_emit.shape[0]
    K = log_emit.shape[1]

    def _backward_step(carry, args):
        log_beta_next = carry          # (K,)
        log_emit_next, log_A_next = args  # log_emit at t+1, log_A at t+1
        # log_beta[t, k] = logsumexp_j (log_A_next[k, j] + log_emit_next[j] + log_beta_next[j])
        log_msg = log_A_next + (log_emit_next + log_beta_next)[None, :]
        log_beta_t = logsumexp(log_msg, axis=1)
        return log_beta_t, log_beta_t

    # We need backward from T-2 to 0. log_beta[T-1] = 0.
    # Pair (log_emit[t+1], log_A[t+1]) for t in [0, T-2].
    inputs_emit = log_emit[1:]                        # (T-1, K)
    inputs_logA = log_A[1:]                           # (T-1, K, K)
    log_beta_init = jnp.zeros(K)
    _, log_beta_rev = jax.lax.scan(
        _backward_step, log_beta_init, (inputs_emit, inputs_logA), reverse=True
    )
    log_beta = jnp.vstack([log_beta_rev, log_beta_init[None, :]])

    # Smoothed posteriors
    log_alpha = jnp.log(filtered + 1e-30)
    log_gamma_un = log_alpha + log_beta
    log_gamma = log_gamma_un - logsumexp(log_gamma_un, axis=1, keepdims=True)
    gamma = jnp.exp(log_gamma)

    # Two-slice marginals: xi[t, i, j] = P(z_t=i, z_{t+1}=j | y_1:T, u_1:T)
    # log_xi[t, i, j] = log_alpha[t, i] + log_A[t+1, i, j] + log_emit[t+1, j] + log_beta[t+1, j] - logZ_t
    log_alpha_t = log_alpha[:-1]                      # (T-1, K)
    log_A_t1 = log_A[1:]                              # (T-1, K, K)
    log_emit_t1 = log_emit[1:]                         # (T-1, K)
    log_beta_t1 = log_beta[1:]                         # (T-1, K)
    log_xi_un = (log_alpha_t[:, :, None]
                  + log_A_t1
                  + (log_emit_t1 + log_beta_t1)[:, None, :])
    Z = logsumexp(log_xi_un.reshape(log_xi_un.shape[0], -1), axis=1)
    log_xi = log_xi_un - Z[:, None, None]
    xi = jnp.exp(log_xi)

    return ll, np.asarray(gamma), np.asarray(xi), np.asarray(log_emit)


# =====  Multi-session E-step accumulator  =====
@dataclass
class _EStats:
    n_seq: int = 0
    total_loglik: float = 0.0
    total_T: int = 0
    init_count: np.ndarray = None       # (K,)
    gamma_sum: np.ndarray = None        # (K,)
    cont_sum: np.ndarray = None         # (K, D_cont)
    cont_sumsq: np.ndarray = None       # (K, D_cont)
    zone_count: np.ndarray = None       # (K, K_zone)
    event_count: np.ndarray = None      # (K, n_events)
    # For transitions, we keep per-session arrays since the M-step is multinomial logreg
    xi_per_seq: list = field(default_factory=list)        # [(T_s-1, K, K), ...]
    gamma_per_seq: list = field(default_factory=list)     # [(T_s, K), ...]
    cov_per_seq: list = field(default_factory=list)       # [(T_s, n_cov), ...]


def _zero_stats(K, D_cont, K_zone, n_events) -> _EStats:
    return _EStats(
        init_count=np.zeros(K),
        gamma_sum=np.zeros(K),
        cont_sum=np.zeros((K, D_cont)),
        cont_sumsq=np.zeros((K, D_cont)),
        zone_count=np.zeros((K, K_zone)),
        event_count=np.zeros((K, n_events)),
    )


def e_step(params: GLMHMMParams, sessions) -> _EStats:
    """Sessions: list of dicts with X_cont, X_zone, X_events, U (covariates)."""
    K = params.K
    stats = _zero_stats(K, params.D_cont, params.K_zone, params.n_events)
    stats.n_seq = len(sessions)

    for s in sessions:
        ll, gamma, xi, _ = _per_session_e_step(
            params, s["X_cont"], s["X_zone"], s["X_events"], s["U"]
        )
        T = gamma.shape[0]
        stats.total_loglik += ll
        stats.total_T += T
        stats.init_count += gamma[0]
        stats.gamma_sum += gamma.sum(axis=0)
        stats.cont_sum += gamma.T @ s["X_cont"]
        stats.cont_sumsq += gamma.T @ (s["X_cont"] ** 2)
        zone_oh = np.zeros((T, params.K_zone))
        zone_oh[np.arange(T), s["X_zone"]] = 1.0
        stats.zone_count += gamma.T @ zone_oh
        stats.event_count += gamma.T @ s["X_events"]
        stats.xi_per_seq.append(xi)
        stats.gamma_per_seq.append(gamma)
        stats.cov_per_seq.append(s["U"])

    return stats


# =====  M-step: emissions + initial dist  =====
def _m_step_emissions(stats: _EStats, K, D_cont, K_zone, n_events,
                       var_floor=1e-3, zone_dirichlet=1.0, event_beta=0.5):
    pi = stats.init_count / max(stats.n_seq, 1)
    pi = pi / pi.sum()

    gs = stats.gamma_sum + 1e-12
    mu = stats.cont_sum / gs[:, None]
    var = stats.cont_sumsq / gs[:, None] - mu ** 2
    sigma = np.sqrt(np.maximum(var, var_floor))

    p_zone = stats.zone_count + zone_dirichlet
    p_zone = p_zone / p_zone.sum(axis=1, keepdims=True)

    q_events = (stats.event_count + event_beta) / (gs[:, None] + 2 * event_beta)
    q_events = np.clip(q_events, 1e-6, 1 - 1e-6)
    return pi, mu, sigma, p_zone, q_events


# =====  M-step: transition params via gradient ascent on soft-target softmax  =====
def _softmax_logreg_msteps(W_curr, b_curr, stats: _EStats, K, n_cov,
                            n_grad_steps=80, lr=0.05, l2=1e-3):
    """For each source state i, fit W[i, :, :] (n_cov, K) and b[i, :] (K,) by
    maximizing sum_t sum_j xi[t, i, j] log P(j | W[i] u_{t+1} + b[i]) - l2 * ||W[i]||^2.

    Uses Adam-like momentum gradient ascent. Returns updated W (K, n_cov, K), b (K, K).
    """
    # Aggregate across sessions: build the design matrix X (sum_t, n_cov) and
    # per-source target distributions.
    # xi[s][t, i, j] gives the soft target at time t+1 in session s for source i.
    # The covariate features at time t+1 in session s are cov_per_seq[s][t+1].

    # Stack covariates at t+1 (the "destination" time) and the corresponding
    # xi rows for source i.
    if not stats.xi_per_seq:
        return W_curr, b_curr

    cov_blocks = []
    xi_blocks = []
    for xi, cov in zip(stats.xi_per_seq, stats.cov_per_seq):
        # xi has length T-1 (transitions from t to t+1, t=0..T-2)
        # Covariate at t+1 corresponds to indices 1..T-1 of cov.
        T = cov.shape[0]
        cov_blocks.append(cov[1:T])           # (T-1, n_cov)
        xi_blocks.append(xi)                  # (T-1, K, K)
    X = np.concatenate(cov_blocks, axis=0)   # (sum_t, n_cov)
    XI = np.concatenate(xi_blocks, axis=0)   # (sum_t, K, K)
    # Pre-cast to JAX
    X_j = jnp.asarray(X)
    XI_j = jnp.asarray(XI)

    W = jnp.asarray(W_curr); b = jnp.asarray(b_curr)

    def loss_per_source(W_i, b_i, X_j, xi_i_targets):
        """Negative log-likelihood (to minimise) of soft-target softmax regression
        for one source state i.

        W_i: (n_cov, K), b_i: (K,), X_j: (N, n_cov), xi_i_targets: (N, K)
        Total xi mass for source i in row n is xi_i_targets[n].sum() = gamma_i[n].
        """
        logits = X_j @ W_i + b_i[None, :]                   # (N, K)
        log_probs = logits - logsumexp(logits, axis=1, keepdims=True)
        # weighted cross-entropy
        nll = -jnp.sum(xi_i_targets * log_probs)
        reg = l2 * jnp.sum(W_i ** 2)
        return nll + reg

    # Adam-like updates per source state
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    m_W = jnp.zeros_like(W); v_W = jnp.zeros_like(W)
    m_b = jnp.zeros_like(b); v_b = jnp.zeros_like(b)

    grad_fn_W = jit(grad(loss_per_source, argnums=0))
    grad_fn_b = jit(grad(loss_per_source, argnums=1))

    for step in range(n_grad_steps):
        for i in range(K):
            xi_i = XI_j[:, i, :]   # (N, K)
            gW = grad_fn_W(W[i], b[i], X_j, xi_i)
            gb = grad_fn_b(W[i], b[i], X_j, xi_i)
            t = step + 1
            m_W = m_W.at[i].set(beta1 * m_W[i] + (1 - beta1) * gW)
            v_W = v_W.at[i].set(beta2 * v_W[i] + (1 - beta2) * gW ** 2)
            m_b = m_b.at[i].set(beta1 * m_b[i] + (1 - beta1) * gb)
            v_b = v_b.at[i].set(beta2 * v_b[i] + (1 - beta2) * gb ** 2)
            mhat_W = m_W[i] / (1 - beta1 ** t)
            vhat_W = v_W[i] / (1 - beta2 ** t)
            mhat_b = m_b[i] / (1 - beta1 ** t)
            vhat_b = v_b[i] / (1 - beta2 ** t)
            W = W.at[i].set(W[i] - lr * mhat_W / (jnp.sqrt(vhat_W) + eps))
            b = b.at[i].set(b[i] - lr * mhat_b / (jnp.sqrt(vhat_b) + eps))

    return np.asarray(W), np.asarray(b)


def m_step(stats: _EStats, params: GLMHMMParams, **kwargs) -> GLMHMMParams:
    K = params.K; D_cont = params.D_cont; K_zone = params.K_zone
    n_events = params.n_events; n_cov = params.n_cov
    pi, mu, sigma, p_zone, q_events = _m_step_emissions(
        stats, K, D_cont, K_zone, n_events,
        var_floor=kwargs.get("var_floor", 1e-3),
        zone_dirichlet=kwargs.get("zone_dirichlet", 1.0),
        event_beta=kwargs.get("event_beta", 0.5),
    )
    W_new, b_new = _softmax_logreg_msteps(
        params.W, params.b, stats, K, n_cov,
        n_grad_steps=kwargs.get("trans_grad_steps", 60),
        lr=kwargs.get("trans_lr", 0.05),
        l2=kwargs.get("trans_l2", 1e-3),
    )
    return GLMHMMParams(pi=pi, W=W_new, b=b_new, mu=mu, sigma=sigma,
                         p_zone=p_zone, q_events=q_events)


# =====  Fit  =====
def fit(params: GLMHMMParams, sessions: List[dict],
        max_iters=100, tol=1e-3, verbose=False, **m_kwargs):
    history = {"loglik": [], "delta": []}
    prev_ll = -np.inf
    for it in range(max_iters):
        stats = e_step(params, sessions)
        ll = stats.total_loglik
        history["loglik"].append(ll)
        delta = ll - prev_ll
        history["delta"].append(delta)
        if verbose:
            print(f"  iter {it+1:>3}: ll={ll:.2f}  Δ={delta:+.4f}")
        params = m_step(stats, params, **m_kwargs)
        if it > 0 and abs(delta) < tol:
            if verbose:
                print(f"  converged at iter {it+1}")
            break
        prev_ll = ll
    history["n_iter"] = len(history["loglik"])
    history["final_loglik"] = ll
    return params, history


def held_out_loglik(params: GLMHMMParams, sessions):
    total_ll = 0.0; total_T = 0
    for s in sessions:
        ll, _, _, _ = _per_session_e_step(
            params, s["X_cont"], s["X_zone"], s["X_events"], s["U"]
        )
        total_ll += ll
        total_T += s["X_cont"].shape[0]
    return total_ll, total_T


def viterbi_states(params, x_cont, x_zone, x_events, covariates):
    log_emit = np.asarray(_emit_log_probs_jit(
        jnp.asarray(params.mu), jnp.asarray(params.sigma),
        jnp.asarray(params.p_zone), jnp.asarray(params.q_events),
        jnp.asarray(x_cont), jnp.asarray(x_zone), jnp.asarray(x_events)))
    log_A = np.asarray(_trans_log_probs_jit(
        jnp.asarray(params.W), jnp.asarray(params.b),
        jnp.asarray(covariates)))
    A = jnp.exp(jnp.asarray(log_A))
    states = hmm_posterior_mode(jnp.asarray(params.pi), A, jnp.asarray(log_emit))
    return np.asarray(states)


def smoothed_posteriors_and_transitions(params, x_cont, x_zone, x_events, covariates):
    """Return (gamma (T, K), per-bin transition tensor (T, K, K), marginal_ll)."""
    ll, gamma, _, _ = _per_session_e_step(params, x_cont, x_zone, x_events, covariates)
    log_A = np.asarray(_trans_log_probs_jit(
        jnp.asarray(params.W), jnp.asarray(params.b),
        jnp.asarray(covariates)))
    return gamma, np.exp(log_A), ll


# =====  Number of free parameters (for AIC/BIC)  =====
def n_free_params(K, D_cont, K_zone, n_events, n_cov):
    """Approximate count of free parameters."""
    pi = K - 1
    W = K * n_cov * (K - 1)            # softmax: K-1 columns identifiable per source
    b = K * (K - 1)
    gauss = K * D_cont * 2             # mu + sigma per dim per state
    cat = K * (K_zone - 1)
    bern = K * n_events
    return pi + W + b + gauss + cat + bern


def n_free_params_standard(K, D_cont, K_zone, n_events):
    """Number of free params for standard MixedHMM (fixed transitions)."""
    pi = K - 1
    A = K * (K - 1)                    # K source rows × K-1 free entries
    gauss = K * D_cont * 2
    cat = K * (K_zone - 1)
    bern = K * n_events
    return pi + A + gauss + cat + bern
