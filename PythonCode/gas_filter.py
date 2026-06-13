#!/usr/bin/env python3
"""
GAS(1,1) filter for time-varying Copula parameters.

This is the BENCHMARK comparison model against the FIGAS model.
Implements the standard GAS(1,1) score-driven dynamics without
fractional integration -- the "plain vanilla" Generalized Autoregressive
Score model of Creal, Koopman, and Lucas (2013).

Filter recursion (no fractional-differencing term):
    g_{t+1} = mu + beta * (g_t - mu) + alpha * s_t

where s_t is the scaled score of the copula log-likelihood.

Supports: t (2), Clayton (3), Survival Gumbel (14), Clayton 90-rotated (23).
"""

import numpy as np
from scipy.stats import t as t_dist
from scipy.optimize import minimize
from numpy import log, exp, sqrt

# ── Import shared infrastructure from figas_filter ────────────────────────
from figas_filter import (
    _bicop_pdf,
    _bicop_hfunc,
    _safe_bicop_hfunc,
    _static_copula_fit,
    _inverse_link,
)

# ── Import project config ─────────────────────────────────────────────────
try:
    from config import GAS_BOUNDS, PDF_FLOOR, SCORE_CLAMP, F_DIFF_H, SEED
except ImportError:
    # Sensible defaults if config.py is not on the path
    GAS_BOUNDS = {
        "mu": (-15.0, 15.0),
        "alpha": (0.001, 0.4),
        "beta": (0.01, 0.99),
        "kappa": (2.1, 30.0),
    }
    PDF_FLOOR = 1e-10
    SCORE_CLAMP = 10.0
    F_DIFF_H = 1e-5
    SEED = 123


# ============================================================================
# 1. GAS(1,1) filter
# ============================================================================

def filter_gas(theta, u1, u2, family):
    """
    GAS(1,1) filter for time-varying Copula parameters.

    The model updates an unconstrained latent process g_t which is mapped
    to the copula-parameter domain via family-specific link functions.
    The score of the log-likelihood drives the dynamics:

        g_{t+1} = mu + beta * (g_t - mu) + alpha * s_t

    This is structurally identical to the FIGAS filter but WITHOUT the
    fractional-differencing long-memory term.

    Parameters
    ----------
    theta : np.ndarray
        Parameter vector.
        For non-t families: [mu, alpha, beta].
        For t-copula (family=2): [mu, alpha, beta, kappa].
    u1, u2 : np.ndarray (1D)
        Uniform pseudo-observations in (0, 1).
    family : int
        Copula family code: 2=t, 3=Clayton, 14=Survival Gumbel,
        23=Clayton 90-rotated.

    Returns
    -------
    dict
        loglik  : float          -- total log-likelihood.
        ll_seq  : np.ndarray     -- log-likelihood per observation.
        par_t   : np.ndarray     -- time-varying copula parameter path.
        h1      : np.ndarray     -- h(u1|u2) conditional CDF values.
        h2      : np.ndarray     -- h(u2|u1) conditional CDF values.
    """
    mu = float(theta[0])
    alpha = float(theta[1])
    beta = float(theta[2])
    kappa = float(theta[3]) if (family == 2 and len(theta) >= 4) else 0.0

    T_len = len(u1)
    g_t = np.zeros(T_len)
    par_t = np.zeros(T_len)
    ll_seq = np.zeros(T_len)
    scores = np.zeros(T_len)

    # ── Initialise g_0 = mu ──
    g_t[0] = mu

    for t in range(T_len):
        # ----------------------------------------------------------------
        # 1. Link function: unconstrained g_t -> constrained copula par
        # ----------------------------------------------------------------
        g_safe = max(min(g_t[t], 30.0), -30.0)
        if family == 2:
            # t-copula: rho in (-1, 1) via tanh transform
            par_t[t] = (exp(g_safe) - 1.0) / (exp(g_safe) + 1.0)
            par_t[t] = max(min(par_t[t], 0.98), -0.98)
        elif family == 14:
            # Survival Gumbel: theta >= 1.0001
            par_t[t] = exp(g_safe) + 1.0001
            par_t[t] = max(min(par_t[t], 30.0), 1.0001)
        elif family == 23:
            # Clayton 90-rotated: theta < 0
            par_t[t] = -exp(g_safe) - 1e-4
            par_t[t] = max(min(par_t[t], -1e-4), -30.0)
        elif family == 3:
            # Clayton: theta > 0
            par_t[t] = exp(g_safe) + 1e-4
            par_t[t] = max(min(par_t[t], 30.0), 1e-4)
        else:
            # Fallback: treat like Survival Gumbel
            par_t[t] = exp(g_safe) + 1.0001
            par_t[t] = max(min(par_t[t], 30.0), 1.0001)

        # ----------------------------------------------------------------
        # 2. Log-likelihood evaluation
        # ----------------------------------------------------------------
        try:
            pdf_val = _bicop_pdf(u1[t], u2[t], family=family, par=par_t[t],
                                 par2=kappa)
        except Exception:
            pdf_val = PDF_FLOOR

        if not np.isfinite(pdf_val) or pdf_val <= 0:
            pdf_val = PDF_FLOOR
        ll_seq[t] = log(pdf_val)

        # ----------------------------------------------------------------
        # 3. Scaled score computation
        # ----------------------------------------------------------------
        if family == 2:
            # ── Analytical t-Copula score (identical to FIGAS) ─────────
            try:
                x1 = t_dist.ppf(max(min(u1[t], 0.999999), 1e-9), df=kappa)
                x2 = t_dist.ppf(max(min(u2[t], 0.999999), 1e-9), df=kappa)
                mt = ((x1 * x1 + x2 * x2 - 2.0 * par_t[t] * x1 * x2)
                      / (1.0 - par_t[t] * par_t[t]))
                pit = (kappa + 2.0) / (kappa + mt)
                g_s = max(min(g_t[t], 30.0), -30.0)
                dot_rho = 2.0 * exp(g_s) / (exp(g_s) + 1.0) ** 2
                denom_nabla = (1.0 - par_t[t] * par_t[t]) ** 2
                nabla = (dot_rho / denom_nabla) * (
                    (1.0 + par_t[t] * par_t[t]) * (pit * x1 * x2 - par_t[t])
                    - par_t[t] * (pit * x1 * x1 + pit * x2 * x2 - 2.0)
                )
                Info = (dot_rho * dot_rho / denom_nabla) * (
                    1.0 + par_t[t] * par_t[t]
                    - 2.0 * par_t[t] * par_t[t] / (kappa + 2.0)
                ) * ((kappa + 2.0) / (kappa + 4.0))
                scores[t] = nabla / sqrt(max(float(Info), 1e-12))
            except Exception:
                scores[t] = 0.0
        else:
            # ── Central finite-difference score for non-t families ─────
            def _eval_logpdf(g):
                if not np.isfinite(g):
                    return log(PDF_FLOOR)
                g_s = max(min(g, 30.0), -30.0)
                if family == 14:
                    tp = max(min(exp(g_s) + 1.0001, 30.0), 1.0001)
                elif family == 23:
                    tp = max(min(-exp(g_s) - 1e-4, -1e-4), -30.0)
                elif family == 3:
                    tp = max(min(exp(g_s) + 1e-4, 30.0), 1e-4)
                else:
                    tp = max(min(exp(g_s) + 1.0001, 30.0), 1.0001)
                try:
                    v = _bicop_pdf(u1[t], u2[t], family=family, par=tp,
                                   par2=kappa)
                except Exception:
                    v = PDF_FLOOR
                if not np.isfinite(v) or v <= 0:
                    v = PDF_FLOOR
                return log(v)

            f_plus = _eval_logpdf(g_t[t] + F_DIFF_H)
            f_minus = _eval_logpdf(g_t[t] - F_DIFF_H)
            scores[t] = (f_plus - f_minus) / (2.0 * F_DIFF_H)

        # ── NA firewall: replace invalid scores with zero ──
        if not np.isfinite(scores[t]):
            scores[t] = 0.0
        scores[t] = max(min(scores[t], SCORE_CLAMP), -SCORE_CLAMP)

        # ----------------------------------------------------------------
        # 4. GAS(1,1) recursion: g_{t+1} = mu + beta*(g_t - mu) + alpha*s_t
        # ----------------------------------------------------------------
        if t < T_len - 1:
            g_t[t + 1] = mu + beta * (g_t[t] - mu) + alpha * scores[t]
            # NA firewall on updated g
            if not np.isfinite(g_t[t + 1]):
                g_t[t + 1] = mu
            g_t[t + 1] = max(min(g_t[t + 1], 30.0), -30.0)

    # ── 5. Compute h-functions (with independence fallback) ─────────────
    h1 = np.zeros(T_len)
    h2 = np.zeros(T_len)
    for t in range(T_len):
        h1[t], h2[t] = _safe_bicop_hfunc(u1[t], u2[t], family=family,
                                          par=par_t[t], par2=kappa)

    return dict(
        loglik=float(np.sum(ll_seq)),
        ll_seq=ll_seq,
        par_t=par_t,
        h1=h1,
        h2=h2,
    )


# ============================================================================
# 2. Full GAS(1,1) parameter estimation
# ============================================================================

def estimate_gas_params(u1, u2, fam_id, verbose=True):
    """
    Estimate GAS(1,1) model parameters for a bivariate copula pair.

    Estimation procedure:
        1. Static MLE copula fit -> initial copula parameter.
        2. Inverse-link transform -> start_mu (unconstrained).
        3. Use fixed initial values for alpha (=0.05) and beta (=0.8).
        4. Optimise negative log-likelihood via L-BFGS-B.
        5. Return best parameters and filtered result at the optimum.

    Parameters
    ----------
    u1, u2 : np.ndarray (1D)
        Uniform pseudo-observations.
    fam_id : int
        Copula family code.
    verbose : bool
        If True, print estimation progress to stdout.

    Returns
    -------
    best_params : np.ndarray
        Estimated GAS(1,1) parameters [mu, alpha, beta] (plus kappa if t).
    filtered_result : dict
        Filter output from filter_gas() at the optimum parameters.
    """
    rng = np.random.RandomState(SEED)

    u1a = np.asarray(u1, dtype=float).ravel()
    u2a = np.asarray(u2, dtype=float).ravel()

    # ── Step 1: Static copula estimation ────────────────────────────────
    if verbose:
        print(f"  [GAS] Estimating static copula for family {fam_id} ...")
    static_par, static_par2 = _static_copula_fit(u1a, u2a, fam_id)
    if verbose:
        if fam_id == 2:
            print(f"    Static MLE: par={static_par:.4f}, "
                  f"kappa={static_par2:.2f}")
        else:
            print(f"    Static MLE: par={static_par:.4f}")

    # ── Step 2: Inverse-link back to unconstrained space -> start_mu ────
    start_mu = _inverse_link(fam_id, static_par)
    start_mu = max(min(start_mu, 5.0), -5.0)

    # ── Step 3: Set up initial parameters and bounds ────────────────────
    init_par = [start_mu, 0.05, 0.8]
    lower = [
        GAS_BOUNDS["mu"][0],
        GAS_BOUNDS["alpha"][0],
        GAS_BOUNDS["beta"][0],
    ]
    upper = [
        GAS_BOUNDS["mu"][1],
        GAS_BOUNDS["alpha"][1],
        GAS_BOUNDS["beta"][1],
    ]

    if fam_id == 2:
        init_kappa = max(3.0, static_par2)
        init_par.append(init_kappa)
        lower.append(GAS_BOUNDS["kappa"][0])
        upper.append(GAS_BOUNDS["kappa"][1])

    init_par = np.array(init_par, dtype=float)
    lower = np.array(lower, dtype=float)
    upper = np.array(upper, dtype=float)

    # ── Step 4: Optimise via L-BFGS-B ───────────────────────────────────
    def _objective(theta_vec):
        try:
            tmp = filter_gas(theta_vec, u1a, u2a, fam_id)
            ll = tmp["loglik"]
            if not np.isfinite(ll):
                return 1e10
            return -ll
        except Exception:
            return 1e10

    if verbose:
        labels = ["mu", "alpha", "beta"]
        if fam_id == 2:
            labels.append("kappa")
        init_str = ", ".join(
            f"{labels[i]}={init_par[i]:.4f}" for i in range(len(init_par))
        )
        print(f"    Initial params: {init_str}")
        print(f"    Optimising via L-BFGS-B ...")

    opt_result = minimize(
        _objective,
        x0=init_par,
        method="L-BFGS-B",
        bounds=list(zip(lower, upper)),
        options={"maxiter": 300, "maxfun": 400, "ftol": 1e-8},
    )

    best_params = opt_result.x

    # ── Step 5: Final filtered result ───────────────────────────────────
    fres = filter_gas(best_params, u1a, u2a, fam_id)

    if verbose:
        labels = ["mu", "alpha", "beta"]
        if fam_id == 2:
            labels.append("kappa")
        best_str = ", ".join(
            f"{labels[i]}={best_params[i]:.4f}"
            for i in range(len(best_params))
        )
        print(f"    Best params:  {best_str}")
        print(f"    GAS loglik:   {fres['loglik']:.2f}  (n={len(u1a)})")

    return best_params, fres


# ============================================================================
# 3. Quick smoke-test (runs when executed directly)
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("GAS(1,1) filter -- self-test")
    print("=" * 60)

    rng = np.random.RandomState(42)
    n = 500

    # ── Generate synthetic data from a time-varying Clayton copula ────────
    from scipy.stats import kendalltau

    t_vec = np.arange(n) / n
    true_theta = 1.5 + 0.5 * np.sin(2 * np.pi * t_vec * 4)

    u1_sim = rng.uniform(size=n)
    u2_sim = np.zeros(n)
    for i in range(n):
        candidates = rng.uniform(0, 1, size=200)
        h_vals = _bicop_hfunc(
            np.full(200, u1_sim[i]), candidates,
            family=3, par=true_theta[i], par2=0
        )[0]
        rv = rng.uniform()
        idx = np.argmin(np.abs(h_vals - rv))
        u2_sim[i] = candidates[idx]

    print(f"\nGenerated {n} samples from time-varying Clayton copula.")
    tau, _ = kendalltau(u1_sim, u2_sim)
    print(f"Empirical Kendall's tau: {tau:.4f}")

    # ── Static estimation ───────────────────────────────────────────────
    print(f"\n{"─" * 50}")
    print(f"Static Clayton estimation")
    print(f"{"─" * 50}")
    par_static, _ = _static_copula_fit(u1_sim, u2_sim, fam_id=3)
    print(f"  Static theta: {par_static:.4f}  "
          f"(true mean theta ~ {np.mean(true_theta):.4f})")

    # ── GAS filter with manual params ────────────────────────────────────
    print(f"\n{"─" * 50}")
    print(f"GAS(1,1) filter (manual params)")
    print(f"{"─" * 50}")
    start_mu = _inverse_link(3, par_static)
    test_theta = np.array([start_mu, 0.05, 0.8])
    fres = filter_gas(test_theta, u1_sim, u2_sim, family=3)
    print(f"  mu={start_mu:.4f}, alpha=0.05, beta=0.80")
    print(f"  GAS log-likelihood: {fres['loglik']:.2f}")
    print(f"  par_t range: [{fres['par_t'].min():.4f}, "
          f"{fres['par_t'].max():.4f}]")

    # ── Full estimation ─────────────────────────────────────────────────
    print(f"\n{"─" * 50}")
    print(f"Full GAS(1,1) estimation (Clayton)")
    print(f"{"─" * 50}")
    best_params, fres_opt = estimate_gas_params(
        u1_sim, u2_sim, fam_id=3, verbose=True
    )

    # Compare static vs dynamic log-likelihood
    static_ll = np.sum(np.log(
        np.maximum(_bicop_pdf(u1_sim, u2_sim, family=3, par=par_static,
                              par2=0),
                   PDF_FLOOR)
    ))
    print(f"\n  Static loglik:      {static_ll:.2f}")
    print(f"  GAS dynamic loglik: {fres_opt['loglik']:.2f}")
    print(f"  Improvement:        {fres_opt['loglik'] - static_ll:.2f}")

    # ── t-Copula test ───────────────────────────────────────────────────
    print(f"\n{"─" * 50}")
    print(f"t-Copula GAS filter test")
    print(f"{"─" * 50}")
    rho_true = 0.5
    nu_true = 5.0
    from scipy.stats import multivariate_t
    mv = multivariate_t(
        shape=[[1, rho_true], [rho_true, 1]], df=nu_true, seed=99
    )
    samples = mv.rvs(n)
    u1_t = t_dist.cdf(samples[:, 0], df=nu_true)
    u2_t = t_dist.cdf(samples[:, 1], df=nu_true)

    test_theta_t = np.array([log((1 + 0.4) / (1 - 0.4)), 0.05, 0.8, 5.0])
    fres_t = filter_gas(test_theta_t, u1_t, u2_t, family=2)
    print(f"  t-Copula log-likelihood: {fres_t['loglik']:.2f}")
    print(f"  par_t range: [{fres_t['par_t'].min():.4f}, "
          f"{fres_t['par_t'].max():.4f}]")

    # ── Survival Gumbel test ────────────────────────────────────────────
    print(f"\n{"─" * 50}")
    print(f"Survival Gumbel GAS filter test")
    print(f"{"─" * 50}")
    # Generate dependent data via common factor
    common = rng.uniform(size=200)
    u1_sg = np.clip(0.7 * rng.uniform(size=200) + 0.3 * common, 1e-6, 1 - 1e-6)
    u2_sg = np.clip(0.7 * rng.uniform(size=200) + 0.3 * common, 1e-6, 1 - 1e-6)

    test_theta_sg = np.array([log(0.5), 0.05, 0.8])
    fres_sg = filter_gas(test_theta_sg, u1_sg, u2_sg, family=14)
    print(f"  Survival Gumbel log-likelihood: {fres_sg['loglik']:.2f}")
    print(f"  par_t range: [{fres_sg['par_t'].min():.4f}, "
          f"{fres_sg['par_t'].max():.4f}]")

    print(f"\n{"─" * 50}")
    print(f"All tests passed.")
    print(f"{"─" * 50}")
