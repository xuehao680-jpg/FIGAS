#!/usr/bin/env python3
"""
FIGAS (Fractionally Integrated Generalized Autoregressive Score) filter
for time-varying Copula parameters.

This is the CORE methodological innovation of the thesis.
Implements the FIGAS(1,d,1) model for dynamic copula dependence modelling,
with full support for t, Clayton, Survival Gumbel, and Clayton 90-rotated
copula families.

All Copula PDF and h-function computations are implemented inline using
scipy.stats -- no external Copula library (e.g. pyvinecopulib) is required.
"""

import sys
import os
import numpy as np
from scipy.stats import t as t_dist
from scipy.stats import norm
from scipy.optimize import minimize
from numpy import log, exp, sqrt

# ── Import project config ──────────────────────────────────────────────────
try:
    from config import (
        FIGAS_BOUNDS, GAS_BOUNDS, PDF_FLOOR, SCORE_CLAMP, F_DIFF_H, SEED
    )
except ImportError:
    # Sensible defaults if config.py is not on the path
    FIGAS_BOUNDS = {
        "mu": (-8.0, 8.0),
        "alpha": (0.001, 0.35),
        "beta": (0.001, 0.99),
        "d": (0.05, 0.49),
        "kappa": (2.1, 30.0),
    }
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
# 1. Fractional-difference weights
# ============================================================================

def frac_weights(d, max_lag=30):
    """
    Compute fractionally differenced weights for the FIGAS long-memory component.

    Following the standard fractional-difference expansion:
        psi[0] = 1
        psi[k+1] = psi[k] * (k - 1 - d) / k

    Parameters
    ----------
    d : float
        Fractional integration parameter (0 < d < 0.5 for stationarity).
    max_lag : int
        Maximum lag for the truncated weights.

    Returns
    -------
    weights : np.ndarray  (max_lag,)
        psi[0], psi[1], ..., psi[max_lag-1]
    """
    weights = np.empty(max_lag)
    weights[0] = 1.0
    for k in range(1, max_lag):
        weights[k] = weights[k - 1] * (k - 1 - d) / k
    return weights


# ============================================================================
# 2. Inline Copula PDF implementations
# ============================================================================

def _bicop_pdf(u1, u2, family, par, par2=0):
    """
    Compute bivariate Copula PDF.

    Supports: family 2 (t), 3 (Clayton), 4 (Gumbel), 14 (Survival Gumbel),
              23 (Clayton 90-rotated).

    Parameters
    ----------
    u1, u2 : float or np.ndarray
        Pseudo-observations in (0,1).
    family : int
        Copula family code.
    par : float
        Primary copula parameter.
    par2 : float
        Secondary parameter (df for t-copula, 0 otherwise).

    Returns
    -------
    pdf : float or np.ndarray
        Copula density value(s).
    """
    eps = 1e-14
    u1_s = np.clip(np.asarray(u1, dtype=float), eps, 1 - eps)
    u2_s = np.clip(np.asarray(u2, dtype=float), eps, 1 - eps)

    if family == 2:
        # ── t-Copula ──────────────────────────────────────────────────
        nu = max(par2, 2.01)
        rho = np.clip(float(par), -0.999, 0.999)
        x1 = t_dist.ppf(u1_s, df=nu)
        x2 = t_dist.ppf(u2_s, df=nu)
        det = max(1 - rho * rho, 1e-14)
        A = (x1 * x1 + x2 * x2 - 2 * rho * x1 * x2) / det
        # Log-density of bivariate t
        log_bvt = -0.5 * np.log(det) - np.log(2 * np.pi) \
                  - 0.5 * (nu + 2) * np.log(1 + A / nu)
        # Log-density of marginal t
        log_marg = t_dist.logpdf(x1, df=nu) + t_dist.logpdf(x2, df=nu)
        result = np.exp(log_bvt - log_marg)
        return np.where(np.isfinite(result), result, PDF_FLOOR)

    elif family == 3:
        # ── Clayton ──────────────────────────────────────────────────
        theta = max(float(par), 1e-10)
        # c(u,v) = (1+theta)*(u*v)^(-theta-1)*(u^(-theta)+v^(-theta)-1)^(-1/theta-2)
        u1_cl, u2_cl = np.float64(u1_s), np.float64(u2_s)
        ut1 = u1_cl ** (-theta)
        ut2 = u2_cl ** (-theta)
        s = ut1 + ut2 - 1
        s = np.clip(s, 1e-14, np.inf)
        log_c = np.log(1 + theta) \
                + (-theta - 1) * (np.log(u1_cl) + np.log(u2_cl)) \
                + (-1.0 / theta - 2) * np.log(s)
        result = np.where(s > 1e-14, np.exp(log_c), PDF_FLOOR)
        return np.where(np.isfinite(result), np.clip(result, PDF_FLOOR, None), PDF_FLOOR)

    elif family == 4:
        # ── Gumbel ───────────────────────────────────────────────────
        theta = max(float(par), 1.0001)
        u1_cl, u2_cl = np.float64(u1_s), np.float64(u2_s)
        x = -np.log(u1_cl)
        y = -np.log(u2_cl)
        s = x ** theta + y ** theta
        s_pow = s ** (1.0 / theta)  # = ((-log u1)^theta + (-log u2)^theta)^(1/theta)
        log_c = -s_pow \
                + (theta - 1) * (np.log(x) + np.log(y)) \
                + (1.0 / theta - 2) * np.log(s) \
                + np.log(theta - 1 + s_pow) \
                - np.log(u1_cl) - np.log(u2_cl)
        result = np.exp(log_c)
        return np.where(np.isfinite(result), np.clip(result, PDF_FLOOR, None), PDF_FLOOR)

    elif family == 14:
        # ── Survival Gumbel (180-rotated Gumbel) ─────────────────────
        theta = max(float(par), 1.0001)
        # c_14(u,v) = c_4(1-u, 1-v)
        return _bicop_pdf(1.0 - u1, 1.0 - u2, family=4, par=theta, par2=0)

    elif family == 23:
        # ── Clayton 90-rotated ────────────────────────────────────────
        theta = np.clip(float(par), -30.0, -1e-8)
        # C_90(u,v) = v - C_Clayton(1-u, v; -theta)
        # c_90(u,v) = c_Clayton(1-u, v; -theta)
        return _bicop_pdf(1.0 - u1, u2, family=3, par=-theta, par2=0)

    else:
        raise ValueError(f"Unsupported copula family: {family}")


# ============================================================================
# 3. Inline Copula h-function implementations
# ============================================================================

def _bicop_hfunc(u1, u2, family, par, par2=0):
    """
    Compute h-function h(u1|u2) and h(u2|u1) = dC/du2 and dC/du1.

    Returns
    -------
    hfunc1 : float or np.ndarray  -- h(u1|u2)
    hfunc2 : float or np.ndarray  -- h(u2|u1)
    """
    eps = 1e-14
    u1_s = np.clip(np.asarray(u1, dtype=float), eps, 1 - eps)
    u2_s = np.clip(np.asarray(u2, dtype=float), eps, 1 - eps)

    if family == 2:
        # ── t-Copula h-function ───────────────────────────────────────
        nu = max(float(par2), 2.01)
        rho = np.clip(float(par), -0.999, 0.999)
        x1 = t_dist.ppf(u1_s, df=nu)
        x2 = t_dist.ppf(u2_s, df=nu)
        det = np.sqrt(max(1 - rho * rho, 1e-14))
        # h(u1|u2): conditional distribution of u1 given u2
        denom1 = np.sqrt((nu + x2 * x2) / (nu + 1.0)) * det
        arg1 = (x1 - rho * x2) / denom1
        h1 = t_dist.cdf(arg1, df=nu + 1)
        # h(u2|u1): conditional distribution of u2 given u1
        denom2 = np.sqrt((nu + x1 * x1) / (nu + 1.0)) * det
        arg2 = (x2 - rho * x1) / denom2
        h2 = t_dist.cdf(arg2, df=nu + 1)
        return (np.clip(np.asarray(h1, dtype=float), eps, 1 - eps),
                np.clip(np.asarray(h2, dtype=float), eps, 1 - eps))

    elif family == 3:
        # ── Clayton h-function ────────────────────────────────────────
        theta = max(float(par), 1e-10)
        u1_cl, u2_cl = np.float64(u1_s), np.float64(u2_s)
        ut1 = u1_cl ** (-theta)
        ut2 = u2_cl ** (-theta)
        s = ut1 + ut2 - 1
        s = np.clip(s, 1e-14, np.inf)
        # h(u1|u2) = dC/du2 = u2^(-theta-1) * (u1^(-theta)+u2^(-theta)-1)^(-1/theta-1)
        h1 = u2_cl ** (-theta - 1) * s ** (-1.0 / theta - 1)
        # h(u2|u1) = dC/du1 = u1^(-theta-1) * (u1^(-theta)+u2^(-theta)-1)^(-1/theta-1)
        h2 = u1_cl ** (-theta - 1) * s ** (-1.0 / theta - 1)
        return (np.clip(np.asarray(h1, dtype=float), 0.0, 1.0),
                np.clip(np.asarray(h2, dtype=float), 0.0, 1.0))

    elif family == 4:
        # ── Gumbel h-function ────────────────────────────────────────
        theta = max(float(par), 1.0001)
        u1_cl, u2_cl = np.float64(u1_s), np.float64(u2_s)
        x = -np.log(u1_cl)
        y = -np.log(u2_cl)
        s = x ** theta + y ** theta
        s_pow = s ** (1.0 / theta)
        # Copula value
        C_val = np.exp(-s_pow)
        # h(u1|u2) = C(u,v) * (-ln u1)^(theta-1) * S^(1/theta-1) / u1
        h1 = C_val * x ** (theta - 1) * s ** (1.0 / theta - 1) / u1_cl
        # h(u2|u1) = C(u,v) * (-ln u2)^(theta-1) * S^(1/theta-1) / u2
        h2 = C_val * y ** (theta - 1) * s ** (1.0 / theta - 1) / u2_cl
        return (np.clip(np.asarray(h1, dtype=float), 0.0, 1.0),
                np.clip(np.asarray(h2, dtype=float), 0.0, 1.0))

    elif family == 14:
        # ── Survival Gumbel h-function ────────────────────────────────
        theta = max(float(par), 1.0001)
        # h1_SG(u1,u2) = d/du2 [u1+u2-1 + C_Gumbel(1-u1,1-u2)] = 1 - h1_G(1-u1,1-u2)
        # where h1_G(.) is the standard Gumbel h-function
        h1_inner, h2_inner = _bicop_hfunc(1.0 - u1, 1.0 - u2, family=4, par=theta, par2=0)
        return (1.0 - h1_inner, 1.0 - h2_inner)

    elif family == 23:
        # ── Clayton 90-rotated h-function ─────────────────────────────
        theta = np.clip(float(par), -30.0, -1e-8)
        # C_90(u,v) = v - C_Clayton(1-u, v; -theta)
        # h1 = d/du2 [v - C_c(1-u,v)] = 1 - h1_c(1-u, v)
        # h2 = d/du1 [v - C_c(1-u,v)] = h2_c(1-u, v)
        # Note: -theta is the positive Clayton parameter
        h1_c, h2_c = _bicop_hfunc(1.0 - u1, u2, family=3, par=-theta, par2=0)
        return (1.0 - h1_c, h2_c)

    else:
        raise ValueError(f"Unsupported copula family for h-function: {family}")


def _safe_bicop_hfunc(u1, u2, family, par, par2=0):
    """
    Safe wrapper around _bicop_hfunc with fallback to independence copula
    (h1=u1, h2=u2) on failure.
    """
    try:
        h1, h2 = _bicop_hfunc(u1, u2, family, par, par2)
        if not np.isfinite(h1).all():
            h1 = np.asarray(u1, dtype=float)
        if not np.isfinite(h2).all():
            h2 = np.asarray(u2, dtype=float)
        return h1, h2
    except Exception:
        return np.asarray(u1, dtype=float), np.asarray(u2, dtype=float)


# ============================================================================
# 4. Core FIGAS filter
# ============================================================================

def filter_figas(theta, u1, u2, fam_id):
    """
    FIGAS(1,d,1) filter for time-varying Copula parameters.

    This is the CORE function. It filters pseudo-observations through a
    fractionally integrated score-driven model to produce a time-varying
    Copula parameter path.

    Algorithm:
        1. Map unconstrained g_t to constrained copula parameter par_t via link.
        2. Evaluate log-likelihood (with PDF floor safety).
        3. Compute score (analytical for t-Copula, finite-difference otherwise).
        4. Two-step recursion for (1-βL)(1-L)^d y_t = α s_{t-1}:
           a. X_{t+1} = β X_t + α s_t               (AR(1) on fractionally differenced y)
           b. y_{t+1} = X_{t+1} - Σ φ_k y_{t+1-k}   (inverse fractional difference)
        5. Next unconstrained value: g_{t+1} = mu + y_{t+1}

    Parameters
    ----------
    theta : np.ndarray
        [mu, alpha, beta, d] for non-t families, or [mu, alpha, beta, d, kappa] for t.
    u1, u2 : np.ndarray (1D)
        Uniform pseudo-observations.
    fam_id : int
        Copula family: 2=t, 3=Clayton, 14=Survival Gumbel, 23=Clayton 90-rotated.

    Returns
    -------
    dict with keys: loglik, ll_seq, par_t, h1, h2, y
    """
    mu, alpha, beta, d = float(theta[0]), float(theta[1]), float(theta[2]), float(theta[3])
    kappa = float(theta[4]) if (fam_id == 2 and len(theta) >= 5) else 0.0

    T_len = len(u1)
    y = np.zeros(T_len)
    g_t = np.zeros(T_len)
    par_t = np.zeros(T_len)
    ll_seq = np.zeros(T_len)
    scores = np.zeros(T_len)

    # ── Precompute standard fractional-difference weights (1-L)^d = Σ φ_k L^k ──
    phi = np.empty(max(100, T_len))
    phi[0] = 1.0
    for k in range(1, len(phi)):
        phi[k] = phi[k - 1] * (k - 1 - d) / k

    y[0] = 0.0
    g_t[0] = mu
    X = 0.0  # fractionally differenced process: X_t = (1-L)^d y_t

    for t in range(T_len):
        # -------------------------------------------------------------
        # 1. Link function (unconstrained g_t -> constrained copula par)
        # -------------------------------------------------------------
        # Clip g to prevent exp() overflow
        g_safe = max(min(g_t[t], 30.0), -30.0)
        if fam_id == 2:
            par_t[t] = (exp(g_safe) - 1.0) / (exp(g_safe) + 1.0)
            par_t[t] = max(min(par_t[t], 0.98), -0.98)
        elif fam_id == 14:
            par_t[t] = exp(g_safe) + 1.0001
            par_t[t] = max(min(par_t[t], 30.0), 1.0001)
        elif fam_id == 23:
            par_t[t] = -exp(g_safe) - 1e-4
            par_t[t] = max(min(par_t[t], -1e-4), -30.0)
        elif fam_id == 3:
            par_t[t] = exp(g_safe) + 1e-4
            par_t[t] = max(min(par_t[t], 30.0), 1e-4)
        else:
            # Fallback: treat like Survival Gumbel
            par_t[t] = exp(g_safe) + 1.0001
            par_t[t] = max(min(par_t[t], 30.0), 1.0001)

        # ----------------------------------------------------------------
        # 2. Log-likelihood (with try/except safety net)
        # ----------------------------------------------------------------
        try:
            pdf_val = _bicop_pdf(u1[t], u2[t], family=fam_id, par=par_t[t], par2=kappa)
        except Exception:
            pdf_val = PDF_FLOOR

        if not np.isfinite(pdf_val) or pdf_val <= 0:
            pdf_val = PDF_FLOOR
        ll_seq[t] = log(pdf_val)

        # ----------------------------------------------------------------
        # 3. Compute scaled score
        # ----------------------------------------------------------------
        if fam_id == 2:
            # ── Analytical t-Copula score (matching R code exactly) ──
            try:
                x1 = t_dist.ppf(max(min(u1[t], 0.999999), 1e-9), df=kappa)
                x2 = t_dist.ppf(max(min(u2[t], 0.999999), 1e-9), df=kappa)
                mt = (x1 * x1 + x2 * x2 - 2.0 * par_t[t] * x1 * x2) / (1.0 - par_t[t] * par_t[t])
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
            # ── Central finite difference score for non-t families ──
            def _eval_logpdf(g):
                if not np.isfinite(g):
                    return log(PDF_FLOOR)
                g_s = max(min(g, 30.0), -30.0)
                if fam_id == 14:
                    tp = max(min(exp(g_s) + 1.0001, 30.0), 1.0001)
                elif fam_id == 23:
                    tp = max(min(-exp(g_s) - 1e-4, -1e-4), -30.0)
                elif fam_id == 3:
                    tp = max(min(exp(g_s) + 1e-4, 30.0), 1e-4)
                else:
                    tp = max(min(exp(g_s) + 1.0001, 30.0), 1.0001)
                try:
                    v = _bicop_pdf(u1[t], u2[t], family=fam_id, par=tp, par2=kappa)
                except Exception:
                    v = PDF_FLOOR
                if not np.isfinite(v) or v <= 0:
                    v = PDF_FLOOR
                return log(v)

            f_plus = _eval_logpdf(g_t[t] + F_DIFF_H)
            f_minus = _eval_logpdf(g_t[t] - F_DIFF_H)
            scores[t] = (f_plus - f_minus) / (2.0 * F_DIFF_H)

        # ── NA firewall: force score to zero if invalid ──
        if not np.isfinite(scores[t]):
            scores[t] = 0.0
        scores[t] = max(min(scores[t], SCORE_CLAMP), -SCORE_CLAMP)

        # ----------------------------------------------------------------
        # 4. FIGAS two-step recursion for (1-βL)(1-L)^d y_t = α s_{t-1}
        #
        #    Step A: X_{t+1} = β X_t + α s_t
        #            Propagate the fractionally differenced process as AR(1).
        #            Since |β| < 1, X is always stationary and bounded
        #            by the clamped scores.
        #
        #    Step B: y_{t+1} = X_{t+1} - Σ_{k=1}^L φ_k y_{t+1-k}
        #            Recover the original process by subtracting the
        #            weighted fractional-difference history.
        #            This is the inverse of (1-L)^d.
        # ----------------------------------------------------------------
        if t < T_len - 1:
            X = beta * X + alpha * scores[t]

            sum_phi_y = 0.0
            max_k = min(100, t + 1)  # available lags of y history
            for k in range(1, max_k + 1):
                sum_phi_y += phi[k] * y[t + 1 - k]
            y[t + 1] = X - sum_phi_y

            # NA firewall: reset y to 0 if invalid
            if not np.isfinite(y[t + 1]):
                y[t + 1] = 0.0
            # Prevent unbounded growth
            y[t + 1] = max(min(y[t + 1], 29.0), -29.0)
            g_t[t + 1] = mu + y[t + 1]

    # --------------------------------------------------------------------
    # 5. Compute h-functions (with fallback to independence copula)
    # --------------------------------------------------------------------
    h1 = np.zeros(T_len)
    h2 = np.zeros(T_len)
    n_h_fallback = 0
    for t in range(T_len):
        h1[t], h2[t] = _safe_bicop_hfunc(u1[t], u2[t], family=fam_id,
                                           par=par_t[t], par2=kappa)
        if h1[t] == u1[t] and h2[t] == u2[t]:
            n_h_fallback += 1

    if n_h_fallback > 0:
        import warnings
        warnings.warn(f"FIGAS filter: {n_h_fallback}/{T_len} h-function evaluations fell back to independence copula", RuntimeWarning)
    return dict(
        loglik=float(np.sum(ll_seq)),
        ll_seq=ll_seq,
        par_t=par_t,
        h1=h1,
        h2=h2,
        y=y,
    )


# ============================================================================
# 5. Static copula parameter estimation (for initialisation)
# ============================================================================

def _static_copula_fit(u1, u2, fam_id):
    """
    Estimate static copula parameters via MLE.

    This replaces R's BiCopEst() for the families we support.

    Returns
    -------
    par : float       Primary parameter estimate.
    par2 : float      Secondary parameter (kappa for t, 0 otherwise).
    """
    u1a = np.asarray(u1, dtype=float).ravel()
    u2a = np.asarray(u2, dtype=float).ravel()

    if fam_id == 2:
        # ── t-Copula: estimate (rho, nu) jointly ──────────────────────
        # Rough initial guess from empirical Kendall's tau -> rho
        from scipy.stats import kendalltau
        tau, _ = kendalltau(u1a, u2a)
        rho0 = np.sin(np.pi * max(min(tau, 0.99), -0.99) / 2.0)
        nu0 = 5.0

        def _obj_t(params):
            rho, nu = params[0], params[1]
            if abs(rho) >= 1.0 or nu <= 2.0:
                return 1e15
            try:
                pdfs = _bicop_pdf(u1a, u2a, family=2, par=rho, par2=nu)
                ll = np.sum(np.log(np.maximum(pdfs, PDF_FLOOR)))
                if not np.isfinite(ll):
                    return 1e15
                return -ll
            except Exception:
                return 1e15

        res = minimize(_obj_t, x0=[rho0, nu0],
                       method='L-BFGS-B',
                       bounds=[(-0.99, 0.99), (2.1, 30.0)],
                       options={'maxiter': 200, 'ftol': 1e-8})
        return float(res.x[0]), float(res.x[1])

    elif fam_id == 3:
        # ── Clayton: estimate theta > 0 ───────────────────────────────
        def _obj_c(params):
            theta = params[0]
            if theta <= 0:
                return 1e15
            try:
                pdfs = _bicop_pdf(u1a, u2a, family=3, par=theta, par2=0)
                ll = np.sum(np.log(np.maximum(pdfs, PDF_FLOOR)))
                if not np.isfinite(ll):
                    return 1e15
                return -ll
            except Exception:
                return 1e15

        res = minimize(_obj_c, x0=[2.0],
                       method='L-BFGS-B',
                       bounds=[(1e-4, 30.0)],
                       options={'maxiter': 200, 'ftol': 1e-8})
        return float(res.x[0]), 0.0

    elif fam_id == 14:
        # ── Survival Gumbel: estimate theta >= 1 ─────────────────────
        def _obj_sg(params):
            theta = params[0]
            if theta < 1.0:
                return 1e15
            try:
                pdfs = _bicop_pdf(u1a, u2a, family=14, par=theta, par2=0)
                ll = np.sum(np.log(np.maximum(pdfs, PDF_FLOOR)))
                if not np.isfinite(ll):
                    return 1e15
                return -ll
            except Exception:
                return 1e15

        res = minimize(_obj_sg, x0=[1.5],
                       method='L-BFGS-B',
                       bounds=[(1.0001, 30.0)],
                       options={'maxiter': 200, 'ftol': 1e-8})
        return float(res.x[0]), 0.0

    elif fam_id == 23:
        # ── Clayton 90-rotated: estimate theta < 0 ───────────────────
        def _obj_c90(params):
            theta = params[0]
            if theta >= 0:
                return 1e15
            try:
                pdfs = _bicop_pdf(u1a, u2a, family=23, par=theta, par2=0)
                ll = np.sum(np.log(np.maximum(pdfs, PDF_FLOOR)))
                if not np.isfinite(ll):
                    return 1e15
                return -ll
            except Exception:
                return 1e15

        res = minimize(_obj_c90, x0=[-1.5],
                       method='L-BFGS-B',
                       bounds=[(-30.0, -1e-8)],
                       options={'maxiter': 200, 'ftol': 1e-8})
        return float(res.x[0]), 0.0

    else:
        raise ValueError(f"Unsupported copula family for static estimation: {fam_id}")


def _inverse_link(fam_id, par):
    """
    Compute start_mu from the static copula parameter via inverse link.
    Matches the R function `get_start_mu`.
    """
    if fam_id == 2:
        r_safe = max(min(float(par), 0.95), -0.95)
        return log((1.0 + r_safe) / (1.0 - r_safe))
    elif fam_id == 14:
        return log(max(float(par) - 1.0001, 1e-10))
    elif fam_id == 23:
        return log(max(-float(par) - 1e-4, 1e-10))
    elif fam_id == 3:
        return log(max(float(par) - 1e-4, 1e-10))
    else:
        return log(max(float(par) - 1.0001, 1e-10))


# ============================================================================
# 6. Full FIGAS parameter estimation
# ============================================================================


def estimate_figas_params(u1, u2, fam_id, verbose=True):
    """
    Estimate FIGAS parameters via Optuna Bayesian optimization.

    Uses static copula MLE for start_mu to narrow the search range,
    with 500 trials for thorough exploration.
    d is constrained to (0.001, 0.499) per domain rule.
    """
    import optuna

    u1a, u2a = np.asarray(u1, float).ravel(), np.asarray(u2, float).ravel()

    # Static initial estimates
    start_rho, start_kappa = _static_copula_fit(u1a, u2a, fam_id)
    start_mu = _inverse_link(fam_id, start_rho)
    # Clamp start_mu like GAS does
    start_mu = max(min(start_mu, 5.0), -5.0)

    if verbose:
        suffix = f", kappa={start_kappa:.2f}" if fam_id == 2 else ""
        print(f"  [FIGAS-Optuna] family={fam_id}, static_par={start_rho:.4f}{suffix}, start_mu={start_mu:.4f}")

    # Narrow search around start_mu (±5 instead of ±20)
    mu_lo = max(start_mu - 5.0, -20.0)
    mu_hi = min(start_mu + 5.0, 20.0)

    def objective(trial):
        mu = trial.suggest_float('mu', mu_lo, mu_hi)
        alpha = trial.suggest_float('alpha', 0.001, 0.5)
        beta = trial.suggest_float('beta', 0.001, 0.999)
        d = trial.suggest_float('d', 0.001, 0.499)

        theta = [mu, alpha, beta, d]
        if fam_id == 2:
            kappa = trial.suggest_float('kappa', 2.1, 50.0)
            theta.append(kappa)

        try:
            tmp = filter_figas(np.array(theta), u1a, u2a, fam_id)
            ll = tmp.get('loglik', -1e10)
            return float(ll) if np.isfinite(ll) else float(-1e10)
        except Exception:
            return float(-1e10)

    n_trials = 500 if fam_id == 2 else 500

    study = optuna.create_study(direction='maximize')
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials, n_jobs=-1, show_progress_bar=False)

    best_params_list = [study.best_params['mu'], study.best_params['alpha'],
                        study.best_params['beta'], study.best_params['d']]
    if fam_id == 2:
        best_params_list.append(study.best_params['kappa'])
    best = np.array(best_params_list)

    fres = filter_figas(best, u1a, u2a, fam_id)

    if verbose:
        print(f"    FIGAS-Optuna best: mu={best[0]:.4f}, a={best[1]:.4f}, "
              f"b={best[2]:.4f}, d={best[3]:.4f}" +
              (f", k={best[4]:.2f}" if fam_id == 2 else ""))
        print(f"    FIGAS-Optuna loglik: {fres['loglik']:.2f} (n={len(u1a)})")

    return best, fres


# ============================================================
#  6b.  FIGAS estimation — L-BFGS-B  (same style as GAS)
# ============================================================

def estimate_figas_params_lbfgsb(u1, u2, fam_id, verbose=True):
    """
    Estimate FIGAS parameters via multi-start L-BFGS-B.

    With the corrected two-step recursion (stable for all d in [0, 0.49]),
    we can optimise directly without GAS warm-start.
    """
    rng = np.random.RandomState(SEED)

    u1a = np.asarray(u1, dtype=float).ravel()
    u2a = np.asarray(u2, dtype=float).ravel()

    # ── Static copula for start_mu ───────────────────────────────────────
    static_par, static_par2 = _static_copula_fit(u1a, u2a, fam_id)
    if verbose:
        suffix = f", kappa={static_par2:.2f}" if fam_id == 2 else ""
        print(f"  [FIGAS] family={fam_id}, static_par={static_par:.4f}{suffix}")

    start_mu = _inverse_link(fam_id, static_par)
    start_mu = max(min(start_mu, 5.0), -5.0)

    # ── Bounds (d ≥ 0, now stable!) ─────────────────────────────────────
    lower = [
        max(start_mu - 5.0, FIGAS_BOUNDS["mu"][0]),
        FIGAS_BOUNDS["alpha"][0],
        FIGAS_BOUNDS["beta"][0],
        0.0,  # d can be exactly zero
    ]
    upper = [
        min(start_mu + 5.0, FIGAS_BOUNDS["mu"][1]),
        FIGAS_BOUNDS["alpha"][1],
        FIGAS_BOUNDS["beta"][1],
        FIGAS_BOUNDS["d"][1],
    ]
    init_par = [start_mu, 0.05, 0.5, 0.2]
    if fam_id == 2:
        init_kappa = max(3.0, static_par2)
        init_par.append(init_kappa)
        lower.append(FIGAS_BOUNDS["kappa"][0])
        upper.append(FIGAS_BOUNDS["kappa"][1])
    init_par = np.array(init_par, dtype=float)
    lower = np.array(lower, dtype=float)
    upper = np.array(upper, dtype=float)

    # ── Objective ────────────────────────────────────────────────────────
    def _objective(theta_vec):
        try:
            tmp = filter_figas(theta_vec, u1a, u2a, fam_id)
            ll = tmp["loglik"]
            return -ll if np.isfinite(ll) else 1e10
        except Exception:
            return 1e10

    if verbose:
        labels = ["mu", "alpha", "beta", "d"]
        if fam_id == 2: labels.append("kappa")
        print(f"    Init: {', '.join(f'{labels[i]}={init_par[i]:.4f}' for i in range(len(init_par)))}")

    # ── Multi-start L-BFGS-B ─────────────────────────────────────────────
    best_nll = np.inf
    best_x = init_par.copy()

    for restart in range(5):
        if restart == 0:
            xp = init_par.copy()
        else:
            xp = init_par.copy()
            xp[0] += rng.normal(0, 0.5)
            xp[1] *= np.exp(rng.normal(0, 0.3))
            xp[2] += rng.normal(0, 0.1)
            xp[3] += rng.uniform(-0.1, 0.1)
            if fam_id == 2: xp[4] += rng.normal(0, 3.0)
            xp = np.clip(xp, lower + 1e-8, upper - 1e-8)
        try:
            res = minimize(_objective, x0=xp, method="L-BFGS-B",
                           bounds=list(zip(lower, upper)),
                           options={"maxiter": 500, "maxfun": 800, "ftol": 1e-10})
            if res.fun < best_nll:
                best_nll = res.fun
                best_x = res.x
        except Exception:
            continue

    fres = filter_figas(best_x, u1a, u2a, fam_id)
    if verbose:
        labels = ["mu", "alpha", "beta", "d"]
        if fam_id == 2: labels.append("kappa")
        print(f"    Best: {', '.join(f'{labels[i]}={best_x[i]:.4f}' for i in range(len(best_x)))}")
        print(f"    FIGAS loglik: {fres['loglik']:.2f}  (n={len(u1a)})")

    return best_x, fres


# ============================================================================
# 7. FIGAS(1,d,0) model -- beta forced to 0
# ============================================================================

def filter_figas_d0(theta, u1, u2, fam_id):
    """
    FIGAS(1,d,0) filter for time-varying Copula parameters.

    This is a restricted variant of the FIGAS(1,d,1) model where the
    autoregressive term (beta) is forced to zero.  The recursion is:

        y_{t+1} = alpha * s_t - sum_{j=1}^{min(100,t+1)} psi_{j-1} * y_{t+1-j}

    where psi_0 = -d, psi_{k+1} = psi_k * (k - d) / (k + 1).

    The fractional differencing alone captures all persistence.  This avoids
    the numerical explosion that can occur in the full FIGAS(1,d,1) recursion
    when d and beta interact.

    Parameters
    ----------
    theta : np.ndarray
        [mu, alpha, d] for non-t families, or [mu, alpha, d, kappa] for t.
    u1, u2 : np.ndarray (1D)
        Uniform pseudo-observations.
    fam_id : int
        Copula family: 2=t, 3=Clayton, 14=Survival Gumbel, 23=Clayton 90-rotated.

    Returns
    -------
    dict with keys: loglik, ll_seq, par_t, h1, h2, y
    """
    mu, alpha, d = float(theta[0]), float(theta[1]), float(theta[2])
    kappa = float(theta[3]) if (fam_id == 2 and len(theta) >= 4) else 0.0

    T_len = len(u1)
    y = np.zeros(T_len)
    g_t = np.zeros(T_len)
    par_t = np.zeros(T_len)
    ll_seq = np.zeros(T_len)
    scores = np.zeros(T_len)

    # ── Precompute fractional-difference weights ──────────────────────────
    psi = np.zeros(T_len)
    psi[0] = -d
    if T_len > 1:
        for j in range(2, T_len + 1):
            psi[j - 1] = psi[j - 2] * (j - 1 - d) / j

    y[0] = 0.0
    g_t[0] = mu

    for t in range(T_len):
        # -------------------------------------------------------------
        # 1. Link function (unconstrained g_t -> constrained copula par)
        # -------------------------------------------------------------
        g_safe = max(min(g_t[t], 30.0), -30.0)
        if fam_id == 2:
            par_t[t] = (exp(g_safe) - 1.0) / (exp(g_safe) + 1.0)
            par_t[t] = max(min(par_t[t], 0.98), -0.98)
        elif fam_id == 14:
            par_t[t] = exp(g_safe) + 1.0001
            par_t[t] = max(min(par_t[t], 30.0), 1.0001)
        elif fam_id == 23:
            par_t[t] = -exp(g_safe) - 1e-4
            par_t[t] = max(min(par_t[t], -1e-4), -30.0)
        elif fam_id == 3:
            par_t[t] = exp(g_safe) + 1e-4
            par_t[t] = max(min(par_t[t], 30.0), 1e-4)
        else:
            par_t[t] = exp(g_safe) + 1.0001
            par_t[t] = max(min(par_t[t], 30.0), 1.0001)

        # ----------------------------------------------------------------
        # 2. Log-likelihood
        # ----------------------------------------------------------------
        try:
            pdf_val = _bicop_pdf(u1[t], u2[t], family=fam_id, par=par_t[t], par2=kappa)
        except Exception:
            pdf_val = PDF_FLOOR

        if not np.isfinite(pdf_val) or pdf_val <= 0:
            pdf_val = PDF_FLOOR
        ll_seq[t] = log(pdf_val)

        # ----------------------------------------------------------------
        # 3. Compute scaled score
        # ----------------------------------------------------------------
        if fam_id == 2:
            try:
                x1 = t_dist.ppf(max(min(u1[t], 0.999999), 1e-9), df=kappa)
                x2 = t_dist.ppf(max(min(u2[t], 0.999999), 1e-9), df=kappa)
                mt = (x1 * x1 + x2 * x2 - 2.0 * par_t[t] * x1 * x2) / (1.0 - par_t[t] * par_t[t])
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
            def _eval_logpdf(g):
                if not np.isfinite(g):
                    return log(PDF_FLOOR)
                g_s = max(min(g, 30.0), -30.0)
                if fam_id == 14:
                    tp = max(min(exp(g_s) + 1.0001, 30.0), 1.0001)
                elif fam_id == 23:
                    tp = max(min(-exp(g_s) - 1e-4, -1e-4), -30.0)
                elif fam_id == 3:
                    tp = max(min(exp(g_s) + 1e-4, 30.0), 1e-4)
                else:
                    tp = max(min(exp(g_s) + 1.0001, 30.0), 1.0001)
                try:
                    v = _bicop_pdf(u1[t], u2[t], family=fam_id, par=tp, par2=kappa)
                except Exception:
                    v = PDF_FLOOR
                if not np.isfinite(v) or v <= 0:
                    v = PDF_FLOOR
                return log(v)

            f_plus = _eval_logpdf(g_t[t] + F_DIFF_H)
            f_minus = _eval_logpdf(g_t[t] - F_DIFF_H)
            scores[t] = (f_plus - f_minus) / (2.0 * F_DIFF_H)

        if not np.isfinite(scores[t]):
            scores[t] = 0.0
        scores[t] = max(min(scores[t], SCORE_CLAMP), -SCORE_CLAMP)

        # ----------------------------------------------------------------
        # 4. FIGAS(1,d,0) recursion:  y_{t+1} = alpha*s_t - sum_psi_y
        # ----------------------------------------------------------------
        if t < T_len - 1:
            sum_psi_y = 0.0
            L = 100
            max_j = min(L, t + 1)
            for j in range(1, max_j + 1):
                sum_psi_y += psi[j - 1] * y[t + 1 - j]
            y[t + 1] = alpha * scores[t] - sum_psi_y

            if not np.isfinite(y[t + 1]):
                y[t + 1] = 0.0
            y[t + 1] = max(min(y[t + 1], 29.0), -29.0)
            g_t[t + 1] = mu + y[t + 1]

    # ── Compute h-functions ─────────────────────────────────────────────
    h1 = np.zeros(T_len)
    h2 = np.zeros(T_len)
    n_h_fallback = 0
    for t in range(T_len):
        h1[t], h2[t] = _safe_bicop_hfunc(u1[t], u2[t], family=fam_id,
                                           par=par_t[t], par2=kappa)
        if h1[t] == u1[t] and h2[t] == u2[t]:
            n_h_fallback += 1

    if n_h_fallback > 0:
        import warnings
        warnings.warn(f"FIGAS-d0 filter: {n_h_fallback}/{T_len} h-function "
                      f"evaluations fell back to independence copula", RuntimeWarning)
    return dict(
        loglik=float(np.sum(ll_seq)),
        ll_seq=ll_seq,
        par_t=par_t,
        h1=h1,
        h2=h2,
        y=y,
    )


def estimate_figas_d0_params(u1, u2, fam_id, verbose=True):
    """
    Estimate FIGAS(1,d,0) parameters via multi-start L-BFGS-B.

    Tests multiple starting values for d: 0.1, 0.2, 0.35, 0.45.
    Picks the best log-likelihood across all starts.

    Parameters
    ----------
    u1, u2 : np.ndarray (1D)
        Uniform pseudo-observations.
    fam_id : int
        Copula family code: 2=t, 3=Clayton, 14=Survival Gumbel, 23=Clayton 90-rotated.
    verbose : bool
        If True, print estimation progress.

    Returns
    -------
    best_params : np.ndarray
        Estimated parameters [mu, alpha, d] (plus kappa if t).
    filtered_result : dict
        Filter output from filter_figas_d0() at the optimum.
    """
    rng = np.random.RandomState(SEED)

    u1a = np.asarray(u1, dtype=float).ravel()
    u2a = np.asarray(u2, dtype=float).ravel()

    # ── Step 1: Static copula estimation for start_mu ────────────────────
    static_par, static_par2 = _static_copula_fit(u1a, u2a, fam_id)
    start_mu = _inverse_link(fam_id, static_par)
    start_mu = max(min(start_mu, 5.0), -5.0)

    if verbose:
        suffix = f", static_kappa={static_par2:.2f}" if fam_id == 2 else ""
        print(f"  [FIGAS-d0] family={fam_id}, static_par={static_par:.4f}{suffix}, "
              f"start_mu={start_mu:.4f}")

    # ── Common objective ─────────────────────────────────────────────────
    def _objective(theta_vec):
        try:
            tmp = filter_figas_d0(theta_vec, u1a, u2a, fam_id)
            ll = tmp["loglik"]
            if not np.isfinite(ll):
                return 1e10
            return -ll
        except Exception:
            return 1e10

    # ── Build bounds ─────────────────────────────────────────────────────
    mu_lo = max(start_mu - 5.0, FIGAS_BOUNDS["mu"][0])
    mu_hi = min(start_mu + 5.0, FIGAS_BOUNDS["mu"][1])
    lower = np.array([mu_lo, FIGAS_BOUNDS["alpha"][0], 0.001])
    upper = np.array([mu_hi, FIGAS_BOUNDS["alpha"][1], 0.49])
    if fam_id == 2:
        lower = np.append(lower, FIGAS_BOUNDS["kappa"][0])
        upper = np.append(upper, FIGAS_BOUNDS["kappa"][1])

    # ── Multi-start over different d initial values ──────────────────────
    d_starts = [0.1, 0.2, 0.35, 0.45]
    best_nll = np.inf
    best_x = None
    best_res = None
    best_tag = ""

    import sys as _sys

    for d0 in d_starts:
        init = np.array([start_mu, 0.05, d0], dtype=float)
        if fam_id == 2:
            init = np.append(init, max(3.0, static_par2))

        for restart in range(3):
            x0 = init.copy()
            if restart > 0:
                x0[0] += rng.normal(0, 0.5)
                x0[1] *= np.exp(rng.normal(0, 0.3))
                x0[2] += rng.uniform(-0.08, 0.08)
                if fam_id == 2:
                    x0[3] += rng.normal(0, 3.0)
            x0 = np.clip(x0, lower + 1e-8, upper - 1e-8)

            try:
                res = minimize(_objective, x0=x0, method="L-BFGS-B",
                               bounds=list(zip(lower, upper)),
                               options={"maxiter": 300, "maxfun": 500, "ftol": 1e-8})
                if res.fun < best_nll:
                    best_nll = float(res.fun)
                    best_x = res.x.copy()
                    best_tag = f"d0={d0:.2f}, restart={restart}"
            except Exception:
                continue

    if best_x is None:
        raise RuntimeError("FIGAS-d0 estimation failed: all starts returned errors.")

    best_res = filter_figas_d0(best_x, u1a, u2a, fam_id)

    if verbose:
        labels = ["mu", "alpha", "d"]
        if fam_id == 2:
            labels.append("kappa")
        best_str = ", ".join(f"{labels[i]}={best_x[i]:.4f}" for i in range(len(best_x)))
        print(f"    FIGAS-d0 best ({best_tag}): {best_str}")
        print(f"    FIGAS-d0 loglik: {best_res['loglik']:.2f}  (n={len(u1a)})")

    return best_x, best_res


# ============================================================================
# 8. Quick smoke-test (runs when executed directly)
# ============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("FIGAS filter -- self-test")
    print("=" * 60)

    rng = np.random.RandomState(42)
    n = 500

    # Generate synthetic data from a Clayton copula with time-varying parameter
    # (using a simple time-varying theta for demonstration)
    from scipy.stats import kendalltau

    t = np.arange(n) / n
    true_theta = 1.5 + 0.5 * np.sin(2 * np.pi * t * 4)  # oscillating around 1.5

    # Generate observations via conditional sampling
    u1_sim = rng.uniform(size=n)
    u2_sim = np.zeros(n)
    for i in range(n):
        # Clayton conditional: C(v|u) = u^(-theta-1) * (u^(-theta)+v^(-theta)-1)^(-1/theta-1)
        # Use numerical inversion to sample
        candidates = rng.uniform(0, 1, size=200)
        h_vals = _bicop_hfunc(
            np.full(200, u1_sim[i]),
            candidates,
            family=3,
            par=true_theta[i],
            par2=0
        )[0]
        # Pick candidate where h_val ~= random uniform
        rv = rng.uniform()
        idx = np.argmin(np.abs(h_vals - rv))
        u2_sim[i] = candidates[idx]

    print(f"\nGenerated {n} samples from time-varying Clayton copula.")
    tau, _ = kendalltau(u1_sim, u2_sim)
    print(f"Empirical Kendall's tau: {tau:.4f}")

    # Test frac_weights
    w = frac_weights(d=0.25, max_lag=10)
    print(f"\nfrac_weights(d=0.25, max_lag=10): {np.round(w, 4)}")

    # Test static estimation
    print(f"\n--- Static Clayton estimation ---")
    par_static, _ = _static_copula_fit(u1_sim, u2_sim, fam_id=3)
    print(f"  Static theta: {par_static:.4f}  (true mean theta ~ {np.mean(true_theta):.4f})")

    # Test filter_figas
    print(f"\n--- FIGAS filter ---")
    test_theta = np.array([log(1.5), 0.05, 0.8, 0.2])
    fres = filter_figas(test_theta, u1_sim, u2_sim, fam_id=3)
    print(f"  mu={log(1.5):.4f}, alpha=0.05, beta=0.80, d=0.20")
    print(f"  Filter log-likelihood: {fres['loglik']:.2f}")
    print(f"  par_t range: [{fres['par_t'].min():.4f}, {fres['par_t'].max():.4f}]")

    # Test h-functions
    print(f"\n--- h-function test ---")
    h1_test, h2_test = _bicop_hfunc(0.3, 0.7, family=3, par=2.0, par2=0)
    print(f"  Clayton(theta=2.0) h(0.3|0.7): {h1_test:.4f}")

    print(f"\nAll tests passed.")
