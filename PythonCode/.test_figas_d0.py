#!/usr/bin/env python3
"""
Test script for FIGAS(1,d,0) model -- beta forced to 0.

Generates synthetic data from a t-copula with time-varying parameter driven
by a FIGAS(1,d,0) recursion with d=0.3.  Fits both GAS(1,1) and FIGAS(1,d,0)
and compares the results.  FIGAS should beat GAS because the true DGP has
long memory (d > 0).
"""

import sys
import os
import numpy as np
from numpy import log, exp, sqrt
from scipy.stats import t as t_dist
from scipy.stats import norm

# ═══════════════════════════════════════════════════════════════════════════
# Ensure the project root is on sys.path so we can import local modules.
# ═══════════════════════════════════════════════════════════════════════════
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from figas_filter import (
    filter_figas_d0,
    estimate_figas_d0_params,
    _bicop_pdf,
    _bicop_hfunc,
    _static_copula_fit,
    _inverse_link,
    PDF_FLOOR, SCORE_CLAMP, F_DIFF_H, SEED,
)
from gas_filter import filter_gas, estimate_gas_params


# ═══════════════════════════════════════════════════════════════════════════
# 1. Simulate data from FIGAS(1,d,0) DGP
# ═══════════════════════════════════════════════════════════════════════════

def simulate_figas_d0_dgp(true_params, T, fam_id, seed=42):
    """
    Simulate observations from a FIGAS(1,d,0) model.

    The data-generating process:
      1. At each t, compute copula parameter par_t from g_t via link.
      2. Sample (u1[t], u2[t]) from the copula with parameter par_t[t].
      3. Compute the score of the log-likelihood.
      4. Update y[t+1] = alpha*s_t - sum_psi_y  (beta = 0).
      5. Set g[t+1] = mu + y[t+1].

    Parameters
    ----------
    true_params : list
        [mu, alpha, d] for non-t families, or [mu, alpha, d, kappa] for t.
    T : int
        Number of observations.
    fam_id : int
        Copula family (must be 2 = t-copula for this test).
    seed : int
        Random seed.

    Returns
    -------
    u1, u2 : np.ndarray
        Simulated pseudo-observations.
    true_par : np.ndarray
        True time-varying copula parameter path.
    """
    rng = np.random.RandomState(seed)
    mu, alpha, d = float(true_params[0]), float(true_params[1]), float(true_params[2])
    kappa = float(true_params[3]) if (fam_id == 2 and len(true_params) >= 4) else 0.0

    # ── Precompute fractional-difference weights ─────────────────────────
    psi = np.zeros(T)
    psi[0] = -d
    for j in range(2, T + 1):
        psi[j - 1] = psi[j - 2] * (j - 1 - d) / j

    y = np.zeros(T)
    g_t = np.zeros(T)
    par_t = np.zeros(T)
    scores = np.zeros(T)
    u1 = np.zeros(T)
    u2 = np.zeros(T)

    y[0] = 0.0
    g_t[0] = mu

    for t in range(T):
        # -------------------------------------------------------------
        # 1. Link function: g_t -> copula parameter
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

        # -------------------------------------------------------------
        # 2. Sample from copula with parameter par_t[t]
        # -------------------------------------------------------------
        if fam_id == 2:
            rho = par_t[t]
            # Sample via bivariate t decomposition:
            # x1 = Z1, x2 = rho*Z1 + sqrt(1-rho^2)*Z2  (normal copula base)
            # Then convert to t via chi-square scaling:
            #   t_nu = Z / sqrt(chi2_nu / nu)
            Z1 = rng.normal(0, 1)
            Z2 = rng.normal(0, 1)
            z2 = rho * Z1 + sqrt(max(1 - rho * rho, 0)) * Z2
            # t-scaling
            w = rng.chisquare(kappa) / kappa
            w = max(w, 1e-10)
            x1 = Z1 / sqrt(w)
            x2 = z2 / sqrt(w)
            u1[t] = float(t_dist.cdf(x1, df=kappa))
            u2[t] = float(t_dist.cdf(x2, df=kappa))
        else:
            # Not supported in this test — fallback
            u1[t] = rng.uniform(0, 1)
            u2[t] = rng.uniform(0, 1)

        # Force strictly inside (0,1)
        u1[t] = max(min(u1[t], 0.999999), 1e-9)
        u2[t] = max(min(u2[t], 0.999999), 1e-9)

        # -------------------------------------------------------------
        # 3. Compute score for the current observation
        # -------------------------------------------------------------
        if fam_id == 2:
            try:
                x1_t = t_dist.ppf(u1[t], df=kappa)
                x2_t = t_dist.ppf(u2[t], df=kappa)
                mt = (x1_t * x1_t + x2_t * x2_t - 2.0 * par_t[t] * x1_t * x2_t) / (
                    1.0 - par_t[t] * par_t[t]
                )
                pit = (kappa + 2.0) / (kappa + mt)
                g_s = max(min(g_t[t], 30.0), -30.0)
                dot_rho = 2.0 * exp(g_s) / (exp(g_s) + 1.0) ** 2
                denom_nabla = (1.0 - par_t[t] * par_t[t]) ** 2
                nabla = (dot_rho / denom_nabla) * (
                    (1.0 + par_t[t] * par_t[t]) * (pit * x1_t * x2_t - par_t[t])
                    - par_t[t] * (pit * x1_t * x1_t + pit * x2_t * x2_t - 2.0)
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
                g_s_ = max(min(g, 30.0), -30.0)
                if fam_id == 14:
                    tp = max(min(exp(g_s_) + 1.0001, 30.0), 1.0001)
                elif fam_id == 23:
                    tp = max(min(-exp(g_s_) - 1e-4, -1e-4), -30.0)
                elif fam_id == 3:
                    tp = max(min(exp(g_s_) + 1e-4, 30.0), 1e-4)
                else:
                    tp = max(min(exp(g_s_) + 1.0001, 30.0), 1.0001)
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

        # -------------------------------------------------------------
        # 4. FIGAS(1,d,0) recursion: y_{t+1} = alpha*s_t - sum_psi_y
        # -------------------------------------------------------------
        if t < T - 1:
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

    return u1, u2, par_t


# ═══════════════════════════════════════════════════════════════════════════
# 2. Main test
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("FIGAS(1,d,0) vs GAS(1,1) -- Long-memory DGP test")
    print("=" * 70)

    # ── DGP parameters ───────────────────────────────────────────────────
    FAM_ID = 2                    # t-copula
    T = 1500                      # sample size (larger to help identify d)
    TRUE_MU = 0.8                 # unconstrained intercept (gives rho ~ 0.38)
    TRUE_ALPHA = 0.35             # score sensitivity (stronger signal)
    TRUE_D = 0.40                 # long-memory parameter (strong)
    TRUE_KAPPA = 4.0              # t-copula df (heavier tails = more volatile scores)

    true_params = [TRUE_MU, TRUE_ALPHA, TRUE_D, TRUE_KAPPA]

    print(f"\nDGP: FIGAS(1,d,0) t-copula  (d={TRUE_D}, T={T})")
    print(f"  True params:  mu={TRUE_MU}, alpha={TRUE_ALPHA}, "
          f"d={TRUE_D}, kappa={TRUE_KAPPA}")
    print(f"  (beta=0 by construction)")
    print()

    # ── Simulate data ────────────────────────────────────────────────────
    print("Simulating data from FIGAS(1,d,0) DGP ...")
    u1_sim, u2_sim, true_par = simulate_figas_d0_dgp(
        true_params, T, fam_id=FAM_ID, seed=SEED
    )
    print(f"  Generated {T} observations.")
    print(f"  True par_t range: [{true_par.min():.4f}, {true_par.max():.4f}]")
    print(f"  True par_t mean:  {true_par.mean():.4f}")
    print(f"  True par_t std:   {true_par.std():.4f}")

    # ── Compute empirical Kendall's tau ──────────────────────────────────
    from scipy.stats import kendalltau
    tau_emp, _ = kendalltau(u1_sim, u2_sim)
    print(f"  Empirical Kendall's tau: {tau_emp:.4f}")

    # ── Static copula baseline ───────────────────────────────────────────
    static_par, static_kappa = _static_copula_fit(u1_sim, u2_sim, fam_id=FAM_ID)
    static_ll = np.sum(np.log(np.maximum(
        _bicop_pdf(u1_sim, u2_sim, family=FAM_ID, par=static_par, par2=static_kappa),
        PDF_FLOOR
    )))
    print(f"\n  Static t-copula MLE:  rho={static_par:.4f}, "
          f"kappa={static_kappa:.2f}")
    print(f"  Static loglik:         {static_ll:.2f}")

    # ══════════════════════════════════════════════════════════════════════
    # 3. Fit GAS(1,1)
    # ══════════════════════════════════════════════════════════════════════
    print()
    print("-" * 70)
    print("Fitting GAS(1,1) model ...")
    print("-" * 70)

    gas_params, gas_res = estimate_gas_params(u1_sim, u2_sim, fam_id=FAM_ID, verbose=True)
    gas_ll = gas_res["loglik"]
    gas_aic = -2 * gas_ll + 2 * len(gas_params)

    # ══════════════════════════════════════════════════════════════════════
    # 4. Fit FIGAS(1,d,0)
    # ══════════════════════════════════════════════════════════════════════
    print()
    print("-" * 70)
    print("Fitting FIGAS(1,d,0) model ...")
    print("-" * 70)

    figas_params, figas_res = estimate_figas_d0_params(
        u1_sim, u2_sim, fam_id=FAM_ID, verbose=True
    )
    figas_ll = figas_res["loglik"]
    figas_aic = -2 * figas_ll + 2 * len(figas_params)

    # ══════════════════════════════════════════════════════════════════════
    # 5. Comparison
    # ══════════════════════════════════════════════════════════════════════
    print()
    print("=" * 70)
    print("RESULTS COMPARISON")
    print("=" * 70)

    print(f"\n{'Model':<20} {'Params':<35} {'LogLik':>12} {'AIC':>12} {'n_params':>10}")
    print("-" * 90)

    gas_labels = ["mu", "alpha", "beta", "kappa"]
    gas_str = ", ".join(f"{gas_labels[i]}={gas_params[i]:.4f}"
                        for i in range(len(gas_params)))
    print(f"{'GAS(1,1)':<20} {gas_str:<35} {gas_ll:>12.2f} {gas_aic:>12.2f} "
          f"{len(gas_params):>10}")

    figas_labels = ["mu", "alpha", "d", "kappa"]
    figas_str = ", ".join(f"{figas_labels[i]}={figas_params[i]:.4f}"
                          for i in range(len(figas_params)))
    print(f"{'FIGAS(1,d,0)':<20} {figas_str:<35} {figas_ll:>12.2f} {figas_aic:>12.2f} "
          f"{len(figas_params):>10}")

    # ── Interpretation ───────────────────────────────────────────────────
    delta_ll = figas_ll - gas_ll
    delta_aic = figas_aic - gas_aic

    print()
    print("-" * 70)
    if delta_ll > 0 and delta_aic < 0:
        print(f"*** FIGAS(1,d,0) BEATS GAS(1,1) ***")
        print(f"  Log-likelihood gain: +{delta_ll:.2f}")
        print(f"  AIC improvement:     {delta_aic:.2f}")
    elif delta_ll > 0:
        print(f"FIGAS(1,d,0) has better log-likelihood (+{delta_ll:.2f}) "
              f"but AIC penalty is positive ({delta_aic:.2f})")
    else:
        print(f"GAS(1,1) outperforms FIGAS(1,d,0).")
        print(f"  Delta loglik: {delta_ll:.2f}")

    print(f"\nTrue d = {TRUE_D}, Estimated d = {figas_params[2]:.4f}")
    d_error = abs(figas_params[2] - TRUE_D)
    print(f"|d_hat - d_true| = {d_error:.4f}")
    if d_error < 0.15:
        print("  d estimate is reasonably close to the true value.")
    else:
        print("  d estimate deviates from the true value (but may still be useful).")

    # ── Quick check: is estimated FIGAS-d0 d actually > 0? ──────────────
    if figas_params[2] < 0.01:
        print("\n  WARNING: Estimated d ~ 0. FIGAS collapsed to essentially no")
        print("  long memory. This may indicate identifiability issues or that")
        print("  the DGP was not sufficiently long-memory for the given T.")

    print()
    print("=" * 70)
    print("Test complete.")
    print("=" * 70)

    # ── Return results as exit code indicator ────────────────────────────
    return 0 if delta_ll > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
