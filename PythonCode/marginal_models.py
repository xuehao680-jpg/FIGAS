#!/usr/bin/env python3
"""
GJR-GARCH(1,1) + Student-t marginal distribution modeling
for 5 Belt-and-Road stock indices.

Refactored from R rugarch code (lines 189-327 of /mnt/d/zcc/Rcode/代码(1).txt).
Each asset gets its own ARMA-GJR-GARCH(1,1)-t model, followed by PIT
transformation to Uniform(0,1) for downstream Copula modeling.
"""

import warnings
import pickle

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from scipy.special import gammaln
from statsmodels.stats.diagnostic import acorr_ljungbox
import matplotlib
matplotlib.use("Agg")                      # non-interactive backend
import matplotlib.pyplot as plt

from config import *

# Suppress convergence / numerical warnings for cleaner output.
warnings.filterwarnings("ignore")


# ============================================================
#  Helper  –  custom ARCH-LM test (exact R equivalent)
# ============================================================
def arch_lm_test(residuals, lags=5):
    """
    Custom ARCH-LM test matching R's arch_lm_test() exactly.

    H0: no ARCH effects (homoskedasticity)
    HA: ARCH effects present

    Parameters
    ----------
    residuals : array_like
        Standardised (or raw) residuals.
    lags : int
        Number of lags to include in the auxiliary regression.

    Returns
    -------
    LM : float
        Test statistic  LM = (n - lags) * R^2.
    p_value : float
        p-value from chi^2(lags).
    """
    z = np.asarray(residuals, dtype=float)
    z = np.where(np.isfinite(z), z, 0.0)   # replace NaN/Inf with 0
    z2 = z ** 2
    n = len(z2)

    # Build lagged squared-residual matrix  (same NA-padding as R)
    X = np.column_stack([
        np.concatenate([np.full(i, np.nan), z2[:-i]])
        for i in range(1, lags + 1)
    ])

    # Drop rows containing NaN  (observations 1 … lags)
    start = lags
    y_reg = z2[start:]                     # length = n - lags
    X_reg = X[start:, :]                   # (n - lags) x lags

    # Remove any remaining NaN/Inf rows  (safety for degenerate fits)
    valid_mask = np.isfinite(y_reg) & np.all(np.isfinite(X_reg), axis=1)
    y_reg = y_reg[valid_mask]
    X_reg = X_reg[valid_mask, :]

    # Guard against too few observations
    if len(y_reg) < lags + 2:
        return np.nan, 1.0

    # OLS with intercept
    Xc = np.column_stack([np.ones(len(X_reg)), X_reg])
    try:
        beta, _resid, _rank, _sv = np.linalg.lstsq(Xc, y_reg, rcond=None)
    except np.linalg.LinAlgError:
        return np.nan, 1.0

    y_hat = Xc @ beta
    ss_res = np.sum((y_reg - y_hat) ** 2)
    ss_tot = np.sum((y_reg - np.mean(y_reg)) ** 2)
    r_squared = 1.0 - ss_res / ss_tot

    n_eff = len(y_reg)                     # = n - lags
    LM = n_eff * r_squared
    p_value = 1.0 - stats.chi2.cdf(LM, lags)

    return LM, p_value


# ============================================================
#  ARMAGARCHResult  –  Drop-in replacement for ARCHModelResult
# ============================================================
class ARMAGARCHResult:
    """Wrapper around custom ARMA-GJR-GARCH-t MLE results.

    Mimics arch.univariate.base.ARCHModelResult interface so
    downstream code (residual extraction, PIT, saving) works unchanged.
    """

    def __init__(self, params, param_names, conditional_volatility, resid):
        self.params = pd.Series(
            np.asarray(params, dtype=float),
            index=list(param_names),
        )
        self.conditional_volatility = np.asarray(conditional_volatility, dtype=float)
        self.resid = np.asarray(resid, dtype=float)


# ============================================================
#  Core recursion  –  ARMA(p,q) + GJR-GARCH(1,1)
# ============================================================
def _compute_arma_gjr_garch(returns, mu, ar, ma, omega, alpha, gamma, beta):
    """Compute residuals and conditional volatilities for ARMA-GJR-GARCH.

    Parameters
    ----------
    returns : ndarray (n,)
        Observed returns  r_0 … r_{n-1}.
    mu      : float
        Mean intercept.
    ar      : ndarray (p,)
        AR coefficients  [φ₁, …, φₚ]  (empty if p = 0).
    ma      : ndarray (q,)
        MA coefficients  [θ₁, …, θ_q]  (empty if q = 0).
    omega, alpha, gamma, beta : float
        GJR-GARCH(1,1) parameters.

    Returns
    -------
    epsilon : ndarray (n,)
        Raw residuals  ε_t = r_t - μ_t.
    sigma   : ndarray (n,)
        Conditional standard deviation  σ_t.
    z       : ndarray (n,)
        Standardised residuals  z_t = ε_t / σ_t.
    """
    n = len(returns)
    p = len(ar)
    q = len(ma)

    # ---- ARMA residuals  (ε_t = r_t - μ - Σ φ_i r_{t-i} - Σ θ_j ε_{t-j}) ----
    epsilon = np.zeros(n)
    for t in range(n):
        mu_t = mu
        # AR terms — use lagged *returns*
        for i in range(p):
            if t - 1 - i >= 0:
                mu_t += ar[i] * returns[t - 1 - i]
        # MA terms — use lagged *residuals*
        for j in range(q):
            if t - 1 - j >= 0:
                mu_t += ma[j] * epsilon[t - 1 - j]
        epsilon[t] = returns[t] - mu_t

    # ---- GJR-GARCH(1,1) conditional variance -------------------------------
    sigma2 = np.zeros(n)
    # Initialise with sample variance (robust)
    sigma2[0] = float(np.var(returns)) if np.var(returns) > 1e-12 else 1e-6

    for t in range(1, n):
        e_prev = epsilon[t - 1]
        I_neg = 1.0 if e_prev < 0 else 0.0
        sigma2[t] = (omega
                     + alpha * e_prev ** 2
                     + gamma * I_neg * e_prev ** 2
                     + beta * sigma2[t - 1])

    sigma2 = np.maximum(sigma2, 1e-12)
    sigma = np.sqrt(sigma2)

    # Standardised innovations
    z = np.where(sigma > 1e-12, epsilon / sigma, 0.0)

    return epsilon, sigma, z


# ============================================================
#  Negative log-likelihood  –  ARMA-GJR-GARCH(1,1)-Student-t
# ============================================================
def _arma_gjr_garch_t_nll(params, returns, p, q):
    """Negative log-likelihood for ARMA(p,q)-GJR-GARCH(1,1) with Student-t.

    Parameter vector layout:
      [mu, φ₁ … φₚ, θ₁ … θ_q, ω, α, γ, β, ν]

    Parameters
    ----------
    params  : ndarray   Parameter vector.
    returns : ndarray   Return series (n,).
    p, q    : int       AR and MA orders.

    Returns
    -------
    nll : float   Negative log-likelihood (→ ∞ on invalid parameters).
    """
    # ---- unpack -----------------------------------------------------------
    mu = params[0]
    ar = params[1:1 + p] if p > 0 else np.array([])
    ma = params[1 + p:1 + p + q] if q > 0 else np.array([])
    omega = params[1 + p + q]
    alpha = params[1 + p + q + 1]
    gamma = params[1 + p + q + 2]
    beta = params[1 + p + q + 3]
    nu = params[1 + p + q + 4]

    # ---- soft parameter checks --------------------------------------------
    if omega <= 1e-12 or alpha < 0 or gamma < 0 or beta < 0 or nu <= 2.01:
        return 1e15

    # Covariance stationarity heuristic (GJR-GARCH with symmetric innovations)
    if alpha + 0.5 * gamma + beta >= 1.0:
        return 1e15

    # AR / MA near-unit-root penalty
    penalty = 0.0
    for phi in ar:
        if abs(phi) >= 0.999:
            penalty += 1e8 * (abs(phi) - 0.99) ** 2
    for theta in ma:
        if abs(theta) >= 0.999:
            penalty += 1e8 * (abs(theta) - 0.99) ** 2

    # ---- compute residuals & volatility -----------------------------------
    epsilon, sigma, z = _compute_arma_gjr_garch(
        returns, mu, ar, ma, omega, alpha, gamma, beta,
    )

    # ---- burn-in ----------------------------------------------------------
    start = max(p, q, 1)
    z_use = z[start:]
    sigma_use = sigma[start:]

    if len(z_use) < 10:
        return 1e15

    # ---- Student-t log-likelihood -----------------------------------------
    nu_safe = max(nu, 2.01)
    const = gammaln((nu_safe + 1.0) / 2.0) - gammaln(nu_safe / 2.0) \
            - 0.5 * np.log(np.pi * (nu_safe - 2.0))

    # Include Jacobian term:  f(r_t) = f_z(z_t) / σ_t
    ll = const - np.log(sigma_use) \
         - (nu_safe + 1.0) / 2.0 * np.log(1.0 + z_use ** 2 / (nu_safe - 2.0))

    nll = -np.sum(ll)

    if not np.isfinite(nll):
        return 1e15

    return nll + penalty


# ============================================================
#  Single-asset fitting  –  multi-start L-BFGS-B
# ============================================================
def _fit_single_arma_gjr_garch_t(returns, p, q, n_restarts=5):
    """Fit ARMA(p,q)-GJR-GARCH(1,1)-t by joint MLE.

    Uses L-BFGS-B with multiple random starting points to avoid
    local optima.  Returns an ARMAGARCHResult.
    """
    n = len(returns)
    ret_var = float(np.var(returns))
    ret_mean = float(np.mean(returns))

    # ---- default starting point -------------------------------------------
    mu0 = ret_mean
    ar0 = np.zeros(p)
    ma0 = np.zeros(q)
    omega0 = max(ret_var * 0.05, 1e-6)
    alpha0 = 0.05
    gamma0 = 0.05
    beta0 = 0.85
    nu0 = 10.0

    x0 = np.concatenate([
        [mu0], ar0, ma0,
        [omega0, alpha0, gamma0, beta0, nu0],
    ])

    # ---- bounds -----------------------------------------------------------
    bounds = []
    bounds.append((-3.0 * np.sqrt(ret_var), 3.0 * np.sqrt(ret_var)))  # mu
    for _ in range(p):
        bounds.append((-0.999, 0.999))   # φ (stationarity)
    for _ in range(q):
        bounds.append((-0.999, 0.999))   # θ (invertibility)
    bounds.append((1e-12, ret_var * 2.0))  # ω
    bounds.append((1e-12, 0.49))           # α
    bounds.append((1e-12, 0.49))           # γ
    bounds.append((1e-12, 0.999))          # β
    bounds.append((2.1, 50.0))             # ν

    # ---- multi-start optimisation -----------------------------------------
    rng = np.random.RandomState(SEED)
    best_nll = np.inf
    best_x = x0.copy()

    for restart in range(n_restarts):
        if restart == 0:
            x_init = x0.copy()
        else:
            x_init = x0.copy()
            # perturb positive parameters multiplicatively
            for idx in [1 + p + q,       # omega
                        1 + p + q + 1,   # alpha
                        1 + p + q + 2,   # gamma
                        1 + p + q + 3]:  # beta
                x_init[idx] *= np.exp(rng.normal(0, 0.2))
                x_init[idx] = np.clip(x_init[idx], bounds[idx][0] + 1e-8,
                                      min(bounds[idx][1], 0.999) - 1e-8
                                      if bounds[idx][1] is not None else 0.999)
            # perturb nu
            x_init[1 + p + q + 4] = rng.uniform(3.0, 20.0)
            # perturb mu additively
            x_init[0] += rng.normal(0, 0.3 * np.sqrt(ret_var))
            x_init[0] = np.clip(x_init[0], bounds[0][0], bounds[0][1])
            # perturb AR/MA coefficients
            for idx in range(1, 1 + p + q):
                x_init[idx] += rng.normal(0, 0.05)
                x_init[idx] = np.clip(x_init[idx], bounds[idx][0], bounds[idx][1])

            # clip all to bounds strictly
            for i, (lo, hi) in enumerate(bounds):
                lo_val = -np.inf if lo is None else lo
                hi_val = np.inf if hi is None else hi
                x_init[i] = np.clip(x_init[i],
                                    lo_val + 1e-8,
                                    hi_val - 1e-8 if np.isfinite(hi_val) else hi_val)

        try:
            res = minimize(
                _arma_gjr_garch_t_nll,
                x_init,
                args=(returns, p, q),
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 5000, "ftol": 1e-12, "gtol": 1e-8},
            )

            if res.fun < best_nll and np.isfinite(res.fun):
                best_nll = res.fun
                best_x = res.x
        except Exception:
            continue

    # ---- build result -----------------------------------------------------
    mu_opt = best_x[0]
    ar_opt = best_x[1:1 + p] if p > 0 else np.array([])
    ma_opt = best_x[1 + p:1 + p + q] if q > 0 else np.array([])
    omega_opt = best_x[1 + p + q]
    alpha_opt = best_x[1 + p + q + 1]
    gamma_opt = best_x[1 + p + q + 2]
    beta_opt = best_x[1 + p + q + 3]
    nu_opt = best_x[1 + p + q + 4]

    epsilon, sigma, z = _compute_arma_gjr_garch(
        returns, mu_opt, ar_opt, ma_opt,
        omega_opt, alpha_opt, gamma_opt, beta_opt,
    )

    param_names = ["mu"]
    for i in range(p):
        param_names.append(f"ar{i + 1}")
    for i in range(q):
        param_names.append(f"ma{i + 1}")
    param_names.extend(["omega", "alpha1", "gamma1", "beta1", "nu"])

    param_values = np.concatenate([
        [mu_opt], ar_opt, ma_opt,
        [omega_opt, alpha_opt, gamma_opt, beta_opt, nu_opt],
    ])

    return ARMAGARCHResult(param_values, param_names, sigma, epsilon)


# ============================================================
#  Filter (fixed-parameter evaluation)  –  test set
# ============================================================
def _filter_single_arma_gjr_garch_t(returns, params, p, q):
    """Evaluate ARMA(p,q)-GJR-GARCH(1,1)-t with FIXED parameters.

    No optimisation — equivalent to ugarchfilter in rugarch /
    model.fix() in Python arch.

    Parameters
    ----------
    returns : ndarray (n,)   Return series.
    params  : list / ndarray Parameter vector in canonical order.
    p, q    : int            AR and MA orders.

    Returns
    -------
    ARMAGARCHResult
    """
    params = np.asarray(params, dtype=float)

    mu = params[0]
    ar = params[1:1 + p] if p > 0 else np.array([])
    ma = params[1 + p:1 + p + q] if q > 0 else np.array([])
    omega = max(params[1 + p + q], 1e-12)
    alpha = max(params[1 + p + q + 1], 0.0)
    gamma = max(params[1 + p + q + 2], 0.0)
    beta = max(params[1 + p + q + 3], 0.0)
    nu = params[1 + p + q + 4]

    epsilon, sigma, z = _compute_arma_gjr_garch(
        returns, mu, ar, ma, omega, alpha, gamma, beta,
    )

    param_names = ["mu"]
    for i in range(p):
        param_names.append(f"ar{i + 1}")
    for i in range(q):
        param_names.append(f"ma{i + 1}")
    param_names.extend(["omega", "alpha1", "gamma1", "beta1", "nu"])

    param_values = np.concatenate([
        [mu], ar, ma, [omega, alpha, gamma, beta, nu],
    ])

    return ARMAGARCHResult(param_values, param_names, sigma, epsilon)


# ============================================================
#  1.  fit_gjr_garch
# ============================================================
def fit_gjr_garch(returns_train, assets, specs):
    """
    Fit ARMA(p,q)-GJR-GARCH(1,1) with Student-t distribution for each asset
    using custom joint maximum likelihood estimation.

    Parameters
    ----------
    returns_train : pd.DataFrame  (n_train x n_assets)
    assets : list of str           Column names in order.
    specs : dict                   GARCH_SPECS from config.py.

    Returns
    -------
    fit_dict : dict  {asset: ARMAGARCHResult}
    """
    fit_dict = {}

    for asset in assets:
        print(f"\n========== Variable: {asset} ==========")

        p, q = specs[asset]["arma"]  # both AR and MA orders are used

        result = _fit_single_arma_gjr_garch_t(
            returns_train[asset].values.astype(float), p, q,
        )

        # Print coefficients rounded to 6 decimals  (mirrors R output)
        print(round(result.params, 6))

        fit_dict[asset] = result

    return fit_dict


# ============================================================
#  2.  extract_std_residuals
# ============================================================
def extract_std_residuals(fit_dict):
    """
    Extract standardised residuals z = resid / conditional_volatility.

    Returns
    -------
    z_dict : dict       {asset: np.ndarray}
    shape_dict : dict   {asset: float}    Student-t dof  (nu).
    """
    z_dict = {}
    shape_dict = {}

    for asset, fit in fit_dict.items():
        vol = fit.conditional_volatility
        # Guard against zero / NaN conditional volatility
        vol = np.where(np.isfinite(vol) & (vol > 1e-12), vol, 1.0)
        z = fit.resid / vol
        z = np.where(np.isfinite(z), z, 0.0)   # replace NaN/Inf with 0
        z_dict[asset] = z
        shape_dict[asset] = fit.params["nu"]

    return z_dict, shape_dict


# ============================================================
#  3.  pit_transform
# ============================================================
def pit_transform(z_dict, shape_dict):
    """
    Probability Integral Transform:  z ~ t(nu)  →  u ~ Uniform(0,1).

    Uses scipy.stats.t.cdf with the asset-specific degrees of freedom
    obtained from the GARCH fit.

    Returns
    -------
    u_dict : dict  {asset: np.ndarray}
    """
    u_dict = {}
    for asset in z_dict:
        z = z_dict[asset]
        nu = shape_dict[asset]
        u_dict[asset] = stats.t.cdf(z, df=nu)
    return u_dict


# ============================================================
#  4.  residual_diagnostics
# ============================================================
def residual_diagnostics(z_dict):
    """
    Ljung-Box and ARCH-LM tests on standardised residuals.
    """
    for asset, z in z_dict.items():
        print(f"\n--- Residual Diagnostics: {asset} ---")

        # Clean NaN/Inf
        z = np.where(np.isfinite(z), z, 0.0)

        # ----- Ljung-Box -----
        for lag in [5, 10, 15]:
            lb = acorr_ljungbox(z, lags=[lag], model_df=0, return_df=True)
            p_val = lb["lb_pvalue"].values[0]
            if not np.isfinite(p_val):
                p_val = 0.0
            mark = "sqrt" if p_val > 0.05 else "x"
            print(f"LB lag{lag}: p={p_val:.4f} {mark}")

        # ----- ARCH-LM -----
        for lag in [5, 10]:
            _lm_stat, p_val = arch_lm_test(z, lags=lag)
            if not np.isfinite(p_val):
                p_val = 0.0
            mark = "sqrt" if p_val > 0.05 else "x"
            print(f"ARCH lag{lag}: p={p_val:.4f} {mark}")


# ============================================================
#  5.  pit_uniformity_check
# ============================================================
def pit_uniformity_check(u_dict, show_plots=True):
    """
    Plot PIT histograms with Uniform(0,1) reference line and run KS tests.

    For each asset the histogram is drawn with a red-dashed line at y=1
    representing the ideal U(0,1) density.  A one-sample KS test against
    Uniform(0,1) is printed below the plot.

    Parameters
    ----------
    u_dict : dict   {asset: np.ndarray of U(0,1) values}
    show_plots : bool
        If False, plots are saved to disk instead of being displayed.
    """
    for asset, u in u_dict.items():
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(
            u, bins=30, density=True, alpha=0.7,
            color="lightblue", edgecolor="black", linewidth=0.5,
        )
        ax.axhline(y=1.0, linestyle="--", color="red", linewidth=2)
        ax.set_title(f"{asset} PIT Uniformity Check")
        ax.set_xlabel("u")
        ax.set_ylabel("Density")
        fig.tight_layout()

        if show_plots:
            plt.show()
        else:
            fname = OUTPUT_DIR / f"pit_hist_{asset}.png"
            fig.savefig(fname, dpi=150)
            plt.close(fig)

        # KS test against Uniform(0,1)
        ks_stat, ks_pval = stats.kstest(u, "uniform")
        print(f"\n{asset} PIT Uniformity KS p-value: {ks_pval:.6f}")


# ============================================================
#  6.  filter_test_set
# ============================================================
def filter_test_set(fit_dict, test_returns, specs, shape_dict):
    """
    Filter the test set using parameters estimated on the training set
    (NO re-estimation).  Equivalent to ugarchfilter in rugarch.

    For each asset:
      1. Extract training parameters.
      2. Compute residuals and conditional volatilities on test data
         using fixed parameters via _filter_single_arma_gjr_garch_t.
      3. Compute standardised residuals and PIT-transform them.

    Returns
    -------
    z_test_dict : dict         {asset: np.ndarray}
    u_test_dict : dict         {asset: np.ndarray}
    test_filter_results : dict {asset: ARMAGARCHResult}
    """
    z_test_dict = {}
    u_test_dict = {}
    test_filter_results = {}

    print("\n========== Processing Test Set ==========")

    for asset in fit_dict:
        print(f"\n========== Variable: {asset} (Test Set) ==========")

        p, q = specs[asset]["arma"]

        # Get training parameters as flat list (canonical order)
        train_params = fit_dict[asset].params.values.tolist()

        # Filter test data with fixed parameters
        filter_result = _filter_single_arma_gjr_garch_t(
            test_returns[asset].values.astype(float), train_params, p, q,
        )

        # Standardised residuals for the test sample
        z_test = filter_result.resid / np.maximum(
            filter_result.conditional_volatility, 1e-12,
        )
        z_test = np.where(np.isfinite(z_test), z_test, 0.0)
        z_test_dict[asset] = z_test

        # PIT using the *training* shape parameter
        nu = shape_dict[asset]
        u_test = stats.t.cdf(z_test, df=nu)
        u_test_dict[asset] = u_test

        test_filter_results[asset] = filter_result

    print("\n========== Test Set Processing Complete ==========")
    return z_test_dict, u_test_dict, test_filter_results


# ============================================================
#  7.  save_marginal_outputs
# ============================================================
def save_marginal_outputs(u_train_dict, u_test_dict, fit_dict,
                          test_filter_results):
    """
    Persist all marginal-model artefacts to disk.

    Writes
    ------
    *  u_train_matrix  →  U_LIST_CSV   (1944 x 5)
    *  u_test_matrix   →  U_TEST_CSV   (486 x 5)
    *  Full object bundle → MODEL_PKL  (replaces .RData)
    """
    # --- Build DataFrames in the canonical column order ---------------
    var_names = ["hushen300", "ydlMIB", "xxlNZ50", "nfFTSE", "bxBOVESPA"]

    u_train_matrix = pd.DataFrame(u_train_dict)[var_names]
    u_test_matrix  = pd.DataFrame(u_test_dict)[var_names]

    # --- CSVs ---------------------------------------------------------
    u_train_matrix.to_csv(U_LIST_CSV, index=False)
    u_test_matrix.to_csv(U_TEST_CSV, index=False)
    print(f"\nTraining U matrix  →  {U_LIST_CSV}")
    print(f"Test U matrix      →  {U_TEST_CSV}")

    # --- Pickle  (structure mirrors R's RData) -----------------------
    fit_list_ordered = {name: fit_dict[name] for name in var_names}
    test_list_ordered = ({name: test_filter_results[name]
                          for name in var_names}
                         if test_filter_results is not None else None)

    bundle = {
        "version":         2,  # v1 = arch library, v2 = custom joint MLE
        "u_train_matrix":  u_train_matrix,
        "u_test_matrix":   u_test_matrix,
        "fit_list":        fit_list_ordered,
        "test_fit_list":   test_list_ordered,
        "var_names":       var_names,
    }

    with open(MODEL_PKL, "wb") as f:
        pickle.dump(bundle, f)

    print(f"Full marginal data  →  {MODEL_PKL}")


# ============================================================
#  Main  (executed when the file is run as a script)
# ============================================================
if __name__ == "__main__":
    ensure_dirs()

    # ---- 0.  Load & split data ---------------------------------------
    print("Loading data …")
    data = pd.read_csv(DATA_CSV)
    returns_full = data[ASSETS].dropna()

    n = len(returns_full)
    train_size = int(n * TRAIN_RATIO)          # ≈ 1944

    returns_train = returns_full.iloc[:train_size]
    returns_test  = returns_full.iloc[train_size:]

    print(f"Full sample      : {n} observations")
    print(f"Training set     : {returns_train.shape}")
    print(f"Test set         : {returns_test.shape}")

    # ---- 1.  Fit GJR-GARCH(1,1)-t models -----------------------------
    fit_dict = fit_gjr_garch(returns_train, ASSETS, GARCH_SPECS)

    # ---- 2.  Standardised residuals ----------------------------------
    z_dict, shape_dict = extract_std_residuals(fit_dict)

    # ---- 3.  Residual diagnostics ------------------------------------
    residual_diagnostics(z_dict)

    # ---- 4.  PIT → Uniform(0,1) -------------------------------------
    u_dict = pit_transform(z_dict, shape_dict)

    # ---- 5.  PIT uniformity check  (histograms + KS) -----------------
    pit_uniformity_check(u_dict, show_plots=False)

    # ---- 6.  Filter test set -----------------------------------------
    z_test_dict, u_test_dict, test_filter_results = filter_test_set(
        fit_dict, returns_test, GARCH_SPECS, shape_dict
    )

    # ---- 7.  Persist everything --------------------------------------
    save_marginal_outputs(u_dict, u_test_dict, fit_dict,
                          test_filter_results)

    print("\n========== All Processing Complete ==========")
