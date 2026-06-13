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
from statsmodels.stats.diagnostic import acorr_ljungbox
import matplotlib
matplotlib.use("Agg")                      # non-interactive backend
import matplotlib.pyplot as plt

from arch import arch_model

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
#  1.  fit_gjr_garch
# ============================================================
def fit_gjr_garch(returns_train, assets, specs):
    """
    Fit GJR-GARCH(1,1) with Student-t distribution for each asset.

    Parameters
    ----------
    returns_train : pd.DataFrame  (n_train x n_assets)
    assets : list of str           Column names in order.
    specs : dict                   GARCH_SPECS from config.py.

    Returns
    -------
    fit_dict : dict  {asset: arch.univariate.base.ARCHModelResult}
    """
    fit_dict = {}

    for asset in assets:
        print(f"\n========== Variable: {asset} ==========")

        ar_order, _ma_order = specs[asset]["arma"]

        # Construct AR-lag list for Python's arch library.
        # NOTE: arch supports AR in the mean equation via ARX, but does NOT
        # natively expose MA terms – the MA component from the original
        # rugarch ARMA(p,q) is absorbed by the GARCH innovation filter.
        if ar_order > 0:
            ar_lags = ar_order           # int → lags 1, 2, …, p
            mean_type = "ARX"
        else:
            ar_lags = 0
            mean_type = "Constant"

        model = arch_model(
            returns_train[asset].values,
            mean=mean_type,
            lags=ar_lags,
            vol="GARCH",
            p=1, o=1, q=1,               # GJR-GARCH(1,1)
            power=2.0,
            dist="t",
        )

        result = model.fit(disp="off", options={"maxiter": 2000})

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
      1. Create an identical GJR-GARCH(1,1)-t model specification.
      2. Freeze ALL parameters to the training-set estimates via fix_params.
      3. Call .fit() which merely evaluates the likelihood at the fixed
         values, producing filtered conditional volatilities and residuals.
      4. Compute standardised residuals and PIT-transform them.

    Returns
    -------
    z_test_dict : dict         {asset: np.ndarray}
    u_test_dict : dict         {asset: np.ndarray}
    test_filter_results : dict {asset: ARCHModelResult}
    """
    z_test_dict = {}
    u_test_dict = {}
    test_filter_results = {}

    print("\n========== Processing Test Set ==========")

    for asset in fit_dict:
        print(f"\n========== Variable: {asset} (Test Set) ==========")

        ar_order, _ma_order = specs[asset]["arma"]

        if ar_order > 0:
            ar_lags = ar_order
            mean_type = "ARX"
        else:
            ar_lags = 0
            mean_type = "Constant"

        # Build the same specification as during training
        model = arch_model(
            test_returns[asset].values,
            mean=mean_type,
            lags=ar_lags,
            vol="GARCH",
            p=1, o=1, q=1,
            power=2.0,
            dist="t",
        )

        # Apply fixed parameters from training to test data (no re-estimation)
        train_params = fit_dict[asset].params.values.tolist()
        filter_result = model.fix(train_params)

        # Standardised residuals for the test sample
        z_test = filter_result.resid / filter_result.conditional_volatility
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
