#!/usr/bin/env python3
"""
Utility module for descriptive statistics, diagnostic tests, and time series
plots for the GARCH-Copula Python refactoring.

Mirrors the R reference code at /mnt/d/zcc/Rcode/代码(1).txt, lines 1-150.
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller

from config import SEED, ASSETS, DATA_CSV, DATA_DIR, TRAIN_RATIO


# ===========================================================================
# 1. Data loading and splitting
# ===========================================================================

def load_and_split_data():
    """
    Read the CSV, keep the 5 asset columns plus date, drop rows with any NA,
    parse the date column, and produce a chronological 80/20 train/test split.

    Returns
    -------
    tuple
        (returns_full, returns_train, returns_test,
         date_all,       date_train,     date_test)
        Each returns_* is a pandas DataFrame (n x 5).  Each date_* is a
        pandas DatetimeIndex.
    """
    # ── Read data ──────────────────────────────────────────────────────
    raw = pd.read_csv(DATA_CSV)

    # Drop rows where ANY of the five asset columns are missing
    returns_full = raw[ASSETS].dropna()

    # Extract the date column, keeping only rows that survived dropna
    # (same row alignment as R)
    date_col = raw.loc[returns_full.index, "date"]
    date_all = pd.to_datetime(date_col, format="%Y-%m-%d")

    # ── Chronological 80/20 split ──────────────────────────────────────
    np.random.seed(SEED)
    n = len(returns_full)
    train_size = int(round(n * TRAIN_RATIO))

    returns_train = returns_full.iloc[:train_size]
    returns_test  = returns_full.iloc[train_size:]

    date_train = date_all.iloc[:train_size]
    date_test  = date_all.iloc[train_size:]

    print(f"Full sample: {n} rows  |  Train: {train_size}  |  Test: {n - train_size}")

    return returns_full, returns_train, returns_test, date_all, date_train, date_test


# ===========================================================================
# 2. Descriptive statistics
# ===========================================================================

def descriptive_stats(returns_full):
    """
    Compute mean, standard deviation, skewness, and kurtosis for each asset.

    scipy.stats.kurtosis returns *excess* kurtosis; we add 3 to match R's
    kurtosis() which returns the raw (non-excess) kurtosis.

    Parameters
    ----------
    returns_full : pd.DataFrame
        Full-sample return series (n x 5).

    Returns
    -------
    pd.DataFrame
        Columns: [Variable, Mean, Std, Skew, Kurt].
    """
    rows = []
    for var in ASSETS:
        x = returns_full[var].values
        mean_v = np.mean(x)
        std_v  = np.std(x, ddof=1)
        skew_v = stats.skew(x)
        # scipy returns excess kurtosis; add 3 for raw kurtosis (like R)
        kurt_v = stats.kurtosis(x) + 3
        rows.append((var, mean_v, std_v, skew_v, kurt_v))

    df = pd.DataFrame(rows, columns=["Variable", "Mean", "Std", "Skew", "Kurt"])
    print("\n========== Descriptive Statistics ==========")
    print(df.to_string(index=False))
    return df


# ===========================================================================
# 3. Return series line plots
# ===========================================================================

def plot_return_series(returns_full, date_all, assets):
    """
    For each asset, draw a line chart of daily returns over time.

    X-axis uses yearly ticks with "%Y" format.  Styling approximates
    ggplot2's theme_minimal.  Plots are saved to DATA_DIR.

    Parameters
    ----------
    returns_full : pd.DataFrame
        Full-sample return series (n x 5).
    date_all : pd.DatetimeIndex
        Date index matching returns_full.
    assets : list of str
        Asset column names.

    Returns
    -------
    None
    """
    for var in assets:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(date_all, returns_full[var].values,
                color="steelblue", linewidth=0.6)

        ax.set_title(f"{var} 日收益率序列图", fontsize=14)
        ax.set_xlabel("交易日期")
        ax.set_ylabel("收益率 (%)")

        # Yearly ticks
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

        ax.margins(x=0.02)
        fig.tight_layout()

        out_path = DATA_DIR / f"{var}_returns.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved: {out_path}")


# ===========================================================================
# 4. Diagnostic tests (JB + ADF on training data)
# ===========================================================================

def run_diagnostic_tests(returns_train, assets):
    """
    Jarque-Bera normality test and Augmented Dickey-Fuller stationarity test
    for each asset on TRAINING data.  Significance stars are appended.

    Parameters
    ----------
    returns_train : pd.DataFrame
        Training-set returns (n_train x 5).
    assets : list of str
        Asset column names.

    Returns
    -------
    pd.DataFrame
        Columns: [Variable, JB_stat(p), ADF_stat(p)].
    """
    print("\n========== Normality and Stationarity Tests ==========")
    print(f"{'Variable':<12s} {'JB stat(p)':<22s} {'ADF stat(p)':<22s}")

    rows = []
    for var in assets:
        x = returns_train[var].values

        # Jarque-Bera
        jb_stat, jb_p = stats.jarque_bera(x)
        jb_str = f"{jb_stat:.4f}({jb_p:.4f})"
        jb_str += _signif_stars(jb_p)

        # ADF
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adf_res = adfuller(x, autolag="AIC")
        adf_stat, adf_p = adf_res[0], adf_res[1]
        adf_str = f"{adf_stat:.4f}({adf_p:.4f})"
        adf_str += _signif_stars(adf_p)

        print(f"{var:<12s} {jb_str:<22s} {adf_str:<22s}")
        rows.append((var, jb_str, adf_str))

    df = pd.DataFrame(rows, columns=["Variable", "JB_stat(p)", "ADF_stat(p)"])
    return df


# ===========================================================================
# 5. Custom ARCH-LM test (exactly mirroring R lines 8-15)
# ===========================================================================

def arch_lm_test_custom(residuals, lags=5):
    """
    Custom ARCH-LM test replicating the R function at lines 8-15.

    1. Regress squared residuals on lagged squared residuals.
    2. LM = (n - lags) * R^2
    3. p-value = 1 - chisq.cdf(LM, lags)

    Parameters
    ----------
    residuals : array-like
        Residual series (from ARMA fit).
    lags : int
        Number of lags (default 5).

    Returns
    -------
    dict
        {"statistic": LM, "p.value": p_value}.
    """
    x = np.asarray(residuals, dtype=float)
    x2 = x ** 2
    n = len(x2)

    # Build lagged squared residual matrix (same as R's sapply)
    X_lagged = np.column_stack(
        [np.roll(x2, i) for i in range(1, lags + 1)]
    )
    # First `lags` rows have NaN due to roll; drop them
    y = x2[lags:]
    X = X_lagged[lags:, :]

    # OLS: y = X @ beta + eps
    beta, ssr, rank, sv = np.linalg.lstsq(
        np.column_stack([np.ones(len(X)), X]), y, rcond=None
    )
    y_hat = np.column_stack([np.ones(len(X)), X]) @ beta
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - np.sum((y - y_hat) ** 2) / ss_tot

    lm_stat = (n - lags) * r_squared
    p_value = 1 - chi2.cdf(lm_stat, lags)

    return {"statistic": lm_stat, "p.value": p_value}


# ===========================================================================
# 6. ARCH-LM test on ARMA residuals (auto_arima + ARCH-LM)
# ===========================================================================

def run_arch_lm_on_arma_residuals(returns_train, assets):
    """
    For each asset:
      1. Fit ARMA via pmdarima.auto_arima (max_p=5, max_q=5, ic='aicc').
      2. Extract residuals.
      3. Run `arch_lm_test_custom` with lags=5.

    Prints a formatted table with ARCH-LM(5) p-values and significance stars.

    Parameters
    ----------
    returns_train : pd.DataFrame
        Training-set returns.
    assets : list of str
        Asset column names.

    Returns
    -------
    pd.DataFrame
        Columns: [Variable, ARCH_LM5_p].
    """
    try:
        import pmdarima as pm
    except ImportError:
        raise ImportError(
            "pmdarima is required. Install with: pip install pmdarima"
        )

    print("\n========== ARCH-LM(5) on ARMA Residuals ==========")

    rows = []
    for var in assets:
        x = returns_train[var].values

        # Auto ARIMA (non-seasonal, max_p=5, max_q=5, AICc)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = pm.auto_arima(
                x, max_p=5, max_q=5, seasonal=False,
                information_criterion="aicc", trace=False,
                suppress_warnings=True, error_action="ignore",
            )

        residuals = x - fit.predict_in_sample()
        arch_res = arch_lm_test_custom(residuals, lags=5)
        p_val = arch_res["p.value"]

        p_str = f"{p_val:.4f}" + _signif_stars(p_val)
        print(f"  {var:<12s} ARCH-LM(5) p={p_str}")
        rows.append((var, p_val))

    df = pd.DataFrame(rows, columns=["Variable", "ARCH_LM5_p"])
    return df


# ===========================================================================
# 7. Ljung-Box p-value matrix
# ===========================================================================

def ljung_box_matrix(returns_full, assets, lags=[1, 2, 3, 4]):
    """
    Compute Ljung-Box p-values for each asset at each lag.

    Uses statsmodels.stats.diagnostic.acorr_ljungbox.

    Parameters
    ----------
    returns_full : pd.DataFrame
        Full-sample returns.
    assets : list of str
        Asset column names.
    lags : list of int, optional
        Lags to test (default [1, 2, 3, 4]).

    Returns
    -------
    pd.DataFrame
        shape (len(lags), len(assets)).  Rows are "lag_1" ... "lag_k".
    """
    print("\n========== Ljung-Box Test p-values ==========")

    p_matrix = np.zeros((len(lags), len(assets)))

    for j, var in enumerate(assets):
        x = returns_full[var].values
        lb_result = acorr_ljungbox(x, lags=lags, return_df=True)
        p_matrix[:, j] = lb_result["lb_pvalue"].values

    df = pd.DataFrame(
        p_matrix,
        index=[f"lag_{k}" for k in lags],
        columns=assets,
    )
    print(df.to_string())
    return df


# ===========================================================================
# 8. ARMA-GARCH order selection
# ===========================================================================

def select_arma_garch_orders(returns_train, assets):
    """
    For each asset:
      - Use pmdarima.auto_arima (max_p=3, max_q=5, ic='aicc') to select
        ARMA (p, q).
      - For GARCH: loop over [(1,1), (1,2), (2,1)], fit GARCH(p,q)+normal
        via the arch library, and select by AIC.

    Prints selected ARMA and GARCH orders.

    Parameters
    ----------
    returns_train : pd.DataFrame
        Training-set returns.
    assets : list of str
        Asset column names.

    Returns
    -------
    dict
        {asset: {"arma": (p, q), "garch": (p, q), "aic": float}}
    """
    try:
        import pmdarima as pm
    except ImportError:
        raise ImportError(
            "pmdarima is required. Install with: pip install pmdarima"
        )
    try:
        from arch import arch_model
    except ImportError:
        raise ImportError(
            "arch is required. Install with: pip install arch"
        )

    print("\n========== ARMA-GARCH Order Selection ==========")

    result = {}
    garch_candidates = [(1, 1), (1, 2), (2, 1)]

    for var in assets:
        print(f"\nProcessing: {var}")
        x = returns_train[var].values

        # ── Step 1: ARMA order via auto_arima ─────────────────────────
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            arma_fit = pm.auto_arima(
                x, max_p=3, max_q=5, seasonal=False,
                information_criterion="aicc", trace=False,
                suppress_warnings=True, error_action="ignore",
            )
        ar_order = arma_fit.order[0]
        ma_order = arma_fit.order[2]
        print(f"  Selected ARMA order: ({ar_order}, {ma_order})")

        # ── Step 2: GARCH order via AIC grid search ───────────────────
        best_aic = np.inf
        best_garch = None

        for g_order in garch_candidates:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    am = arch_model(
                        x * 100,  # scale for stability
                        mean="ARX", lags=ar_order,
                        vol="GARCH", p=g_order[0], q=g_order[1],
                        dist="normal",
                    )
                    gfit = am.fit(disp="off", show_warning=False)
                    aic_val = gfit.aic
                    if aic_val < best_aic:
                        best_aic = aic_val
                        best_garch = g_order
            except Exception:
                continue

        print(f"  Selected GARCH order: {best_garch}")
        result[var] = {
            "arma": (ar_order, ma_order),
            "garch": best_garch,
            "aic": best_aic,
        }

    return result


# ===========================================================================
# Helpers
# ===========================================================================

def _signif_stars(p_value):
    """Return significance stars for a p-value: *** p<0.01, ** p<0.05, * p<0.1."""
    if p_value < 0.01:
        return "***"
    elif p_value < 0.05:
        return "**"
    elif p_value < 0.1:
        return "*"
    return ""


# ===========================================================================
# Main runner (for quick smoke-test)
# ===========================================================================

if __name__ == "__main__":
    (
        returns_full, returns_train, returns_test,
        date_all, date_train, date_test,
    ) = load_and_split_data()

    descriptive_stats(returns_full)

    plot_return_series(returns_full, date_all, ASSETS)

    run_diagnostic_tests(returns_train, ASSETS)

    run_arch_lm_on_arma_residuals(returns_train, ASSETS)

    ljung_box_matrix(returns_full, ASSETS, lags=[1, 2, 3, 4])

    select_arma_garch_orders(returns_train, ASSETS)
