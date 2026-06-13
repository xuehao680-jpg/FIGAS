"""
Shared pytest fixtures and configuration for the GARCH-Copula test suite.

All fixtures generate synthetic data — no external CSV dependency.
"""

import sys
import math
from pathlib import Path

import numpy as np
import pytest

# ── Ensure PythonCode/ is on sys.path so imports like `import config` work ──
_PROJECT_DIR = Path(__file__).resolve().parents[1]  # PythonCode/
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

# ── Constants used across many tests ─────────────────────────────────────────
SEED = 42
N_OBS = 500                     # default time-series length
N_ASSETS = 5                    # matches config.ASSETS
ASSET_NAMES = ["hushen300", "ydlMIB", "xxlNZ50", "nfFTSE", "bxBOVESPA"]

# ===========================================================================
#  Shared synthetic-data fixtures
# ===========================================================================

@pytest.fixture(scope="session")
def rng():
    """Deterministic NumPy random state for reproducible tests."""
    return np.random.RandomState(SEED)


@pytest.fixture(scope="session")
def uniform_data(rng):
    """
    Synthetic (N_OBS x 5) U(0,1) matrix — simulates PIT-transformed data
    ready for copula modelling.
    """
    return rng.uniform(size=(N_OBS, N_ASSETS))


@pytest.fixture(scope="session")
def weakly_correlated_uniform(rng):
    """
    Weakly correlated U(0,1) samples via a Gaussian copula with
    off-diagonal rho = 0.3, transformed to margins via normal CDF.

    Returns a (N_OBS x 5) array.
    """
    from scipy.stats import norm
    d = 5
    cov = np.full((d, d), 0.3)
    np.fill_diagonal(cov, 1.0)
    L = np.linalg.cholesky(cov)
    z = rng.randn(N_OBS, d) @ L.T
    return norm.cdf(z)


@pytest.fixture(scope="session")
def strongly_correlated_uniform(rng):
    """
    Strongly correlated U(0,1) samples (rho = 0.7).
    """
    from scipy.stats import norm
    d = 5
    cov = np.full((d, d), 0.7)
    np.fill_diagonal(cov, 1.0)
    L = np.linalg.cholesky(cov)
    z = rng.randn(N_OBS, d) @ L.T
    return norm.cdf(z)


@pytest.fixture(scope="session")
def garch_synthetic_returns(rng):
    """
    Synthetic return series from a stable GARCH(1,1) process.

    Generates proper ARMA(0,0)-GARCH(1,1) data:
      r_t = sigma_t * z_t,   z_t ~ N(0,1)
      sigma2_t = omega + alpha * r_{t-1}^2 + beta * sigma2_{t-1}

    Parameters calibrated to produce daily-level returns (~0.5-1% vol).
    Returns a (N_OBS x 5) DataFrame.
    """
    import pandas as pd
    n = N_OBS
    df_list = []
    for i in range(N_ASSETS):
        omega = 0.02
        alpha = 0.08 + 0.01 * i
        beta = 0.88 - 0.01 * i   # sum(alpha+beta) ~ 0.96–0.90 → stationary
        z = rng.randn(n)
        r = np.zeros(n)
        sigma2 = omega / (1.0 - alpha - beta)  # unconditional variance
        for t in range(n):
            sigma2 = max(omega + alpha * r[t - 1] ** 2 + beta * sigma2, 1e-8) if t > 0 else sigma2
            r[t] = np.sqrt(sigma2) * z[t]
        df_list.append(r)

    arr = np.column_stack(df_list)
    df = pd.DataFrame(arr, columns=ASSET_NAMES)
    return df


@pytest.fixture(scope="function")
def tmp_csv(rng, tmp_path):
    """
    Create a temporary CSV file mimicking the structure of yidaiyilu(1).csv.

    Columns: date + the 5 asset columns.  Returns the path.
    """
    import pandas as pd
    n = 200
    dates = pd.date_range("2010-01-01", periods=n, freq="B")
    data = {"date": dates.strftime("%Y-%m-%d")}
    for col in ASSET_NAMES:
        data[col] = rng.randn(n) * 0.02
    df = pd.DataFrame(data)
    out = tmp_path / "test_data.csv"
    df.to_csv(out, index=False)
    return out


@pytest.fixture(scope="session")
def clayton_synthetic(rng):
    """
    Bivariate Clayton-copula data with known parameters for filter tests.

    Returns a dict: u1, u2 (1D arrays), true_theta (scalar).
    """
    from figas_filter import _bicop_hfunc
    n = 400
    theta_true = 2.0
    u1 = rng.uniform(size=n)
    u2 = np.zeros(n)
    for i in range(n):
        candidates = rng.uniform(0, 1, size=200)
        h_vals = _bicop_hfunc(
            np.full(200, u1[i]), candidates,
            family=3, par=theta_true, par2=0
        )[0]
        rv = rng.uniform()
        idx = np.argmin(np.abs(h_vals - rv))
        u2[i] = candidates[idx]
    return {"u1": u1, "u2": u2, "theta": theta_true}


@pytest.fixture(scope="session")
def t_copula_synthetic(rng):
    """
    Bivariate t-copula data with known rho and nu for filter tests.
    """
    from scipy.stats import t as t_dist
    from scipy.stats import multivariate_t
    rho_true = 0.5
    nu_true = 5.0
    n = 400
    mvt = multivariate_t(
        shape=[[1, rho_true], [rho_true, 1]], df=nu_true, seed=SEED + 1
    )
    samples = mvt.rvs(n)
    u1 = t_dist.cdf(samples[:, 0], df=nu_true)
    u2 = t_dist.cdf(samples[:, 1], df=nu_true)
    return {"u1": u1, "u2": u2, "rho": rho_true, "nu": nu_true}
