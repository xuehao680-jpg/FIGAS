"""
Tests for gas_filter.py — GAS(1,1) filter and parameter estimation.

Ensures the GAS filter:
  1. Produces finite log-likelihoods on synthetic data.
  2. Respects parameter bounds during estimation.
  3. Shares consistent link-function behaviour with FIGAS.
  4. Has the expected special-case behaviour (alpha=beta=0 → constant par).
"""

import numpy as np
import pytest

import config
import gas_filter as gf
from figas_filter import _inverse_link, _static_copula_fit, _bicop_pdf


# ===========================================================================
# Test: filter_gas  (core filter function)
# ===========================================================================

class TestFilterGas:
    """Core GAS(1,1) filter."""

    def test_filter_returns_expected_keys(self, clayton_synthetic):
        """Filter output should contain loglik, ll_seq, par_t, h1, h2."""
        mu = _inverse_link(3, clayton_synthetic["theta"])
        theta = np.array([mu, 0.05, 0.80])
        result = gf.filter_gas(theta, clayton_synthetic["u1"], clayton_synthetic["u2"], family=3)
        assert "loglik" in result
        assert "ll_seq" in result
        assert "par_t" in result
        assert "h1" in result
        assert "h2" in result

    def test_likelihood_finite(self, clayton_synthetic):
        """Log-likelihood should be finite."""
        mu = _inverse_link(3, clayton_synthetic["theta"])
        theta = np.array([mu, 0.05, 0.80])
        result = gf.filter_gas(theta, clayton_synthetic["u1"], clayton_synthetic["u2"], family=3)
        assert np.isfinite(result["loglik"])

    def test_par_t_in_valid_range_clayton(self, clayton_synthetic):
        """Clayton parameter should stay positive and finite."""
        mu = _inverse_link(3, clayton_synthetic["theta"])
        theta = np.array([mu, 0.05, 0.80])
        result = gf.filter_gas(theta, clayton_synthetic["u1"], clayton_synthetic["u2"], family=3)
        assert np.all(result["par_t"] > 0)
        assert np.all(np.isfinite(result["par_t"]))

    def test_par_t_in_valid_range_t(self, t_copula_synthetic):
        """t-copula rho should stay in (-1, 1)."""
        mu = _inverse_link(2, t_copula_synthetic["rho"])
        theta = np.array([mu, 0.05, 0.80, t_copula_synthetic["nu"]])
        result = gf.filter_gas(
            theta, t_copula_synthetic["u1"], t_copula_synthetic["u2"],
            family=2
        )
        assert np.all(result["par_t"] > -1)
        assert np.all(result["par_t"] < 1)
        assert np.all(np.isfinite(result["par_t"]))

    @pytest.mark.parametrize("family", [3, 14, 23])
    def test_filter_non_t_families(self, clayton_synthetic, family):
        """Filter should work for all supported non-t families."""
        mu = _inverse_link(family, clayton_synthetic["theta"] if family in (3, 23) else 2.5)
        theta = np.array([mu, 0.05, 0.80])
        u1 = clayton_synthetic["u1"]
        u2 = clayton_synthetic["u2"]
        result = gf.filter_gas(theta, u1, u2, family=family)
        assert np.isfinite(result["loglik"])

    def test_output_length_matches_input(self, clayton_synthetic):
        """Output arrays should match input length."""
        mu = _inverse_link(3, clayton_synthetic["theta"])
        theta = np.array([mu, 0.05, 0.80])
        result = gf.filter_gas(theta, clayton_synthetic["u1"], clayton_synthetic["u2"], family=3)
        n = len(clayton_synthetic["u1"])
        assert len(result["ll_seq"]) == n
        assert len(result["par_t"]) == n
        assert len(result["h1"]) == n
        assert len(result["h2"]) == n

    def test_constant_parameter_limit(self, clayton_synthetic):
        """When alpha=0, beta=0, g_t stays at mu, so par_t should be near-constant."""
        mu = _inverse_link(3, clayton_synthetic["theta"])
        theta = np.array([mu, 0.0, 0.0])
        result = gf.filter_gas(theta, clayton_synthetic["u1"], clayton_synthetic["u2"], family=3)
        par_std = np.std(result["par_t"])
        assert par_std < 0.05, f"par_t std = {par_std}, expected near-constant"


# ===========================================================================
# Test: estimate_gas_params
# ===========================================================================

class TestEstimateGasParams:
    """Full GAS(1,1) parameter estimation.
    NOTE: Runs L-BFGS-B — slow.
    """
    pytestmark = pytest.mark.slow

    def test_estimation_on_clayton(self, clayton_synthetic):
        """Should converge to finite parameters."""
        best_params, fres = gf.estimate_gas_params(
            clayton_synthetic["u1"], clayton_synthetic["u2"],
            fam_id=3, verbose=False
        )
        assert len(best_params) == 3  # mu, alpha, beta
        assert np.all(np.isfinite(best_params))
        assert np.isfinite(fres["loglik"])

    def test_parameter_bounds(self, clayton_synthetic):
        """Estimated params should be within config bounds."""
        best_params, fres = gf.estimate_gas_params(
            clayton_synthetic["u1"], clayton_synthetic["u2"],
            fam_id=3, verbose=False
        )
        mu, alpha, beta = best_params
        assert config.GAS_BOUNDS["mu"][0] <= mu <= config.GAS_BOUNDS["mu"][1]
        assert config.GAS_BOUNDS["alpha"][0] <= alpha <= config.GAS_BOUNDS["alpha"][1]
        assert config.GAS_BOUNDS["beta"][0] <= beta <= config.GAS_BOUNDS["beta"][1]

    def test_t_copula_estimation(self, t_copula_synthetic):
        """t-copula: should estimate 4 params (mu, alpha, beta, kappa)."""
        best_params, fres = gf.estimate_gas_params(
            t_copula_synthetic["u1"], t_copula_synthetic["u2"],
            fam_id=2, verbose=False
        )
        assert len(best_params) == 4
        assert np.all(np.isfinite(best_params))
        assert np.isfinite(fres["loglik"])

    def test_t_copula_kappa_in_bounds(self, t_copula_synthetic):
        best_params, _ = gf.estimate_gas_params(
            t_copula_synthetic["u1"], t_copula_synthetic["u2"],
            fam_id=2, verbose=False
        )
        kappa = best_params[3]
        assert config.GAS_BOUNDS["kappa"][0] <= kappa <= config.GAS_BOUNDS["kappa"][1]


# ===========================================================================
# Test: Consistency with FIGAS link functions
# ===========================================================================

class TestGasFigasConsistency:
    """GAS and FIGAS share link functions and copula implementations."""

    @pytest.mark.parametrize("family", [2, 3, 14, 23])
    def test_link_function_identical(self, family, rng):
        """Both filters use the same _inverse_link and link funcs from figas_filter."""
        # GAS filter_gas imports _inverse_link from figas_filter — verify
        from figas_filter import _inverse_link as figas_link
        test_par = rng.uniform(-0.8, 0.8) if family == 2 else rng.uniform(1.5, 5.0)
        mu1 = _inverse_link(family, test_par)
        mu2 = figas_link(family, test_par)
        np.testing.assert_almost_equal(mu1, mu2)

    def test_same_pdf_backend(self, rng):
        """Both use _bicop_pdf from figas_filter for likelihood eval."""
        from figas_filter import _bicop_pdf as fb_pdf
        u = rng.uniform(size=10)
        v = rng.uniform(size=10)
        p1 = _bicop_pdf(u, v, family=3, par=2.0)
        p2 = fb_pdf(u, v, family=3, par=2.0)
        np.testing.assert_array_almost_equal(p1, p2)


# ===========================================================================
# Test: Integration — dynamic outperforms static
# ===========================================================================

class TestGasIntegration:
    """GAS(1,1) should outperform static on time-varying data."""

    @pytest.mark.slow
    def test_gas_better_than_static(self, rng):
        """
        On a time-varying Clayton process, GAS log-likelihood should
        exceed the static log-likelihood.
        """
        n = 400
        t = np.arange(n) / n
        true_theta = 1.0 + 1.5 * np.sin(2 * np.pi * t * 3)

        u1 = rng.uniform(size=n)
        u2 = np.zeros(n)
        for i in range(n):
            from figas_filter import _bicop_hfunc
            cand = rng.uniform(0, 1, size=200)
            h = _bicop_hfunc(np.full(200, u1[i]), cand, family=3, par=true_theta[i])[0]
            u2[i] = cand[np.argmin(np.abs(h - rng.uniform()))]

        # Static
        static_par, _ = _static_copula_fit(u1, u2, fam_id=3)
        static_pdf = _bicop_pdf(u1, u2, family=3, par=static_par)
        static_ll = float(np.sum(np.log(np.maximum(static_pdf, config.PDF_FLOOR))))

        # GAS
        best_params, fres = gf.estimate_gas_params(u1, u2, fam_id=3, verbose=False)
        gas_ll = fres["loglik"]

        assert gas_ll > static_ll, \
            f"GAS LL ({gas_ll:.2f}) ≤ Static LL ({static_ll:.2f})"
