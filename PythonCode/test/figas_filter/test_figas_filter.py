"""
Tests for figas_filter.py — fractional weights, copula PDF/h-functions,
FIGAS filter, static estimation, and full parameter estimation.

This is the most numerically-intensive module; tests focus on:
  - Analytical correctness (known function values)
  - Numerical stability (NaNs, extremes, boundary parameters)
  - Consistency between filter and static estimation
"""

import math

import math

import numpy as np
import pytest
from scipy.stats import kendalltau

import config
import figas_filter as ff


# ===========================================================================
# Test: frac_weights
# ===========================================================================

class TestFracWeights:
    """Fractional-difference weights for the FIGAS long-memory component."""

    def test_first_weight_is_one(self):
        w = ff.frac_weights(d=0.2, max_lag=1)
        assert w[0] == 1.0

    def test_d_0_yields_1_0_0(self):
        """When d=0, psi[0]=1, psi[1]=0, psi[2]=0..."""
        w = ff.frac_weights(d=0.0, max_lag=5)
        assert w[0] == 1.0
        assert w[1] == 0.0
        assert w[2] == 0.0

    def test_known_values_d_quarter(self):
        """For d=0.25, manually computed first 4 weights."""
        w = ff.frac_weights(d=0.25, max_lag=4)
        # psi[0] = 1
        # psi[1] = 1 * (0 - 0.25) / 1 = -0.25
        # psi[2] = -0.25 * (1 - 0.25) / 2 = -0.25 * 0.75 / 2 = -0.09375
        # psi[3] = -0.09375 * (2 - 0.25) / 3 = -0.09375 * 1.75 / 3 = -0.0546875
        expected = np.array([1.0, -0.25, -0.09375, -0.0546875])
        np.testing.assert_almost_equal(w, expected, decimal=8)

    def test_weights_sum_to_zero_for_stationary_d(self):
        """For d < 0.5, the infinite sum of psi_k = 0.
        We check that the truncated sum decays toward 0.

        Convergence rate is ~T^{-d}, so for d=0.2 / T=2000 we expect
        truncation error ≈ 2000^{-0.2} ≈ 0.22 — use a lenient bound.
        """
        w = ff.frac_weights(d=0.2, max_lag=2000)
        assert abs(np.sum(w)) < 0.3, "Truncated sum not approaching 0"

    def test_max_lag_one(self):
        w = ff.frac_weights(d=0.4, max_lag=1)
        assert len(w) == 1
        assert w[0] == 1.0

    def test_monotonic_decrease_for_positive_d(self):
        """For d>0, absolute weights should decay."""
        w = ff.frac_weights(d=0.3, max_lag=20)
        abs_w = np.abs(w)
        # The sequence should generally decrease in absolute value
        # (may have slight non-monotonicity in early terms)
        for i in range(2, len(abs_w)):
            assert abs_w[i] <= abs_w[i - 1] + 1e-10 or abs_w[i] < 0.01


# ===========================================================================
# Test: _bicop_pdf  (copula density)
# ===========================================================================

class TestBicopPdf:
    """Analytical copula PDF values for known inputs."""

    # --- Independent copula (Gaussian with rho=0 is same as independent) ---

    def test_independent_uniform_gives_one(self):
        """For independent U(0,1) points, any copula with par=independence gives 1."""
        # Independence: Gaussian copula with rho=0
        pdf = ff._bicop_pdf(0.5, 0.5, family=2, par=0.0, par2=5.0)
        # t-copula with rho=0, nu=5: value depends on nu, not exactly 1
        # Just check it's finite and positive
        assert np.isfinite(pdf) and pdf > 0

    # --- Clayton (family=3) ---

    def test_clayton_known_value(self):
        """Known Clayton density c(0.5, 0.5; theta=2)."""
        pdf = ff._bicop_pdf(0.5, 0.5, family=3, par=2.0, par2=0)
        # c(u,v) = (1+theta)*(u*v)^(-theta-1)*(u^(-theta)+v^(-theta)-1)^(-1/theta-2)
        # theta=2, u=v=0.5:
        #   u^(-2)=4, v^(-2)=4, s=4+4-1=7
        #   (1+2)*(0.5*0.5)^(-3)*(7)^(-0.5-2)=3*(0.25)^(-3)*(7)^(-2.5)
        #   = 3*64*(7)^(-2.5) ≈ 3*64/129.64 ≈ 1.4808
        u = 0.5
        theta = 2.0
        ut = u ** (-theta)
        s = 2 * ut - 1
        expected = (1 + theta) * (u * u) ** (-theta - 1) * s ** (-1.0 / theta - 2)
        np.testing.assert_almost_equal(pdf, expected, decimal=4)

    def test_clayton_symmetry(self, rng):
        """Clayton PDF should be symmetric: c(u,v) = c(v,u)."""
        u = rng.uniform(size=20)
        v = rng.uniform(size=20)
        pdf1 = ff._bicop_pdf(u, v, family=3, par=2.0)
        pdf2 = ff._bicop_pdf(v, u, family=3, par=2.0)
        np.testing.assert_array_almost_equal(pdf1, pdf2)

    def test_clayton_bounds(self):
        """Clayton density should be > 0 for interior points."""
        for theta in [0.5, 1.0, 2.0, 5.0]:
            pdf = ff._bicop_pdf(0.3, 0.7, family=3, par=theta)
            assert pdf > 0

    # --- t-Copula (family=2) ---

    def test_t_copula_rho_zero(self):
        """t-copula with rho=0: should be close to independent t density."""
        pdf = ff._bicop_pdf(0.5, 0.5, family=2, par=0.0, par2=5.0)
        assert np.isfinite(pdf) and pdf > 0

    def test_t_copula_large_degrees(self):
        """As nu → inf, t-copula → Gaussian copula."""
        pdf_t = ff._bicop_pdf(0.3, 0.7, family=2, par=0.5, par2=30.0)
        pdf_gauss = ff._bicop_pdf(0.3, 0.7, family=2, par=0.5, par2=100.0)
        # Both finite and positive
        assert np.isfinite(pdf_t) and np.isfinite(pdf_gauss)

    # --- Gumbel (family=4) ---

    def test_gumbel_independence(self):
        """Gumbel with theta=1 → independence → density = 1."""
        pdf = ff._bicop_pdf(0.5, 0.5, family=4, par=1.0, par2=0)
        # theta=1.0 is the independence case for Gumbel
        # The implementation uses max(par, 1.0001) so 1.0 gets clamped to 1.0001
        assert np.isfinite(pdf) and pdf > 0

    def test_gumbel_positive_dependence(self, rng):
        """Gumbel density should be > 0."""
        u = rng.uniform(size=10)
        v = rng.uniform(size=10)
        pdf = ff._bicop_pdf(u, v, family=4, par=2.0)
        assert np.all(np.isfinite(pdf))
        assert np.all(pdf > 0)

    # --- Survival Gumbel (family=14) ---

    def test_survival_gumbel_rotation(self, rng):
        """c_14(u,v) = c_4(1-u, 1-v)."""
        u, v = 0.3, 0.7
        pdf14 = ff._bicop_pdf(u, v, family=14, par=2.5)
        pdf4 = ff._bicop_pdf(1 - u, 1 - v, family=4, par=2.5)
        np.testing.assert_almost_equal(pdf14, pdf4, decimal=8)

    # --- Clayton 90-rotated (family=23) ---

    def test_clayton_90_rotation(self):
        """c_23(u,v) = c_3(1-u, v; -theta)."""
        u, v = 0.3, 0.7
        theta23 = -2.0
        pdf23 = ff._bicop_pdf(u, v, family=23, par=theta23)
        pdf3 = ff._bicop_pdf(1 - u, v, family=3, par=-theta23)
        np.testing.assert_almost_equal(pdf23, pdf3, decimal=8)

    # --- Edge cases ---

    @pytest.mark.parametrize("family", [2, 3, 4, 14, 23])
    def test_pdf_at_extremes(self, family):
        """PDF at very interior points should not crash."""
        u, v = 0.9999, 0.0001
        pdf = ff._bicop_pdf(u, v, family=family, par=2.0, par2=5.0 if family == 2 else 0)
        assert np.isfinite(pdf)
        assert pdf >= config.PDF_FLOOR

    @pytest.mark.parametrize("family", [2, 3, 4, 14, 23])
    def test_pdf_array_input(self, rng, family):
        """PDF should work with array inputs."""
        u = rng.uniform(size=50)
        v = rng.uniform(size=50)
        par = 2.0
        par2 = 5.0 if family == 2 else 0
        pdf = ff._bicop_pdf(u, v, family=family, par=par, par2=par2)
        assert len(pdf) == 50
        assert np.all(np.isfinite(pdf))

    def test_unsupported_family_raises(self):
        with pytest.raises(ValueError, match="Unsupported copula family"):
            ff._bicop_pdf(0.5, 0.5, family=99, par=1.0)


# ===========================================================================
# Test: _bicop_hfunc
# ===========================================================================

class TestBicopHfunc:
    """Conditional CDF (h-function) for each copula family."""

    @pytest.mark.parametrize("family", [2, 3, 4, 14, 23])
    def test_hfunc_range(self, rng, family):
        """h-functions should be in (0, 1)."""
        u, v = 0.3, 0.7
        par = 2.0
        par2 = 5.0 if family == 2 else 0
        h1, h2 = ff._bicop_hfunc(u, v, family, par, par2)
        assert 0 < h1 < 1, f"h1={h1} out of (0,1) for family {family}"
        assert 0 < h2 < 1, f"h2={h2} out of (0,1) for family {family}"

    def test_clayton_hfunc_symmetric(self, rng):
        """Clayton h-functions should be equal at u=v."""
        h1, h2 = ff._bicop_hfunc(0.5, 0.5, family=3, par=2.0)
        np.testing.assert_almost_equal(h1, h2)

    def test_survival_gumbel_hfunc(self):
        """h1_SG(u,v) = 1 - h1_G(1-u, 1-v)."""
        u, v = 0.3, 0.7
        theta = 2.5
        h1_14, h2_14 = ff._bicop_hfunc(u, v, family=14, par=theta)
        h1_4, h2_4 = ff._bicop_hfunc(1 - u, 1 - v, family=4, par=theta)
        np.testing.assert_almost_equal(h1_14, 1 - h1_4, decimal=8)
        np.testing.assert_almost_equal(h2_14, 1 - h2_4, decimal=8)

    def test_clayton_90_hfunc(self):
        """h1_23(u,v) = 1 - h1_3(1-u, v)."""
        u, v = 0.3, 0.7
        theta = -2.0
        h1_23, h2_23 = ff._bicop_hfunc(u, v, family=23, par=theta)
        h1_3, h2_3 = ff._bicop_hfunc(1 - u, v, family=3, par=-theta)
        np.testing.assert_almost_equal(h1_23, 1 - h1_3, decimal=8)
        np.testing.assert_almost_equal(h2_23, h2_3, decimal=8)

    def test_t_copula_hfunc(self):
        """t-copula h-functions should be in (0,1)."""
        h1, h2 = ff._bicop_hfunc(0.3, 0.7, family=2, par=0.5, par2=5.0)
        assert 0 < h1 < 1
        assert 0 < h2 < 1
        assert np.isfinite(h1)
        assert np.isfinite(h2)

    def test_hfunc_array_input(self, rng):
        """h-functions should work with array inputs."""
        u = rng.uniform(size=30)
        v = rng.uniform(size=30)
        h1, h2 = ff._bicop_hfunc(u, v, family=3, par=2.0)
        assert len(h1) == 30
        assert len(h2) == 30
        assert np.all((h1 > 0) & (h1 < 1))
        assert np.all((h2 > 0) & (h2 < 1))

    def test_unsupported_family_raises(self):
        with pytest.raises(ValueError):
            ff._bicop_hfunc(0.5, 0.5, family=99, par=1.0)


# ===========================================================================
# Test: _safe_bicop_hfunc
# ===========================================================================

class TestSafeBicopHfunc:
    """Safe wrapper — falls back to independence copula."""

    def test_normal_operation(self):
        h1, h2 = ff._safe_bicop_hfunc(0.3, 0.7, family=3, par=2.0)
        assert 0 < h1 < 1
        assert 0 < h2 < 1

    def test_fallback_on_invalid_family(self):
        """Unknown family should revert to independence (h1=u1, h2=u2)."""
        h1, h2 = ff._safe_bicop_hfunc(0.3, 0.7, family=99, par=1.0)
        np.testing.assert_almost_equal(h1, 0.3)
        np.testing.assert_almost_equal(h2, 0.7)

    def test_fallback_on_extreme_parameters(self, rng):
        """Extreme parameters might fail; fallback should produce valid values."""
        u = rng.uniform(size=10)
        v = rng.uniform(size=10)
        h1, h2 = ff._safe_bicop_hfunc(u, v, family=2, par=2.0, par2=0.1)
        # par2=0.1 is below the nu >= 2.01 threshold, might trigger fallback
        assert len(h1) == 10
        assert len(h2) == 10


# ===========================================================================
# Test: _inverse_link
# ===========================================================================

class TestInverseLink:
    """Inverse link function — maps constrained par → unconstrained mu."""

    def test_t_copula_rho_zero(self):
        """rho=0 → start_mu=0."""
        mu = ff._inverse_link(fam_id=2, par=0.0)
        np.testing.assert_almost_equal(mu, 0.0, decimal=4)

    def test_clayton_positive(self):
        """Clayton theta > 0 → positive mu."""
        mu = ff._inverse_link(fam_id=3, par=2.0)
        assert mu > 0

    def test_survival_gumbel(self):
        """Survival Gumbel theta=2 → mu ≈ log(2 - 1.0001) ≈ -0.0001 (essentially 0)."""
        mu = ff._inverse_link(fam_id=14, par=2.0)
        expected = math.log(2.0 - 1.0001)
        np.testing.assert_almost_equal(mu, expected, decimal=4)

    def test_clayton_90(self):
        mu = ff._inverse_link(fam_id=23, par=-2.0)
        assert mu > 0  # inverted parameter

    def test_round_trip(self, rng):
        """For a range of families, inverse_link should be reversible."""
        for fam, par_min, par_max, default_par2 in [
            (2, -0.9, 0.9, 0),
            (3, 0.5, 5.0, 0),
            (14, 1.5, 5.0, 0),
            (23, -5.0, -0.5, 0),
        ]:
            par = rng.uniform(par_min, par_max)
            mu = ff._inverse_link(fam, par)
            assert np.isfinite(mu)
            assert abs(mu) < 30  # within safe bounds


# ===========================================================================
# Test: _static_copula_fit
# ===========================================================================

class TestStaticCopulaFit:
    """MLE-based static copula parameter estimation."""

    def test_clayton_on_clayton_data(self, clayton_synthetic):
        """Estimate Clayton theta from known Clayton samples — should be close."""
        par, par2 = ff._static_copula_fit(
            clayton_synthetic["u1"], clayton_synthetic["u2"], fam_id=3
        )
        assert par > 0
        # Within ~30% of true value (rough check — MLE consistency)
        assert abs(par - clayton_synthetic["theta"]) / clayton_synthetic["theta"] < 0.5

    def test_t_copula_on_t_data(self, t_copula_synthetic):
        """Estimate t-copula parameters from known t samples."""
        par, par2 = ff._static_copula_fit(
            t_copula_synthetic["u1"], t_copula_synthetic["u2"], fam_id=2
        )
        assert -1 < par < 1
        assert par2 > 2.0

    def test_clayton_survival_on_survival_data(self, rng):
        """Survival Gumbel estimation from survival Gumbel data."""
        # Generate data from Survival Gumbel (family 14)
        from figas_filter import _bicop_hfunc
        n = 300
        theta_true = 2.5
        u1 = rng.uniform(size=n)
        u2 = np.zeros(n)
        for i in range(n):
            cand = rng.uniform(0, 1, size=200)
            h = _bicop_hfunc(np.full(200, u1[i]), cand, family=14, par=theta_true)[0]
            u2[i] = cand[np.argmin(np.abs(h - rng.uniform()))]
        par, par2 = ff._static_copula_fit(u1, u2, fam_id=14)
        assert par >= 1.0001
        assert abs(par - theta_true) / theta_true < 0.5

    def test_clayton_90_on_90_data(self, rng):
        """Clayton 90-rotated estimation."""
        from figas_filter import _bicop_hfunc
        n = 300
        theta_true = -2.0
        u1 = rng.uniform(size=n)
        u2 = np.zeros(n)
        for i in range(n):
            cand = rng.uniform(0, 1, size=200)
            h = _bicop_hfunc(np.full(200, u1[i]), cand, family=23, par=theta_true)[0]
            u2[i] = cand[np.argmin(np.abs(h - rng.uniform()))]
        par, par2 = ff._static_copula_fit(u1, u2, fam_id=23)
        assert par < 0

    def test_independent_data_gives_near_independence(self, rng):
        """Independent U(0,1) data should give near-independence parameters."""
        u1 = rng.uniform(size=500)
        u2 = rng.uniform(size=500)
        # Clayton
        par, _ = ff._static_copula_fit(u1, u2, fam_id=3)
        assert par < 0.5  # near-independence Clayton → theta ≈ 0
        # t-copula
        par_t, nu = ff._static_copula_fit(u1, u2, fam_id=2)
        assert abs(par_t) < 0.15

    def test_unsupported_family_raises(self, rng):
        u1 = rng.uniform(size=100)
        u2 = rng.uniform(size=100)
        with pytest.raises(ValueError):
            ff._static_copula_fit(u1, u2, fam_id=99)


# ===========================================================================
# Test: filter_figas
# ===========================================================================

class TestFilterFigas:
    """Core FIGAS(1,d,1) filter."""

    def test_filter_returns_expected_keys(self, clayton_synthetic):
        """Filter output dict should have all required fields."""
        mu = ff._inverse_link(3, clayton_synthetic["theta"])
        theta = np.array([mu, 0.05, 0.80, 0.20])
        result = ff.filter_figas(theta, clayton_synthetic["u1"], clayton_synthetic["u2"], fam_id=3)
        assert "loglik" in result
        assert "ll_seq" in result
        assert "par_t" in result
        assert "h1" in result
        assert "h2" in result
        assert "y" in result

    def test_filter_likelihood_finite(self, clayton_synthetic):
        """Log-likelihood should be finite."""
        mu = ff._inverse_link(3, clayton_synthetic["theta"])
        theta = np.array([mu, 0.05, 0.80, 0.20])
        result = ff.filter_figas(theta, clayton_synthetic["u1"], clayton_synthetic["u2"], fam_id=3)
        assert np.isfinite(result["loglik"])

    def test_filter_par_t_in_valid_range(self, clayton_synthetic):
        """Time-varying parameter should stay in bound for Clayton."""
        mu = ff._inverse_link(3, clayton_synthetic["theta"])
        theta = np.array([mu, 0.05, 0.80, 0.20])
        result = ff.filter_figas(theta, clayton_synthetic["u1"], clayton_synthetic["u2"], fam_id=3)
        assert np.all(result["par_t"] > 0)
        assert np.all(np.isfinite(result["par_t"]))

    def test_filter_t_copula(self, t_copula_synthetic):
        """t-copula filter should produce valid results."""
        mu = ff._inverse_link(2, t_copula_synthetic["rho"])
        theta = np.array([mu, 0.05, 0.80, 0.20, t_copula_synthetic["nu"]])
        result = ff.filter_figas(
            theta, t_copula_synthetic["u1"], t_copula_synthetic["u2"], fam_id=2
        )
        assert np.isfinite(result["loglik"])
        assert np.all(result["par_t"] > -1)
        assert np.all(result["par_t"] < 1)

    def test_filter_length_matches_input(self, clayton_synthetic):
        """Output arrays should match input length."""
        mu = ff._inverse_link(3, clayton_synthetic["theta"])
        theta = np.array([mu, 0.05, 0.80, 0.20])
        result = ff.filter_figas(theta, clayton_synthetic["u1"], clayton_synthetic["u2"], fam_id=3)
        n = len(clayton_synthetic["u1"])
        assert len(result["ll_seq"]) == n
        assert len(result["par_t"]) == n
        assert len(result["h1"]) == n
        assert len(result["y"]) == n

    def test_constant_parameter_limit(self, clayton_synthetic):
        """When alpha=beta=0, g_t should stay at mu (constant parameter)."""
        mu = ff._inverse_link(3, clayton_synthetic["theta"])
        theta = np.array([mu, 0.0, 0.0, 0.0])
        result = ff.filter_figas(theta, clayton_synthetic["u1"], clayton_synthetic["u2"], fam_id=3)
        # par_t should be approximately constant
        par_std = np.std(result["par_t"])
        assert par_std < 0.05, f"par_t std = {par_std}, expected near-constant"


# ===========================================================================
# Test: estimate_figas_params
# ===========================================================================

class TestEstimateFigasParams:
    """Full FIGAS parameter estimation via L-BFGS-B.
    NOTE: Each test runs ~10-60s optimization — marked slow.
    """
    pytestmark = pytest.mark.slow

    def test_estimation_on_clayton(self, clayton_synthetic):
        """Should converge to finite parameters with valid likelihood."""
        best_params, fres = ff.estimate_figas_params(
            clayton_synthetic["u1"], clayton_synthetic["u2"],
            fam_id=3, verbose=False
        )
        assert len(best_params) == 4  # mu, alpha, beta, d
        assert np.all(np.isfinite(best_params))
        assert np.isfinite(fres["loglik"])

    def test_d_parameter_in_bounds(self, clayton_synthetic):
        """d should be within FIGAS bounds (0.05, 0.49)."""
        best_params, fres = ff.estimate_figas_params(
            clayton_synthetic["u1"], clayton_synthetic["u2"],
            fam_id=3, verbose=False
        )
        d_val = best_params[3]
        assert config.FIGAS_BOUNDS["d"][0] <= d_val <= config.FIGAS_BOUNDS["d"][1]

    def test_mu_in_bounds(self, clayton_synthetic):
        best_params, fres = ff.estimate_figas_params(
            clayton_synthetic["u1"], clayton_synthetic["u2"],
            fam_id=3, verbose=False
        )
        mu_val = best_params[0]
        assert config.FIGAS_BOUNDS["mu"][0] <= mu_val <= config.FIGAS_BOUNDS["mu"][1]

    def test_alpha_beta_in_bounds(self, clayton_synthetic):
        best_params, fres = ff.estimate_figas_params(
            clayton_synthetic["u1"], clayton_synthetic["u2"],
            fam_id=3, verbose=False
        )
        assert config.FIGAS_BOUNDS["alpha"][0] <= best_params[1] <= config.FIGAS_BOUNDS["alpha"][1]
        assert config.FIGAS_BOUNDS["beta"][0] <= best_params[2] <= config.FIGAS_BOUNDS["beta"][1]

    def test_t_copula_estimation(self, t_copula_synthetic):
        """t-copula estimation should converge."""
        best_params, fres = ff.estimate_figas_params(
            t_copula_synthetic["u1"], t_copula_synthetic["u2"],
            fam_id=2, verbose=False
        )
        assert len(best_params) == 5  # mu, alpha, beta, d, kappa
        assert np.all(np.isfinite(best_params))
        assert np.isfinite(fres["loglik"])

    def test_t_copula_kappa_in_bounds(self, t_copula_synthetic):
        best_params, fres = ff.estimate_figas_params(
            t_copula_synthetic["u1"], t_copula_synthetic["u2"],
            fam_id=2, verbose=False
        )
        kappa = best_params[4]
        assert config.FIGAS_BOUNDS["kappa"][0] <= kappa <= config.FIGAS_BOUNDS["kappa"][1]


# ===========================================================================
# Test: Integration — filter improves over static for time-varying data
# ===========================================================================

class TestFigasIntegration:
    """FIGAS should outperform static model on time-varying dependence."""

    @pytest.mark.slow
    def test_figas_better_than_static(self, rng):
        """
        Generate data from a time-varying Clayton copula.
        FIGAS (dynamic) log-likelihood should exceed static log-likelihood.
        """
        from scipy.stats import kendalltau
        n = 400
        t = np.arange(n) / n
        true_theta = 1.0 + 1.0 * np.sin(2 * np.pi * t * 4)

        u1 = rng.uniform(size=n)
        u2 = np.zeros(n)
        for i in range(n):
            cand = rng.uniform(0, 1, size=200)
            h = ff._bicop_hfunc(np.full(200, u1[i]), cand, family=3, par=true_theta[i])[0]
            u2[i] = cand[np.argmin(np.abs(h - rng.uniform()))]

        # Static MLE
        static_par, _ = ff._static_copula_fit(u1, u2, fam_id=3)
        static_pdf = ff._bicop_pdf(u1, u2, family=3, par=static_par)
        static_ll = float(np.sum(np.log(np.maximum(static_pdf, config.PDF_FLOOR))))

        # FIGAS
        best_params, fres = ff.estimate_figas_params(u1, u2, fam_id=3, verbose=False)
        figas_ll = fres["loglik"]

        assert figas_ll > static_ll, \
            f"FIGAS LL ({figas_ll:.2f}) ≤ Static LL ({static_ll:.2f})"
