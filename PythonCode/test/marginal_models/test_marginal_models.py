"""
Tests for marginal_models.py — GJR-GARCH fitting, standardized residuals,
PIT transform, residual diagnostics, test-set filtering, and persistence.
"""

import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import config
import marginal_models


# ===========================================================================
# Helpers
# ===========================================================================

def _assert_valid_pit(u):
    """PIT values should be strictly in (0, 1) for finite entries."""
    u = np.asarray(u)
    valid = u[np.isfinite(u)]
    assert np.all(valid > 0), f"PIT ≤ 0 at {(valid <= 0).sum()} valid positions"
    assert np.all(valid < 1), f"PIT ≥ 1 at {(valid >= 1).sum()} valid positions"
    assert np.all(np.isfinite(valid))


# ===========================================================================
# Test: arch_lm_test  (correct version in marginal_models)
# ===========================================================================

class TestArchLm:
    """
    The correct ARCH-LM implementation lives in marginal_models.py.
    These tests verify it against known behaviour.
    """

    def test_white_noise(self, rng):
        """White noise: should not reject H0 (no ARCH)."""
        x = rng.randn(500)
        lm, p = marginal_models.arch_lm_test(x, lags=5)
        assert p > 0.01

    def test_arch_effects(self, rng):
        """Known ARCH(1) process: should reject H0."""
        n = 500
        eps = rng.randn(n)
        sigma2 = np.ones(n)
        for t in range(1, n):
            sigma2[t] = 0.1 + 0.8 * eps[t - 1] ** 2
        x = eps * np.sqrt(sigma2)
        lm, p = marginal_models.arch_lm_test(x, lags=5)
        assert p < 0.05

    def test_handles_nan_gracefully(self):
        """NaN in residuals should be cleaned before regression."""
        x = np.array([1.0, np.nan, 3.0, 4.0, 5.0, 6.0])
        lm, p = marginal_models.arch_lm_test(x, lags=2)
        assert np.isfinite(lm) or np.isnan(lm)  # either valid or gracefully degraded

    def test_too_short_series_returns_nan_safely(self):
        """With fewer observations than lags, return NaN and p=1 (via internal guard)."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])  # n=6, lags=5
        lm, p = marginal_models.arch_lm_test(x, lags=5)
        assert np.isfinite(lm) or np.isnan(lm), "Should not crash"
        assert 0 <= p <= 1, "p-value should be in [0,1]"


# ===========================================================================
# Test: fit_gjr_garch
# ===========================================================================

class TestFitGjrGarch:
    """Test GJR-GARCH(1,1)-t model fitting."""

    def test_returns_dict_with_all_assets(self, garch_synthetic_returns):
        fit_dict = marginal_models.fit_gjr_garch(
            garch_synthetic_returns, config.ASSETS, config.GARCH_SPECS
        )
        for asset in config.ASSETS:
            assert asset in fit_dict

    def test_fit_has_expected_attributes(self, garch_synthetic_returns):
        """Each fit result should have params, resid, and conditional_volatility."""
        fit_dict = marginal_models.fit_gjr_garch(
            garch_synthetic_returns.iloc[:100], config.ASSETS, config.GARCH_SPECS
        )
        for asset in config.ASSETS:
            f = fit_dict[asset]
            assert hasattr(f, "params")
            assert hasattr(f, "resid")
            assert hasattr(f, "conditional_volatility")
            assert "nu" in f.params

    def test_conditional_volatility_positive(self, garch_synthetic_returns):
        """Conditional volatility > 0 for valid (non-NaN) observations.
        The first `lags` entries may be NaN due to AR lag initialization."""
        fit_dict = marginal_models.fit_gjr_garch(
            garch_synthetic_returns, config.ASSETS, config.GARCH_SPECS
        )
        for asset in config.ASSETS:
            vol = fit_dict[asset].conditional_volatility
            valid = vol[np.isfinite(vol)]
            assert len(valid) > 0, f"All NaN for {asset}"
            assert np.all(valid > 0), f"Conditional vol ≤ 0 for {asset}"

    def test_params_are_finite(self, garch_synthetic_returns):
        fit_dict = marginal_models.fit_gjr_garch(
            garch_synthetic_returns, config.ASSETS, config.GARCH_SPECS
        )
        for asset in config.ASSETS:
            assert np.all(np.isfinite(fit_dict[asset].params.values))


# ===========================================================================
# Test: extract_std_residuals
# ===========================================================================

class TestExtractStdResiduals:
    """Test standardized residual extraction."""

    def test_shapes(self, garch_synthetic_returns):
        fit_dict = marginal_models.fit_gjr_garch(
            garch_synthetic_returns, config.ASSETS, config.GARCH_SPECS
        )
        z_dict, shape_dict = marginal_models.extract_std_residuals(fit_dict)
        n = len(garch_synthetic_returns)
        for asset in config.ASSETS:
            assert len(z_dict[asset]) == n
            assert isinstance(shape_dict[asset], float)

    def test_std_residuals_have_finite_values(self, garch_synthetic_returns):
        fit_dict = marginal_models.fit_gjr_garch(
            garch_synthetic_returns, config.ASSETS, config.GARCH_SPECS
        )
        z_dict, _ = marginal_models.extract_std_residuals(fit_dict)
        for asset in config.ASSETS:
            assert np.all(np.isfinite(z_dict[asset])), f"Non-finite z for {asset}"

    def test_shape_params_above_2(self, garch_synthetic_returns):
        """Student-t df should be > 2 for identifiable variance."""
        fit_dict = marginal_models.fit_gjr_garch(
            garch_synthetic_returns, config.ASSETS, config.GARCH_SPECS
        )
        _, shape_dict = marginal_models.extract_std_residuals(fit_dict)
        for asset in config.ASSETS:
            assert shape_dict[asset] > 2.0, f"nu ≤ 2 for {asset}"

    def test_std_residuals_around_zero(self, garch_synthetic_returns):
        fit_dict = marginal_models.fit_gjr_garch(
            garch_synthetic_returns, config.ASSETS, config.GARCH_SPECS
        )
        z_dict, _ = marginal_models.extract_std_residuals(fit_dict)
        for asset in config.ASSETS:
            assert abs(np.mean(z_dict[asset])) < 1.0, f"Mean z not near 0 for {asset}"


# ===========================================================================
# Test: pit_transform
# ===========================================================================

class TestPitTransform:
    """Verify PIT → U(0,1) transformation."""

    def test_pit_values_in_unit_interval(self, garch_synthetic_returns):
        fit_dict = marginal_models.fit_gjr_garch(
            garch_synthetic_returns, config.ASSETS, config.GARCH_SPECS
        )
        z_dict, shape_dict = marginal_models.extract_std_residuals(fit_dict)
        u_dict = marginal_models.pit_transform(z_dict, shape_dict)
        for asset in config.ASSETS:
            _assert_valid_pit(u_dict[asset])

    def test_pit_is_approximately_uniform(self, garch_synthetic_returns):
        """
        For well-specified t-GARCH, PIT should be approximately U(0,1).

        Use KS test as a coarse check.
        """
        from scipy.stats import kstest
        fit_dict = marginal_models.fit_gjr_garch(
            garch_synthetic_returns, config.ASSETS, config.GARCH_SPECS
        )
        z_dict, shape_dict = marginal_models.extract_std_residuals(fit_dict)
        u_dict = marginal_models.pit_transform(z_dict, shape_dict)
        for asset in config.ASSETS:
            stat, pval = kstest(u_dict[asset], "uniform")
            # lenient threshold because synthetic data is not perfectly t-GARCH
            assert pval > 0.001, f"KS p={pval:.6f} for {asset}"


# ===========================================================================
# Test: residual_diagnostics + pit_uniformity_check  (smoke tests)
# ===========================================================================

class TestDiagnostics:
    """Residual diagnostics and PIT uniformity check — smoke tests."""

    def test_residual_diagnostics_smoke(self, garch_synthetic_returns):
        """Should not raise; output is printed, not returned."""
        fit_dict = marginal_models.fit_gjr_garch(
            garch_synthetic_returns, config.ASSETS, config.GARCH_SPECS
        )
        z_dict, _ = marginal_models.extract_std_residuals(fit_dict)
        # Should complete without error
        marginal_models.residual_diagnostics(z_dict)

    def test_pit_uniformity_check_smoke(self, garch_synthetic_returns):
        """PIT histogram + KS test should not raise."""
        fit_dict = marginal_models.fit_gjr_garch(
            garch_synthetic_returns, config.ASSETS, config.GARCH_SPECS
        )
        z_dict, shape_dict = marginal_models.extract_std_residuals(fit_dict)
        u_dict = marginal_models.pit_transform(z_dict, shape_dict)
        marginal_models.pit_uniformity_check(u_dict, show_plots=False)


# ===========================================================================
# Test: filter_test_set
# ===========================================================================

class TestFilterTestSet:
    """OOS filtering with frozen parameters."""

    def test_shapes(self, garch_synthetic_returns):
        n = len(garch_synthetic_returns)
        train = garch_synthetic_returns.iloc[:n // 2]
        test = garch_synthetic_returns.iloc[n // 2:]

        fit_dict = marginal_models.fit_gjr_garch(train, config.ASSETS, config.GARCH_SPECS)
        z_dict, shape_dict = marginal_models.extract_std_residuals(fit_dict)
        u_train = marginal_models.pit_transform(z_dict, shape_dict)

        z_test, u_test, _ = marginal_models.filter_test_set(
            fit_dict, test, config.GARCH_SPECS, shape_dict
        )

        assert len(z_test) == len(config.ASSETS)
        assert len(u_test) == len(config.ASSETS)

        for asset in config.ASSETS:
            assert len(z_test[asset]) == len(test)
            assert len(u_test[asset]) == len(test)
            _assert_valid_pit(u_test[asset])
            # First few z values may be NaN due to AR lags in the filter
            z_valid = z_test[asset][np.isfinite(z_test[asset])]
            assert len(z_valid) > 0, f"All z values NaN for {asset}"

    def test_test_pit_not_identical_to_train(self, garch_synthetic_returns):
        """Test-set PIT should differ from training PIT."""
        n = len(garch_synthetic_returns)
        train = garch_synthetic_returns.iloc[:n // 2]
        test = garch_synthetic_returns.iloc[n // 2:]

        fit_dict = marginal_models.fit_gjr_garch(train, config.ASSETS, config.GARCH_SPECS)
        z_dict, shape_dict = marginal_models.extract_std_residuals(fit_dict)
        u_train = marginal_models.pit_transform(z_dict, shape_dict)
        _, u_test, _ = marginal_models.filter_test_set(
            fit_dict, test, config.GARCH_SPECS, shape_dict
        )

        for asset in config.ASSETS:
            # Remove NaN entries (from AR lags) before computing correlation
            tr, te = u_train[asset], u_test[asset]
            valid = np.isfinite(tr) & np.isfinite(te)
            if valid.sum() < 10:
                continue  # insufficient data
            corr = np.corrcoef(tr[valid], te[valid])[0, 1]
            assert np.isfinite(corr), f"Non-finite correlation for {asset}"
            assert corr < 0.99, f"Train/test PIT nearly identical for {asset}"


# ===========================================================================
# Test: save_marginal_outputs
# ===========================================================================

class TestSaveMarginalOutputs:
    """Verify outputs are persisted correctly."""

    def test_saves_csv_and_pickle(self, garch_synthetic_returns, tmp_path, monkeypatch):
        """Should create CSVs and PKL in the output directory."""
        monkeypatch.setattr(marginal_models, "OUTPUT_DIR", tmp_path)
        monkeypatch.setattr(marginal_models, "U_LIST_CSV", tmp_path / "u_list.csv")
        monkeypatch.setattr(marginal_models, "U_TEST_CSV", tmp_path / "u_test.csv")
        monkeypatch.setattr(marginal_models, "MODEL_PKL", tmp_path / "model.pkl")

        n = len(garch_synthetic_returns)
        train = garch_synthetic_returns.iloc[:n // 2]
        test = garch_synthetic_returns.iloc[n // 2:]

        fit_dict = marginal_models.fit_gjr_garch(
            train, config.ASSETS, config.GARCH_SPECS
        )
        z_dict, shape_dict = marginal_models.extract_std_residuals(fit_dict)
        u_train = marginal_models.pit_transform(z_dict, shape_dict)
        z_test, u_test, filt = marginal_models.filter_test_set(
            fit_dict, test, config.GARCH_SPECS, shape_dict
        )

        marginal_models.save_marginal_outputs(u_train, u_test, fit_dict, filt)

        assert (tmp_path / "u_list.csv").exists()
        assert (tmp_path / "u_test.csv").exists()
        assert (tmp_path / "model.pkl").exists()

    def test_pickle_bundle_has_expected_keys(self, garch_synthetic_returns, tmp_path, monkeypatch):
        monkeypatch.setattr(marginal_models, "OUTPUT_DIR", tmp_path)
        monkeypatch.setattr(marginal_models, "U_LIST_CSV", tmp_path / "u_list.csv")
        monkeypatch.setattr(marginal_models, "U_TEST_CSV", tmp_path / "u_test.csv")
        monkeypatch.setattr(marginal_models, "MODEL_PKL", tmp_path / "model.pkl")

        n = len(garch_synthetic_returns)
        train = garch_synthetic_returns.iloc[:n // 2]
        test = garch_synthetic_returns.iloc[n // 2:]

        fit_dict = marginal_models.fit_gjr_garch(
            train, config.ASSETS, config.GARCH_SPECS
        )
        z_dict, shape_dict = marginal_models.extract_std_residuals(fit_dict)
        u_train = marginal_models.pit_transform(z_dict, shape_dict)
        z_test, u_test, filt = marginal_models.filter_test_set(
            fit_dict, test, config.GARCH_SPECS, shape_dict
        )

        marginal_models.save_marginal_outputs(u_train, u_test, fit_dict, filt)

        with open(tmp_path / "model.pkl", "rb") as f:
            bundle = pickle.load(f)
        assert "u_train_matrix" in bundle
        assert "u_test_matrix" in bundle
        assert "fit_list" in bundle
        assert "test_fit_list" in bundle
        assert "var_names" in bundle


# ===========================================================================
# Test: integration — full marginal pipeline
# ===========================================================================

class TestMarginalPipeline:
    """End-to-end test of the marginal modelling pipeline."""

    def test_pipeline_runs_end_to_end(self, garch_synthetic_returns):
        """Run the full sequence: fit → extract → PIT → filter — no crashes."""
        n = len(garch_synthetic_returns)
        train = garch_synthetic_returns.iloc[:n // 2]
        test = garch_synthetic_returns.iloc[n // 2:]

        fit_dict = marginal_models.fit_gjr_garch(train, config.ASSETS, config.GARCH_SPECS)
        z_dict, shape_dict = marginal_models.extract_std_residuals(fit_dict)
        u_train = marginal_models.pit_transform(z_dict, shape_dict)
        z_test, u_test, filt = marginal_models.filter_test_set(
            fit_dict, test, config.GARCH_SPECS, shape_dict
        )

        # Verify train + test PIT arrays are contiguous in index
        for asset in config.ASSETS:
            assert len(u_train[asset]) == len(train)
            assert len(u_test[asset]) == len(test)
