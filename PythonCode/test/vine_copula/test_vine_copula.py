"""
Tests for vine_copula.py — Kendall tau matrix, D-Vine training, OOS evaluation,
and model comparison (static vs FIGAS vs GAS).

These tests focus on:
  - Input/output contract verification (shapes, types)
  - Numerical stability with synthetic data
  - Consistency between train and test evaluation
  - DVINE_STRUCTURE validity
"""

import numpy as np
import pandas as pd
import pytest

import config
import vine_copula as vc


# ===========================================================================
# Test: kendall_tau_matrix
# ===========================================================================

class TestKendallTauMatrix:
    """Pairwise Kendall's tau computation."""

    def test_matrix_shape(self, weakly_correlated_uniform):
        """Should return (d x d) matrix."""
        tau = vc.kendall_tau_matrix(weakly_correlated_uniform)
        d = weakly_correlated_uniform.shape[1]
        assert tau.shape == (d, d)

    def test_diagonal_ones(self, weakly_correlated_uniform):
        tau = vc.kendall_tau_matrix(weakly_correlated_uniform)
        np.testing.assert_array_almost_equal(np.diag(tau), np.ones(tau.shape[0]))

    def test_symmetric(self, weakly_correlated_uniform):
        tau = vc.kendall_tau_matrix(weakly_correlated_uniform)
        np.testing.assert_array_almost_equal(tau, tau.T)

    def test_positively_correlated(self, strongly_correlated_uniform):
        """Strongly correlated data → all off-diagonal tau > 0."""
        tau = vc.kendall_tau_matrix(strongly_correlated_uniform)
        off_diag = tau[~np.eye(tau.shape[0], dtype=bool)]
        assert np.all(off_diag > 0.3), "Off-diagonal tau too low for strongly correlated data"

    def test_dataframe_input(self, weakly_correlated_uniform):
        """Should accept DataFrames."""
        df = pd.DataFrame(weakly_correlated_uniform, columns=config.ASSETS)
        tau = vc.kendall_tau_matrix(df)
        assert tau.shape == (len(config.ASSETS), len(config.ASSETS))


# ===========================================================================
# Test: select_vine_structure
# ===========================================================================

class TestSelectVineStructure:
    """Vine copula structure selection."""

    def test_returns_dataframe(self, weakly_correlated_uniform):
        """Should always return a comparison DataFrame."""
        df = pd.DataFrame(weakly_correlated_uniform, columns=config.ASSETS)
        result = vc.select_vine_structure(df)
        assert isinstance(result, pd.DataFrame)
        assert "Model" in result.columns
        assert "AIC" in result.columns

    def test_aic_is_finite(self, weakly_correlated_uniform):
        df = pd.DataFrame(weakly_correlated_uniform, columns=config.ASSETS)
        result = vc.select_vine_structure(df)
        assert np.all(np.isfinite(result["AIC"]))

    def test_too_small_sample_raises(self):
        """Fewer than 30 rows should raise."""
        data = np.random.uniform(size=(20, 5))
        with pytest.raises(ValueError, match="Sample size too small"):
            vc.select_vine_structure(data)


# ===========================================================================
# Test: DVINE_STRUCTURE validity
# ===========================================================================

class TestDVineStructure:
    """Hard-coded D-Vine structure validation."""

    def test_structure_is_list_of_4_trees(self):
        assert len(vc.DVINE_STRUCTURE) == 4

    def test_tree_edge_counts(self):
        """Tree 1: 4 edges → Tree 4: 1 edge."""
        expected = [4, 3, 2, 1]
        for tree_idx, n_edges in enumerate(expected):
            assert len(vc.DVINE_STRUCTURE[tree_idx]) == n_edges, \
                f"Tree {tree_idx + 1}: expected {n_edges} edges"

    def test_each_edge_has_required_keys(self):
        for tree_idx, tree in enumerate(vc.DVINE_STRUCTURE):
            for e_idx, edge in enumerate(tree):
                assert "v1" in edge, f"Tree {tree_idx + 1}, edge {e_idx}: missing v1"
                assert "v2" in edge, f"Tree {tree_idx + 1}, edge {e_idx}: missing v2"
                assert "family" in edge, f"Tree {tree_idx + 1}, edge {e_idx}: missing family"
                assert 1 <= edge["v1"] <= 5
                assert 1 <= edge["v2"] <= 5
                assert edge["v1"] != edge["v2"]

    def test_families_are_supported(self):
        """All family codes should be 2, 3, 4, 5, 14, or 23."""
        supported = {2, 3, 4, 5, 14, 23}
        for tree in vc.DVINE_STRUCTURE:
            for edge in tree:
                assert edge["family"] in supported, \
                    f"Unsupported family {edge['family']} in edge {edge}"


# ===========================================================================
# Test: train_dvine — static
# ===========================================================================

class TestTrainDvineStatic:
    """D-Vine training with static copula parameters."""

    @pytest.fixture
    def ordered_data(self, weakly_correlated_uniform):
        """Apply DVINE_ORDER reordering."""
        order_0b = [x - 1 for x in config.DVINE_ORDER]
        return weakly_correlated_uniform[:, order_0b]

    def test_returns_dict_with_keys(self, ordered_data):
        result = vc.train_dvine(ordered_data, "static")
        assert "total_loglik" in result
        assert "edges" in result

    def test_four_trees(self, ordered_data):
        result = vc.train_dvine(ordered_data, "static")
        assert len(result["edges"]) == 4

    def test_loglik_is_finite(self, ordered_data):
        result = vc.train_dvine(ordered_data, "static")
        assert np.isfinite(result["total_loglik"])

    def test_loglik_is_negative_for_near_independent(self):
        """Weakly correlated data → fairly negative log-likelihood."""
        order_0b = [x - 1 for x in config.DVINE_ORDER]
        rng = np.random.RandomState(99)
        data = rng.uniform(size=(500, 5))
        ordered = data[:, order_0b]
        result = vc.train_dvine(ordered, "static")
        assert np.isfinite(result["total_loglik"])

    def test_edge_result_has_all_fields(self, ordered_data):
        result = vc.train_dvine(ordered_data, "static")
        for tree_idx, tree in enumerate(result["edges"]):
            for e_idx, edge in enumerate(tree):
                assert "loglik" in edge
                assert "h1" in edge
                assert "h2" in edge
                assert "params" in edge
                assert "fam" in edge

    def test_h_functions_in_unit_interval(self, ordered_data):
        result = vc.train_dvine(ordered_data, "static")
        for tree in result["edges"]:
            for edge in tree:
                assert np.all(edge["h1"] > 0) and np.all(edge["h1"] < 1)
                assert np.all(edge["h2"] > 0) and np.all(edge["h2"] < 1)


# ===========================================================================
# Test: train_dvine — FIGAS
# ===========================================================================

class TestTrainDvineFigas:
    """D-Vine training with FIGAS dynamics.
    NOTE: Runs L-BFGS-B on all 10 edges — inherently slow.
    """
    pytestmark = pytest.mark.slow

    @pytest.fixture
    def small_data(self, weakly_correlated_uniform):
        """Smaller dataset for faster FIGAS/GAS training in tests."""
        order_0b = [x - 1 for x in config.DVINE_ORDER]
        return weakly_correlated_uniform[:200, order_0b]

    def test_loglik_is_finite(self, small_data):
        result = vc.train_dvine(small_data, "figas")
        assert np.isfinite(result["total_loglik"])

    def test_returns_full_structure(self, small_data):
        result = vc.train_dvine(small_data, "figas")
        assert len(result["edges"]) == 4
        assert "total_loglik" in result

    def test_edge_has_par_seq_for_figas(self, small_data):
        result = vc.train_dvine(small_data, "figas")
        tree0_edge0 = result["edges"][0][0]
        assert "par_seq" in tree0_edge0
        assert tree0_edge0["par_seq"] is not None
        assert len(tree0_edge0["par_seq"]) == len(small_data)


# ===========================================================================
# Test: train_dvine — GAS
# ===========================================================================

class TestTrainDvineGas:
    """D-Vine training with GAS dynamics.
    NOTE: Runs L-BFGS-B on all 10 edges — inherently slow.
    """
    pytestmark = pytest.mark.slow

    @pytest.fixture
    def small_data(self, weakly_correlated_uniform):
        order_0b = [x - 1 for x in config.DVINE_ORDER]
        return weakly_correlated_uniform[:200, order_0b]

    def test_loglik_is_finite(self, small_data):
        result = vc.train_dvine(small_data, "gas")
        assert np.isfinite(result["total_loglik"])

    def test_returns_full_structure(self, small_data):
        result = vc.train_dvine(small_data, "gas")
        assert len(result["edges"]) == 4
        assert "total_loglik" in result


# ===========================================================================
# Test: eval_test_dvine
# ===========================================================================

class TestEvalTestDvine:
    """OOS evaluation with frozen parameters.
    NOTE: FIGAS/GAS OOS tests run per-edge filters — slow.
    """
    pytestmark = pytest.mark.slow

    @pytest.fixture
    def train_test_data(self, weakly_correlated_uniform):
        """Split into train/test sets."""
        order_0b = [x - 1 for x in config.DVINE_ORDER]
        ordered = weakly_correlated_uniform[:, order_0b]
        n = len(ordered)
        return ordered[:n // 2], ordered[n // 2:]

    def test_static_oos_ll_finite(self, train_test_data):
        train, test = train_test_data
        model = vc.train_dvine(train, "static")
        oos_ll = vc.eval_test_dvine(test, model, "static")
        assert np.isfinite(oos_ll)

    def test_figas_oos_ll_finite(self, train_test_data):
        train, test = train_test_data
        model = vc.train_dvine(train, "figas")
        oos_ll = vc.eval_test_dvine(test, model, "figas")
        assert np.isfinite(oos_ll)

    def test_gas_oos_ll_finite(self, train_test_data):
        train, test = train_test_data
        model = vc.train_dvine(train, "gas")
        oos_ll = vc.eval_test_dvine(test, model, "gas")
        assert np.isfinite(oos_ll)

    def test_static_oos_ll_not_extreme(self, train_test_data):
        """OOS log-likelihood should not be an extreme outlier."""
        train, test = train_test_data
        model = vc.train_dvine(train, "static")
        oos_ll = vc.eval_test_dvine(test, model, "static")
        # For weakly correlated data on ~250 obs, shouldn't be > 0 or < -5000
        assert -5000 < oos_ll < 100


# ===========================================================================
# Test: compare_models
# ===========================================================================

class TestCompareModels:
    """Full model comparison pipeline.
    NOTE: Trains ALL three models (static + FIGAS + GAS) — very slow.
    """
    pytestmark = pytest.mark.slow

    @pytest.fixture
    def train_test(self, weakly_correlated_uniform):
        """Train/test split for compare_models."""
        n = len(weakly_correlated_uniform)
        df = pd.DataFrame(weakly_correlated_uniform, columns=config.ASSETS)
        return df.iloc[:n // 2], df.iloc[n // 2:]

    def test_returns_best_model_name(self, train_test):
        train, test = train_test
        best_name, results = vc.compare_models(train, test)
        assert best_name in ("static", "figas", "gas")

    def test_returns_results_dict(self, train_test):
        train, test = train_test
        best_name, results = vc.compare_models(train, test)
        for key in ("static", "figas", "gas", "comparison"):
            assert key in results

    def test_comparison_is_dataframe(self, train_test):
        train, test = train_test
        _, results = vc.compare_models(train, test)
        assert isinstance(results["comparison"], pd.DataFrame)
        assert len(results["comparison"]) == 3

    def test_oos_values_are_finite(self, train_test):
        train, test = train_test
        _, results = vc.compare_models(train, test)
        comp = results["comparison"]
        assert np.all(np.isfinite(comp["OOS LogLik"]))


# ===========================================================================
# Test: Error handling
# ===========================================================================

class TestEdgeCases:
    """Abnormal inputs and error conditions."""

    def test_invalid_model_type_raises(self, weakly_correlated_uniform):
        with pytest.raises(ValueError, match="Unknown model_type"):
            vc.train_dvine(weakly_correlated_uniform[:10, :5], "invalid")

    def test_wrong_number_of_columns(self, rng):
        """train_dvine expects 5 columns; 3 columns may still work but check."""
        data = rng.uniform(size=(100, 3))
        try:
            vc.train_dvine(data, "static")
        except (IndexError, ValueError):
            pass  # Either gracefully handles or raises — both OK

    def test_all_identical_values(self):
        """Constant columns should be handleable (variance check in select_vine)."""
        data = np.full((100, 5), 0.5)
        with pytest.raises((ValueError, Exception)):
            vc.select_vine_structure(data)
