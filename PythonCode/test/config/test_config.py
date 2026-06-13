"""
Tests for config.py — constants, bounds, path resolution, and ensure_dirs().
"""

import os
import sys
from pathlib import Path

import pytest


# ===========================================================================
# Import the module-under-test
# ===========================================================================
import config


# ===========================================================================
# Test: Constants exist and are of the correct type
# ===========================================================================

class TestConstants:
    """Verify that all expected constants are defined and have correct types."""

    def test_seed_is_int(self):
        assert isinstance(config.SEED, int)
        assert config.SEED >= 0

    def test_train_ratio_is_float(self):
        assert isinstance(config.TRAIN_RATIO, float)
        assert 0 < config.TRAIN_RATIO < 1

    def test_assets_is_list_of_strings(self):
        assert isinstance(config.ASSETS, list)
        assert len(config.ASSETS) == 5
        for a in config.ASSETS:
            assert isinstance(a, str)

    def test_garch_specs_has_all_assets(self):
        for asset in config.ASSETS:
            assert asset in config.GARCH_SPECS
            spec = config.GARCH_SPECS[asset]
            assert "arma" in spec
            assert "garch" in spec
            assert len(spec["arma"]) == 2
            assert len(spec["garch"]) == 2

    def test_distribution_is_string(self):
        assert isinstance(config.DISTRIBUTION, str)

    def test_family_set_is_list_of_ints(self):
        assert isinstance(config.FAMILY_SET, list)
        for f in config.FAMILY_SET:
            assert isinstance(f, int)

    def test_dvine_order_matches_assets(self):
        assert len(config.DVINE_ORDER) == len(config.ASSETS)
        assert len(config.DVINE_NAMES) == len(config.ASSETS)

    def test_print_width_is_positive_int(self):
        assert isinstance(config.PRINT_WIDTH, int)
        assert config.PRINT_WIDTH > 0


# ===========================================================================
# Test: Path resolution
# ===========================================================================

class TestPaths:
    """Verify path constants resolve correctly."""

    def test_project_root_exists(self):
        assert config.PROJECT_ROOT.exists()

    def test_data_dir_exists_or_creatable(self):
        # The directory may not exist yet, but the parent should
        assert config.DATA_DIR.name == "data"

    def test_data_csv_points_to_file(self):
        assert str(config.DATA_CSV).endswith(".csv")

    def test_output_filenames_match_pattern(self):
        assert str(config.U_LIST_CSV).endswith("11_u_list.csv")
        assert str(config.U_TEST_CSV).endswith("11_u_test.csv")
        assert str(config.MODEL_PKL).endswith(".pkl")


# ===========================================================================
# Test: Optimisation bounds
# ===========================================================================

class TestBounds:
    """Verify that FIGAS and GAS bounds are well-formed."""

    @pytest.mark.parametrize("bounds_dict", ["FIGAS_BOUNDS", "GAS_BOUNDS"])
    def test_bounds_are_dict_of_tuples(self, bounds_dict):
        bd = getattr(config, bounds_dict)
        assert isinstance(bd, dict)
        for key, (lo, hi) in bd.items():
            assert lo < hi, f"{bounds_dict}[{key}]: lower >= upper"

    @pytest.mark.parametrize("key, lo, hi", [
        ("mu", -8.0, 8.0),
        ("alpha", 0.001, 0.35),
        ("beta", 0.001, 0.99),
        ("d", 0.05, 0.49),
        ("kappa", 2.1, 30.0),
    ])
    def test_figas_bounds_values(self, key, lo, hi):
        assert config.FIGAS_BOUNDS[key] == (lo, hi)

    @pytest.mark.parametrize("key, lo, hi", [
        ("mu", -15.0, 15.0),
        ("alpha", 0.001, 0.4),
        ("beta", 0.01, 0.99),
        ("kappa", 2.1, 30.0),
    ])
    def test_gas_bounds_values(self, key, lo, hi):
        assert config.GAS_BOUNDS[key] == (lo, hi)


# ===========================================================================
# Test: Numerical stability constants
# ===========================================================================

class TestNumericalStability:
    """Verify PDF floor, score clamp, and step-size constants."""

    def test_pdf_floor_is_positive(self):
        assert config.PDF_FLOOR > 0

    def test_score_clamp_is_positive(self):
        assert config.SCORE_CLAMP > 0

    def test_f_diff_h_is_small_positive(self):
        assert 0 < config.F_DIFF_H < 0.01


# ===========================================================================
# Test: ensure_dirs
# ===========================================================================

class TestEnsureDirs:
    """Verify directory-creation helper."""

    def test_ensure_dirs_creates_data_dir(self, tmp_path, monkeypatch):
        """Monkeypatch DATA_DIR to a temp location so we don't touch real FS."""
        fake_data = tmp_path / "data"
        monkeypatch.setattr(config, "DATA_DIR", fake_data)
        monkeypatch.setattr(config, "OUTPUT_DIR", fake_data)
        config.ensure_dirs()
        assert fake_data.exists()

    def test_ensure_dirs_is_idempotent(self, tmp_path, monkeypatch):
        fake_data = tmp_path / "data"
        monkeypatch.setattr(config, "DATA_DIR", fake_data)
        monkeypatch.setattr(config, "OUTPUT_DIR", fake_data)
        config.ensure_dirs()
        config.ensure_dirs()  # should not raise
        assert fake_data.exists()
