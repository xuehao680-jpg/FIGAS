#!/usr/bin/env python3
"""Quick import and syntax validation."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

print("Testing imports...")

import config;          print("  config OK")
import utils;           print("  utils OK")
import marginal_models; print("  marginal_models OK")
import figas_filter;    print("  figas_filter OK")
import gas_filter;      print("  gas_filter OK")
import vine_copula;     print("  vine_copula OK")

print("All imports passed!")

# Test key function signatures
print("\nTesting key functions exist...")
assert hasattr(utils, 'load_and_split_data'), 'missing load_and_split_data'
assert hasattr(utils, 'descriptive_stats'), 'missing descriptive_stats'
assert hasattr(utils, 'run_diagnostic_tests'), 'missing run_diagnostic_tests'
assert hasattr(utils, 'ljung_box_matrix'), 'missing ljung_box_matrix'
assert hasattr(marginal_models, 'fit_gjr_garch'), 'missing fit_gjr_garch'
assert hasattr(marginal_models, 'pit_transform'), 'missing pit_transform'
assert hasattr(marginal_models, 'save_marginal_outputs'), 'missing save_marginal_outputs'
assert hasattr(figas_filter, 'filter_figas'), 'missing filter_figas'
assert hasattr(figas_filter, 'estimate_figas_params'), 'missing estimate_figas_params'
assert hasattr(gas_filter, 'filter_gas'), 'missing filter_gas'
assert hasattr(gas_filter, 'estimate_gas_params'), 'missing estimate_gas_params'
assert hasattr(vine_copula, 'DVINE_STRUCTURE'), 'missing DVINE_STRUCTURE'
assert hasattr(vine_copula, 'train_dvine'), 'missing train_dvine'
assert hasattr(vine_copula, 'compare_models'), 'missing compare_models'

print("All function checks passed!")

# Quick data loading test
print("\nTesting data loading...")
returns_full, returns_train, returns_test, date_all, date_train, date_test = utils.load_and_split_data()
print(f"  Full: {returns_full.shape}, Train: {returns_train.shape}, Test: {returns_test.shape}")
assert returns_train.shape[0] == 1944, f"Expected 1944 train rows, got {returns_train.shape[0]}"
assert returns_test.shape[0] == 486, f"Expected 486 test rows, got {returns_test.shape[0]}"
print("  Data dimensions match R code!")

print("\n=== ALL TESTS PASSED ===")
