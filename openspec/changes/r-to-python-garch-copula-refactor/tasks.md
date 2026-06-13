## 1. Project Setup

- [ ] 1.1 Create config.py with all paths (data_dir, output_dir), model parameters (ARMA/GARCH orders), random seed
- [ ] 1.2 Create requirements.txt with dependencies: numpy, pandas, scipy, arch, matplotlib, pyvinecopula
- [ ] 1.3 Verify all dependencies install correctly in the WSL Python environment
- [ ] 1.4 Create __init__.py in PythonCode/

## 2. Utilities Module (utils.py)

- [ ] 2.1 Implement `load_and_split_data()`: read CSV, extract 5 return columns + date, chrono 80/20 split, set seed
- [ ] 2.2 Implement `descriptive_stats()`: mean, std, skewness, kurtosis table
- [ ] 2.3 Implement `plot_return_series()`: 5-line ggplot2-style time series with yearly ticks
- [ ] 2.4 Implement `run_diagnostic_tests()`: JB test, ADF test per asset, print formatted table
- [ ] 2.5 Implement `arch_lm_test()`: custom ARCH-LM (fit ARMA → extract residuals → ArchTest lags=5)
- [ ] 2.6 Implement `ljung_box_matrix()`: 4x5 p-value matrix for lags 1-4
- [ ] 2.7 Implement `select_arma_garch_orders()`: auto.arima (p≤3, q≤5) + GARCH order selection from {(1,1),(1,2),(2,1)}

## 3. Marginal GARCH Models (marginal_models.py)

- [ ] 3.1 Define specs dict: {asset: {armaOrder, garchOrder, dist}} — hardcode from R code results
- [x] 3.2 Implement `fit_gjr_garch()`: fit ARMA(p,q)-GJR-GARCH(1,1)+std via custom joint MLE (L-BFGS-B multi-start), return ARMAGARCHResult. **Changed from arch library (no MA support) to custom scipy.optimize MLE**
- [ ] 3.3 Implement `extract_std_residuals()`: extract standardized residuals from fitted model
- [ ] 3.4 Implement `pit_transform()`: PIT using scipy.stats.t.cdf with estimated shape parameter
- [ ] 3.5 Implement `residual_diagnostics()`: Ljung-Box (lags 5,10,15) and ARCH-LM (lags 5,10) on standardized residuals
- [x] 3.6 Implement `filter_test_set()`: ugarchfilter equivalent — `_filter_single_arma_gjr_garch_t()` with fixed parameters from training on test set
- [ ] 3.7 Implement `save_marginal_outputs()`: save u_train.csv, u_test.csv, fitted_models.pkl

## 4. Vine Copula Selection (vine_copula.py)

- [ ] 4.1 Implement `kendall_tau_matrix()`: compute 5x5 Kendall's tau
- [ ] 4.2 Implement `select_rvine()`: RVineStructureSelect (type=R, AIC, families 1,2,3,4,5,10)
- [ ] 4.3 Implement `select_cvine()`: CVine equivalent
- [ ] 4.4 Implement `select_dvine_tsp()`: TSP on 1-|tau| distance matrix → optimal order → D2RVine → RVineCopSelect
- [ ] 4.5 Implement `compare_vine_models()`: R/C/D-vine comparison table [LogLik, AIC, BIC], identify best

## 5. FIGAS Filter (figas_filter.py)

- [ ] 5.1 Implement `frac_weights()`: psi_j sequence with truncation at L=100
- [ ] 5.2 Implement `filter_figas()`: core FIGAS recursion with family-specific link functions and scores
- [ ] 5.3 Implement t-Copula (family=2) analytical score: nabla/Info with dot_rho derivative
- [ ] 5.4 Implement Central finite difference score for families 3, 14, 23
- [ ] 5.5 Implement NA-firewall: clamp scores [-10,10], replace NA→0, PDF≤0→1e-10
- [ ] 5.6 Implement h-function computation with try/except fallback to independence
- [ ] 5.7 Implement `estimate_figas_params()`: L-BFGS-B optimization with bounds and multi-start init

## 6. GAS Filter (gas_filter.py)

- [ ] 6.1 Implement `filter_gas()`: standard GAS(1,1) recursion g_{t+1}=mu+beta*(g_t-mu)+alpha*s_t
- [ ] 6.2 Implement family-specific score functions (reuse from figas_filter)
- [ ] 6.3 Implement h-function computation for GAS
- [ ] 6.4 Implement `estimate_gas_params()`: L-BFGS-B optimization

## 7. D-Vine Training + OOS Evaluation

- [ ] 7.1 Define D-Vine 4-tree structure as nested list of edges with (v1, v2, cond, family)
- [ ] 7.2 Implement `train_dvine()`: sequential tree-by-tree training supporting static/figas/gas modes
- [ ] 7.3 Implement `eval_test_dvine()`: OOS evaluation on test set, propagating h-functions through trees
- [ ] 7.4 Implement final comparison: print OOS log-likelihoods for static vs FIGAS vs GAS, identify best

## 8. Main Pipeline (main.py)

- [ ] 8.1 Wire up full pipeline: data → utils → marginal → vine select → D-Vine dynamic → OOS eval
- [ ] 8.2 Add progress logging with section headers (matching R code output format)
- [ ] 8.3 Verify output matches R code results (same seed, same data split, similar parameter estimates)

## 9. Testing & Validation

- [ ] 9.1 Run main.py end-to-end and verify no runtime errors
- [ ] 9.2 Compare descriptive statistics output with R code results
- [ ] 9.3 Compare GJR-GARCH parameter estimates with R code results (within tolerance)
- [ ] 9.4 Compare D-Vine model selection results (same best model?)
- [ ] 9.5 Compare FIGAS OOS log-likelihood with R code output
