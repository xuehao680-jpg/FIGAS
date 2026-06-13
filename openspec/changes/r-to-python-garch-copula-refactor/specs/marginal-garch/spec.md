## ADDED Requirements

### Requirement: ARMA-GJR-GARCH with Student-t distribution

The system SHALL fit an ARMA(p,q)-GJR-GARCH(1,1) model with Student-t distributed innovations to each asset's return series using custom joint maximum likelihood estimation via scipy L-BFGS-B.

Note: Python's `arch` library (v8.0.0) does NOT support MA terms in the mean equation. Custom MLE was necessary to fully replicate R's `rugarch` ARMA-GJR-GARCH specification.

#### Scenario: Fit ARMA-GJR-GARCH on training set
- **WHEN** training set returns (1944 observations) are provided for each of the 5 assets
- **THEN** the system estimates all parameters (mu, ar1..p, ma1..q, omega, alpha1, gamma1, beta1, nu) via joint MLE with multi-start L-BFGS-B
- **AND** parameter estimates are printed with 6 decimal precision

#### Scenario: Extract standardized residuals
- **WHEN** ARMA-GJR-GARCH model is fitted
- **THEN** standardized residuals z_t = epsilon_t / sigma_t are extracted (where epsilon_t accounts for ARMA mean dynamics)
- **AND** residuals pass Ljung-Box test at lags 5, 10, 15 (p > 0.05)
- **AND** squared residuals pass ARCH-LM test at lags 5, 10 (p > 0.05)

### Requirement: PIT transformation to uniform margins

The system SHALL transform standardized residuals to uniform (0,1) observations via the probability integral transform using the fitted Student-t CDF.

#### Scenario: PIT with KS test
- **WHEN** standardized residuals z_t and estimated shape parameter are provided
- **THEN** u_t = F_t(z_t; df=shape) is computed
- **AND** u_t histogram is plotted and visually inspected for uniformity
- **AND** Kolmogorov-Smirnov test against Uniform(0,1) is performed (p > 0.05 expected)

### Requirement: Test set filtering with fixed parameters

The system SHALL filter the test set (486 observations) through the fitted ARMA-GJR-GARCH model using fixed parameters from training, without re-estimation.

#### Scenario: Filter test set
- **WHEN** fitted ARMA-GJR-GARCH model and test set returns are provided
- **THEN** conditional volatility sigma_t is computed using fixed parameters via `_filter_single_arma_gjr_garch_t()`
- **AND** standardized residuals for test set are extracted
- **AND** PIT transformation is applied using training set shape parameter nu

### Requirement: Save marginal model outputs

The system SHALL save u_train (1944x5), u_test (486x5), fitted models, and test filter objects to disk for downstream Copula modeling.

#### Scenario: Save to data directory
- **WHEN** all 5 assets' GJR-GARCH models are fitted and PIT transformed
- **THEN** u_train_matrix is saved as CSV to data/11_u_list.csv
- **AND** u_test_matrix is saved as CSV to data/11_u_test.csv
- **AND** fitted model objects (fit_list, test_fit_list) are saved as Pickle to data/11Marginal_Full_Data.pkl
