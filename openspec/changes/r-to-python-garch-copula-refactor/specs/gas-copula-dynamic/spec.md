## ADDED Requirements

### Requirement: GAS(1,1) recursion for time-varying Copula parameter

The system SHALL implement the standard GAS(1,1) recursion:
  g_{t+1} = mu + beta * (g_t - mu) + alpha * s_t
where s_t is the scaled score at time t.

#### Scenario: GAS update with t-Copula family
- **WHEN** u1, u2 pseudo-observations and family=2 are provided
- **THEN** g_t is transformed to rho_t via tanh with clipping to [-0.98, 0.98]
- **AND** score is computed analytically using t-Copula score formula
- **AND** g_{t+1} = mu + beta*(g_t - mu) + alpha*s_t

#### Scenario: GAS update with non-Gaussian families
- **WHEN** family is 14, 23, or 3
- **THEN** score is computed via numDeriv-like central difference (eps=1e-5)
- **AND** g_t is transformed via the family-specific link function
- **AND** scores are clamped to [-10, 10]

### Requirement: GAS parameter estimation

The system SHALL estimate GAS(1,1) parameters [mu, alpha, beta] (plus kappa for t-Copula) via MLE using L-BFGS-B optimization.

#### Scenario: Optimize GAS parameters
- **WHEN** u1, u2 sequences and family are provided
- **THEN** initial mu is derived from static BiCopEst
- **AND** optimization bounds: mu in [-15, 15], alpha in [0.001, 0.4], beta in [0.01, 0.99]
- **AND** estimated GAS parameters and edge log-likelihood are printed

### Requirement: GAS h-function output

The system SHALL compute h-functions (conditional CDF) from the fitted GAS model for use in higher-tree Copula estimation.

#### Scenario: Compute GAS h-functions
- **WHEN** GAS model is fitted
- **THEN** h1[t] and h2[t] are computed via BiCopHfunc at each time t
- **AND** h1, h2 arrays have same length as input
