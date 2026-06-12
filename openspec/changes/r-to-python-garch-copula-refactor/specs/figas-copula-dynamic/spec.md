## ADDED Requirements

### Requirement: Fractional differencing weights

The system SHALL compute fractionally differenced weights psi_j = (j-1-d)/j * psi_{j-1} with psi_1 = -d, truncated at L=100 lags, for a given fractional parameter d in (0.05, 0.49).

#### Scenario: Compute FIGAS weights
- **WHEN** d = 0.2 is provided
- **THEN** psi_1 = -0.2, psi_2 = (1-0.2)/2 * (-0.2) = -0.08, ...
- **AND** the weight sequence decays towards zero
- **AND** length of psi vector equals min(100, T)

### Requirement: FIGAS recursion for time-varying Copula parameter

The system SHALL implement the FIGAS(1,1,d) recursion:
  y_{t+1} = beta*y_t + alpha*s_t - sum_{j=1}^L psi_j * y_{t+1-j}
  g_{t+1} = mu + y_{t+1}
where s_t is the scaled score and y_t is the fractional-filtered innovation.

#### Scenario: FIGAS update with t-Copula family
- **WHEN** u1, u2 pseudo-observations and family=2 (t-Copula) are provided
- **THEN** g_t is transformed to rho_t via tanh with clipping to [-0.98, 0.98]
- **AND** score s_t = nabla / sqrt(Info) is computed analytically (without numDeriv)
- **AND** y_{t+1} is updated using beta*y_t + alpha*s_t - sum(psi_j * y_{t+1-j})

#### Scenario: FIGAS update with Survival Gumbel (family=14)
- **WHEN** family=14
- **THEN** g_t is transformed to theta_t = exp(g_t) + 1.0001, clipped to [1.0001, 30]
- **AND** score is computed via central finite difference (eps=1e-5)

#### Scenario: FIGAS update with Clayton 90-rotated (family=23)
- **WHEN** family=23
- **THEN** g_t is transformed to theta_t = -exp(g_t) - 1e-4, clipped to [-30, -1e-4]
- **AND** score is computed via central finite difference

#### Scenario: FIGAS update with Clayton (family=3)
- **WHEN** family=3
- **THEN** g_t is transformed to theta_t = exp(g_t) + 1e-4, clipped to [1e-4, 30]
- **AND** score is computed via central finite difference

### Requirement: NA firewall for numerical stability

The system SHALL clamp scores to [-10, 10], replace NA/Inf/NaN with 0, and replace zero/negative PDF values with 1e-10 before taking log.

#### Scenario: Handle degenerate PDF
- **WHEN** BiCopPDF returns value <= 0 or raises error
- **THEN** PDF value is set to 1e-10
- **AND** log-likelihood contribution is log(1e-10) = -23.03

#### Scenario: Clamp extreme scores
- **WHEN** score computation returns value > 10 or NaN
- **THEN** score is clamped to 10 (or -10 if negative)
- **AND** NaN/Inf scores are replaced with 0

### Requirement: h-function computation with fallback

The system SHALL compute BiCopHfunc for each time step and fall back to u1[t], u2[t] (independence copula) if computation fails.

#### Scenario: Robust h-function with error handling
- **WHEN** BiCopHfunc raises an error for any time t
- **THEN** h1[t] = u1[t], h2[t] = u2[t] (independence fallback)
- **AND** remaining time steps are unaffected

### Requirement: FIGAS parameter estimation via maximum likelihood

The system SHALL estimate FIGAS parameters [mu, alpha, beta, d] (plus kappa for t-Copula) by minimizing negative log-likelihood using L-BFGS-B with parameter bounds and multiple random initial values.

#### Scenario: Optimize FIGAS parameters
- **WHEN** u1, u2 sequences and family are provided
- **THEN** initial static BiCopEst is computed for starting values
- **AND** start_mu is derived via the inverse link function
- **AND** alpha ~ Uniform(0.02, 0.08), beta ~ Uniform(0.50, 0.90), d ~ Uniform(0.10, 0.35)
- **AND** nlminb (or L-BFGS-B) optimizes with bounds: mu in [-8, 8], alpha in [0.001, 0.35], beta in [0.001, 0.99], d in [0.05, 0.49]
- **AND** final filtered result is computed with optimal parameters

#### Scenario: Print FIGAS results
- **WHEN** optimization completes
- **THEN** static parameters (initial) and dynamic parameters (estimated) are printed
- **AND** edge log-likelihood is printed
