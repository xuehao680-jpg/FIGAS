## ADDED Requirements

### Requirement: Kendall's tau correlation matrix

The system SHALL compute the Kendall rank correlation matrix for the 5-dimensional PIT-transformed data to assess dependence structure.

#### Scenario: Compute Kendall's tau
- **WHEN** u_train matrix (1944x5) is provided
- **THEN** 5x5 Kendall's tau matrix is computed and printed
- **AND** all pairwise tau values are in range [-1, 1]

### Requirement: R-Vine structure selection

The system SHALL automatically select the optimal R-Vine Copula structure (unrestricted tree) using AIC criterion, considering families {Gaussian(1), t(2), Clayton(3), Gumbel(4), Frank(5), SJC(10)}.

#### Scenario: Fit R-Vine
- **WHEN** pseudo-observations (via pobs) are provided
- **THEN** RVineStructureSelect with type="RVine" and selectioncrit="AIC" is called
- **AND** model summary (pair-copula families, parameters, logLik, AIC, BIC) is printed

### Requirement: C-Vine structure selection

The system SHALL select the optimal C-Vine Copula structure (star-shaped) using AIC criterion with the same family set.

#### Scenario: Fit C-Vine
- **WHEN** pseudo-observations are provided
- **THEN** RVineStructureSelect with type="CVine" is called
- **AND** model summary is printed

### Requirement: D-Vine structure selection with TSP ordering

The system SHALL determine the optimal D-Vine variable order by solving the Traveling Salesman Problem on the 1 - |tau| distance matrix.

#### Scenario: TSP-based D-Vine ordering
- **WHEN** Kendall's tau matrix is computed
- **THEN** distance matrix D = 1 - |tau| is calculated
- **AND** TSP is solved via `repetitive_nn` method
- **AND** optimal variable order is printed
- **AND** D-Vine matrix representation is constructed via D2RVine

#### Scenario: Select pair-copulas for D-Vine structure
- **WHEN** D-Vine matrix and pseudo-observations are provided
- **THEN** RVineCopSelect is called with families {1,2,3,4,5,10}
- **AND** summary is printed

### Requirement: Compare three Vine structures

The system SHALL compare R-Vine, C-Vine, and D-Vine models by log-likelihood, AIC, and BIC, and identify the optimal structure.

#### Scenario: Model comparison table
- **WHEN** all three Vine models are fitted
- **THEN** a comparison DataFrame with columns [Model, LogLik, AIC, BIC] is printed
- **AND** the best model by AIC is identified and printed
