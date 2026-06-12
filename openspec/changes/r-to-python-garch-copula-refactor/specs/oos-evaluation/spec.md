## ADDED Requirements

### Requirement: D-Vine tree structure definition

The system SHALL define a 4-tree D-Vine structure with 10 pair-copula edges, each specifying (variable1, variable2, conditioning set, Copula family).

#### Scenario: Define D-Vine structure
- **WHEN** D-Vine is selected as the optimal model
- **THEN** Tree 1 has 4 edges: (1,2|family=14), (2,3|family=2), (3,4|family=2), (4,5|family=14)
- **AND** Tree 2 has 3 edges: (1,3|2|family=2), (2,4|3|family=2), (3,5|4|family=2)
- **AND** Tree 3 has 2 edges: (1,4|2,3|family=2), (2,5|3,4|family=23)
- **AND** Tree 4 has 1 edge: (1,5|2,3,4|family=3)

### Requirement: Sequential D-Vine training

The system SHALL train the D-Vine Copula sequentially from Tree 1 to Tree 4, where each edge uses h-functions from the previous tree as pseudo-observations.

#### Scenario: Train Tree 1
- **WHEN** u_train data is provided
- **THEN** each edge (e.g., edge 1: u[,1] vs u[,2]) is fitted independently
- **AND** h1, h2 values are stored for Tree 2 construction

#### Scenario: Cascade h-functions to higher trees
- **WHEN** Tree k is complete
- **THEN** Tree k+1 edges use h1/h2 from Tree k edges as input (u1=results[[k]][[e]]$h1, u2=results[[k]][[e+1]]$h2)
- **AND** tree-level cumulative log-likelihood is printed

### Requirement: Three-model training pipeline (Static, FIGAS, GAS)

The system SHALL train the same D-Vine structure with three parameterization modes: static (constant Copula parameters), FIGAS(1,1,d) (fractionally integrated GAS), and GAS(1,1) (standard score-driven).

#### Scenario: Train all three models
- **WHEN** D-Vine structure and u_train are provided
- **THEN** static_model is trained with BiCopEst for each edge
- **AND** figas_model is trained with FIGAS recursion for each edge
- **AND** gas_model is trained with GAS(1,1) recursion for each edge
- **AND** total log-likelihoods for all three models are printed

### Requirement: Out-of-sample evaluation

The system SHALL evaluate all three trained models on the test set by computing OOS total log-likelihood through sequential D-Vine filtering.

#### Scenario: Compute OOS log-likelihood
- **WHEN** u_test (486x5) and trained models are provided
- **THEN** for each model type, test edges are filtered sequentially through the D-Vine structure
- **AND** total OOS log-likelihood is the sum across all 10 edges
- **AND** OOS results for static, FIGAS, and GAS are printed
- **AND** the best model by OOS log-likelihood is identified

#### Scenario: Static model test evaluation
- **WHEN** model_type is "static"
- **THEN** each edge uses fixed params (par, par2) from training with BiCopPDF and BiCopHfunc on test data

#### Scenario: FIGAS model test evaluation
- **WHEN** model_type is "figas"
- **THEN** each edge runs filter_figas with estimated params on test data
- **AND** h1, h2 outputs are propagated through trees

#### Scenario: GAS model test evaluation
- **WHEN** model_type is "gas"
- **THEN** each edge runs filter_gas with estimated params on test data
- **AND** h1, h2 outputs are propagated through trees