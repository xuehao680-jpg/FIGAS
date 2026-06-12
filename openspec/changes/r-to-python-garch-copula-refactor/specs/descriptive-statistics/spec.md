## ADDED Requirements

### Requirement: Data loading and train-test split

The system SHALL load the CSV data file, extract 5 return columns (hushen300, ydlMIB, xxlNZ50, nfFTSE, bxBOVESPA) and date column, and split chronologically into 80% training and 20% testing with fixed random seed.

#### Scenario: Load and split data
- **WHEN** the CSV file at data/yidaiyilu(1).csv exists
- **THEN** 5 return columns are extracted as a DataFrame
- **AND** rows with any NA are dropped
- **AND** date column is parsed as Date
- **AND** first 80% (1944 obs) assigned to training, last 20% (486 obs) to testing
- **AND** random seed is set to 123

### Requirement: Descriptive statistics table

The system SHALL compute and display mean, standard deviation, skewness, and kurtosis for each of the 5 return series.

#### Scenario: Compute descriptive stats
- **WHEN** full return series are provided
- **THEN** a DataFrame with columns [Variable, Mean, Std, Skew, Kurt] is printed
- **AND** statistics are computed using scipy.stats (skew, kurtosis)

### Requirement: Time series plots

The system SHALL generate individual return series plots for each asset with date on x-axis and return on y-axis, using yearly tick marks.

#### Scenario: Plot return series
- **WHEN** full return series and date column are provided
- **THEN** 5 line plots are generated (one per asset)
- **AND** x-axis shows yearly ticks (2015, 2016, ..., 2020)
- **AND** each plot is titled "<asset_name> 日收益率序列图"

### Requirement: Statistical tests (JB, ADF, ARCH-LM)

The system SHALL perform Jarque-Bera normality test, ADF stationarity test, and ARCH-LM heteroskedasticity test for each asset on the training set.

#### Scenario: Run diagnostic tests
- **WHEN** training set returns are provided
- **THEN** JB test statistic and p-value are printed per asset
- **AND** ADF test statistic and p-value are printed per asset
- **AND** for ARCH-LM: ARMA residuals are extracted first via auto.arima, then ArchTest(lags=5) is applied
- **AND** ARCH-LM p-values are annotated with significance stars (***/**/* for 0.01/0.05/0.1)

### Requirement: Ljung-Box white noise test

The system SHALL compute Ljung-Box test p-values at lags 1, 2, 3, 4 for all 5 return series and display them as a matrix.

#### Scenario: Compute Ljung-Box p-values
- **WHEN** full return series are provided
- **THEN** a 4x5 p-value matrix is printed
- **AND** row names are lag_1 through lag_4

### Requirement: ARMA order selection

The system SHALL automatically select optimal ARMA(p,q) orders for each asset using AICc criterion with p ≤ 3 and q ≤ 5.

#### Scenario: Select ARMA orders
- **WHEN** training set returns are provided
- **THEN** auto.arima (or equivalent) is called with max.p=3, max.q=5, ic="aicc"
- **AND** selected (p, q) order is printed per asset

### Requirement: GARCH order selection

The system SHALL select optimal GARCH order from candidates {(1,1), (1,2), (2,1)} by comparing AIC under sGARCH-normal specification.

#### Scenario: Select GARCH orders
- **WHEN** ARMA orders are determined
- **THEN** three GARCH specifications are fitted and compared
- **AND** the order with lowest AIC is printed per asset
