# Sktime Adapter for OnlineLinearModel

This document describes the sktime adapter for the OnlineLinearModel estimator in the ondil package.

## Overview

The `OnlineLinearModelSktime` class is an adapter that makes the ondil `OnlineLinearModel` compatible with the sktime framework. This allows you to use ondil's online linear regression capabilities within sktime's time series analysis ecosystem.

## Features

- **Online Learning**: Supports incremental learning through the `update()` method
- **Multiple Data Formats**: Handles both 2D (standard) and 3D (multivariate time series) data
- **Flexible Estimation Methods**: Supports OLS, Lasso, and Elastic Net
- **Information Criteria**: Automatic model selection using AIC, BIC, or HQC
- **sktime Compatibility**: Full integration with sktime's regression interface

## Installation

First, ensure you have both ondil and sktime installed:

```bash
pip install ondil sktime
```

Note: ondil requires numba, which will be installed automatically.

## Basic Usage

### Standard 2D Data

```python
import numpy as np
from ondil.estimators import OnlineLinearModelSktime

# Create sample data
X = np.random.random((100, 5))  # 100 samples, 5 features
y = np.random.random(100)       # Target values

# Create and fit the adapter
adapter = OnlineLinearModelSktime(
    method="ols",
    fit_intercept=True,
    scale_inputs=True
)

# Fit the model
adapter.fit(X, y)

# Make predictions
predictions = adapter.predict(X)

# Update with new data (online learning)
X_new = np.random.random((20, 5))
y_new = np.random.random(20)
adapter.update(X_new, y_new)
```

### 3D Time Series Data

```python
import numpy as np
from ondil.estimators import OnlineLinearModelSktime

# Create 3D time series data
# Shape: (n_instances, n_timepoints, n_features)
X_3d = np.random.random((50, 10, 3))  # 50 series, 10 timepoints, 3 features
y_3d = np.random.random(50)           # Target for each series

# The adapter automatically flattens 3D data to 2D
adapter = OnlineLinearModelSktime(method="lasso")
adapter.fit(X_3d, y_3d)

# Predictions work the same way
predictions = adapter.predict(X_3d)
```

## Parameters

- `forget` (float, default=0.0): Exponential discounting of old observations
- `scale_inputs` (bool, default=True): Whether to scale input features
- `fit_intercept` (bool, default=True): Whether to fit an intercept term
- `regularize_intercept` (bool, default=False): Whether to regularize the intercept
- `method` (str, default="ols"): Estimation method ("ols", "lasso", "elasticnet")
- `ic` (str, default="bic"): Information criterion for model selection ("aic", "bic", "hqc", "max")

## Methods

### Core Methods
- `fit(X, y)`: Fit the model to training data
- `predict(X)`: Make predictions on new data
- `update(X, y, sample_weight=None)`: Update model with new data (online learning)
- `score(X, y)`: Calculate R² score

### Properties
- `coef_`: Model coefficients
- `coef_path_`: Full regularization path coefficients (for path-based methods)

### sktime Integration
- `get_fitted_params()`: Get fitted parameters as a dictionary
- `get_test_params()`: Get test parameters for sktime compatibility

## Online Learning Example

```python
import numpy as np
from ondil.estimators import OnlineLinearModelSktime

# Initial training data
X_initial = np.random.random((50, 3))
y_initial = np.random.random(50)

# Create and fit adapter
adapter = OnlineLinearModelSktime(
    method="lasso",
    forget=0.1,  # Exponential forgetting
    ic="aic"
)

adapter.fit(X_initial, y_initial)
print(f"Initial coefficients: {adapter.coef_}")

# Simulate streaming data
for batch in range(5):
    X_batch = np.random.random((10, 3))
    y_batch = np.random.random(10)
    
    # Update model with new batch
    adapter.update(X_batch, y_batch)
    print(f"Batch {batch + 1} coefficients: {adapter.coef_}")
```

## Integration with sktime

The adapter is fully compatible with sktime's regression interface:

```python
from sktime.utils.estimator_checks import check_estimator
from ondil.estimators import OnlineLinearModelSktime

# Run sktime's estimator checks
estimator = OnlineLinearModelSktime()
# check_estimator(estimator)  # Uncomment to run full checks
```

## Data Format Handling

The adapter automatically handles data format conversion:

- **2D Data**: Used directly with the underlying OnlineLinearModel
- **3D Data**: Flattened from (n_instances, n_timepoints, n_features) to (n_instances, n_timepoints * n_features)

This allows seamless integration with both standard machine learning workflows (2D) and time series analysis (3D).

## Advanced Usage

### Custom Estimation Methods

```python
# Using different estimation methods
adapters = {
    'ols': OnlineLinearModelSktime(method="ols"),
    'lasso': OnlineLinearModelSktime(method="lasso"),
    'elasticnet': OnlineLinearModelSktime(method="elasticnet"),
}

# Fit all methods
for name, adapter in adapters.items():
    adapter.fit(X, y)
    print(f"{name} R² score: {adapter.score(X, y):.3f}")
```

### Information Criteria Comparison

```python
# Compare different information criteria
for ic in ["aic", "bic", "hqc"]:
    adapter = OnlineLinearModelSktime(method="lasso", ic=ic)
    adapter.fit(X, y)
    print(f"{ic.upper()} selected coefficients: {np.sum(np.abs(adapter.coef_) > 1e-6)}")
```

## Limitations

1. **Memory Scaling**: The adapter flattens 3D data, which may use significant memory for large time series
2. **Time Dependencies**: The current implementation doesn't preserve temporal structure within time series
3. **Batch Updates**: The `update()` method processes data in batches rather than single observations

## Future Enhancements

Potential improvements for future versions:
- Native support for temporal dependencies
- More efficient 3D data handling
- Integration with sktime's forecasting interface
- Support for multivariate output regression

## Testing

The adapter includes comprehensive tests to ensure sktime compatibility:

```python
# Run basic tests
python -m pytest tests/test_sktime_adapter.py

# Run with sktime's test suite (if available)
from sktime.tests.test_estimator import run_tests_for_class
from ondil.estimators import OnlineLinearModelSktime
# run_tests_for_class(OnlineLinearModelSktime)
```