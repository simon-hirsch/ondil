# Estimation Methods

## Overview

`EstimationMethod()` classes do the actual hard lifting of fitting coefficients (or weights). They take more technical parameters like the length of the regularization path or upper bounds on certain coefficients. These parameters depend on the individual estimation method. In general, we aim to provide sensible out-of-the-box defaults. This [page](estimators_and_methods.md) explains the difference in detail. `Estimator` classes often take a method parameter, to which either a string or an instance of the `EstimationMethod()` can be passed, e.g.

```python
from ondil.estimators import OnlineLinearModel
from ondil.methods import LassoPath

fit_intercept = True
scale_inputs = True

model = OnlineLinearModel(
    method="lasso",  # default parameters
    fit_intercept=fit_intercept,
    scale_inputs=scale_inputs,
)
# or equivalent
model = OnlineLinearModel(
    method=LassoPath(),  # default parameters
    fit_intercept=fit_intercept,
    scale_inputs=scale_inputs,
)
# or with user-defined parameters
model = OnlineLinearModel(
    method=LassoPath(
        lambda_n=10
    ),  # only 10 different regularization strengths
    fit_intercept=fit_intercept,
    scale_inputs=scale_inputs,
)
```

More information on coordinate descent can also be found on this [page](coordinate_descent.md) and in the API Reference below.

## API Reference

!!! note
    We don't document the classmethods of the `EstimationMethod` since these are only used internally.


::: ondil.methods.OrdinaryLeastSquares

::: ondil.methods.LassoPath

::: ondil.methods.Ridge

::: ondil.methods.ElasticNetPath
