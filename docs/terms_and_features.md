# Terms and Features: The Building Blocks of Distributional Regression

## Overview

The `ondil` package implements a hierarchical architecture for building distributional regression models. At the highest level, you work with **Estimators** that manage the overall modeling process. Estimators contain **Equations** that define how each distribution parameter is modeled. Equations are composed of **Terms** that represent different types of relationships, and Terms use **Features** to transform input data into model predictors.

This page explains the logical hierarchy: **Estimator → Equation → Terms → Features** and provides API references for each component.

## The Hierarchy in Detail

### Estimators

Estimators are the top-level objects that users interact with. They provide the main interface for fitting, updating, and predicting with distributional regression models.

**Key Estimator Classes:**
- `OnlineDistributionalRegression`: Main estimator for distributional regression
- `OnlineLinearModel`: Simplified estimator for single-parameter models
- `OnlineLasso`, `OnlineRidge`: Specialized estimators with built-in regularization

**Key Methods:**
- `fit(X, y)`: Initial model fitting
- `update(X, y)`: Incremental updates with new data
- `predict_distribution_parameters(X)`: Predict all distribution parameters
- `predict(X)`: Predict expected values (location parameter)

### Equations

Equations define how each parameter of the target distribution is modeled. In distributional regression, we model multiple parameters simultaneously (e.g., location, scale, shape for a Student-t distribution).

An equation is specified as a dictionary where keys are parameter indices and values define which features to include:

```python
equation = {
    0: "all",      # Location parameter: use all features
    1: [0, 2, 4], # Scale parameter: use features 0, 2, and 4
    2: "intercept" # Shape parameter: intercept only
}
```

### Terms

Terms are the fundamental building blocks that define different types of relationships between inputs and outputs. Each term contributes to the linear predictor for a distribution parameter.

**Types of Terms:**

#### Linear Terms
- `LinearTerm`: Standard linear regression terms
- `RegularizedLinearTerm`: Linear terms with LASSO/ridge regularization
- `InterceptTerm`: Simple intercept-only terms

#### Time Series Terms
- `TimeSeriesTerm`: Autoregressive terms using lagged features
- `RegularizedTimeSeriesTerm`: Time series terms with regularization

#### Special Terms
- `ScikitLearnEstimatorTerm`: Wrapper for any scikit-learn compatible estimator

**Term Interface:**
All terms implement:
- `fit()`: Initial fitting
- `update()`: Incremental updates
- `predict_in_sample_during_fit/update()`: In-sample predictions
- `predict_out_of_sample()`: Future predictions

### Features

Features are transformations applied to input data to create predictors. They define how raw data is converted into model inputs.

**Types of Features:**

#### Linear Features
- `LinearFeature`: Identity transformation (no change)
- `InterceptFeature`: Constant intercept term

#### Time Series Features
- `LaggedTheta`: Lagged fitted values of the current parameter
- `LaggedTarget`: Lagged target values
- `LaggedSquaredResidual`: Lagged squared residuals
- `LaggedAbsoluteResidual`: Lagged absolute residuals
- `LaggedResidual`: Lagged raw residuals

**Feature Interface:**
Features implement:
- `make_design_matrix_*()`: Create design matrices for different prediction contexts
- `lags`: Property defining required lag structure

## API Reference

### Core Classes

::: ondil.estimators.OnlineDistributionalRegression
    options:
      heading_level: 3
      show_signature: true
      show_signature_annotations: true

::: ondil.terms.LinearTerm
    options:
      heading_level: 3
      show_signature: true
      show_signature_annotations: true

::: ondil.terms.TimeSeriesTerm
    options:
      heading_level: 3
      show_signature: true
      show_signature_annotations: true

::: ondil.terms.features.LinearFeature
    options:
      heading_level: 3
      show_signature: true
      show_signature_annotations: true

::: ondil.terms.features.LaggedTheta
    options:
      heading_level: 3
      show_signature: true
      show_signature_annotations: true
