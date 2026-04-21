# Online Scaling and Preprocessing

## Feature Scaling: StandardScaler vs. RobustScaler (Mean Absolute Deviation)

This module provides two normalization strategies with different assumptions
about the data distribution and sensitivity to outliers.

### `StandardScaler`

Transforms each feature as: `z = (x - mean) / std` and hence

- Centers data around `0` using the arithmetic mean.
- Scales to unit variance using standard deviation.
- Works best when features are approximately Gaussian and outliers are limited.
- Sensitive to extreme values, since both `mean` and `std` are strongly affected by them.

### Robust scaling via `MeanAbsoluteDeviation`

Transforms each feature as: `z_robust = (x - mean) / MAD` where:
- `center` is typically a robust location estimate
- `MAD = mean(|x - center|)` is the mean absolute deviation from the mean.
- Reduces the influence of outliers compared to standard deviation-based scaling.
- Better for heavy-tailed, skewed, or contaminated datasets.
- May be less efficient than `StandardScaler` on clean, truly normal data.

### Choosing between them

- Use **`StandardScaler`** for well-behaved, near-normal features and models that assume standardized Gaussian-like inputs.
- Use **robust MAD-based scaling** when outliers are common or feature distributions are non-Gaussian and you want more stable scaling factors.

## API Reference

::: ondil.scaler.OnlineScaler

::: ondil.scaler.OnlineMeanAbsoluteDeviationScaler