"""Example: Distributional regression pipeline with spline features.

This example uses the diabetes dataset from scikit-learn, creates
nonlinear spline features, and fits an online distributional regression
model inside `DistributionalRegressionPipeline`.
"""

# %%

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import SplineTransformer

from ondil.distributions import Normal
from ondil.estimators import OnlineDistributionalRegression
from ondil.links import InverseSoftPlusShiftValue
from ondil.pipeline import DistributionalRegressionPipeline

np.set_printoptions(precision=3, suppress=True)

# %%
# Use a single feature so the spline basis is easy to interpret.
X_df, y = load_diabetes(return_X_y=True, as_frame=True)
X = X_df[["bmi"]].to_numpy()
y = y.to_numpy()

# Build a simple train/update split to demonstrate incremental learning.
n_train = 375
n_update = 25
X_train, y_train = X[:n_train], y[:n_train]
X_update, y_update = X[n_train : n_train + n_update], y[n_train : n_train + n_update]
X_test, y_test = X[n_train + n_update :], y[n_train + n_update :]

# %%
# Model location (mu) and scale (sigma) with spline features
equation = {
    0: "all",
    1: "all",
}
distribution = Normal(
    scale_link=InverseSoftPlusShiftValue(value=0.1)  # Ensure sigma > 0.1 for stability
)

pipe = DistributionalRegressionPipeline(
    steps=[
        (
            "spline",
            SplineTransformer(
                n_knots=6,
                degree=2,
                include_bias=False,
                extrapolation="linear",
            ),
        ),
        (
            "model",
            OnlineDistributionalRegression(
                distribution=distribution,
                equation=equation,
                method="ols",
                fit_intercept=True,
                scale_inputs=False,
            ),
        ),
    ]
)

# %%
# Fit on initial data.
pipe.fit(X_train, y_train)

# Predict distribution parameters and a few quantiles on holdout data.
theta_before = pipe.predict_distribution_parameters(X_test)
q_before = pipe.predict_quantile(X_test, quantile=np.array([0.1, 0.5, 0.9]))

print("Before update")
print("theta shape:", theta_before.shape)  # (n_samples, 2) for Normal(mu, sigma)
print("first 3 mu predictions:", theta_before[:3, 0])
print("first 3 median predictions:", q_before[:3, 1])

# %%
# Incremental update with new observations.
pipe.update(X_update, y_update)

theta_after = pipe.predict_distribution_parameters(X_test)
q_after = pipe.predict_quantile(X_test, quantile=np.array([0.1, 0.5, 0.9]))

print("\nAfter update")
print("theta shape:", theta_after.shape)
print("first 3 mu predictions:", theta_after[:3, 0])
print("first 3 median predictions:", q_after[:3, 1])

# Simple sanity check to confirm update changed the model.
mean_abs_change = np.mean(np.abs(theta_after[:, 0] - theta_before[:, 0]))
print("\nMean absolute change in predicted mu after update:", round(mean_abs_change, 4))
print(
    "Holdout size:",
    X_test.shape[0],
    "| y holdout mean:",
    round(float(y_test.mean()), 3),
)

# %%
