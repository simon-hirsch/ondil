# %%
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes

from ondil.diagnostics import PITHistogramDisplay, QQDisplay, WormPlotDisplay
from ondil.distributions import Normal
from ondil.estimators import OnlineDistributionalRegression

# %%
## Diabetes data set
X, y = load_diabetes(return_X_y=True)

## Simple linear regression with OLS
# We use all variables to model both mu and keep sigma as constant (intercept)
# Can also use: "intercept" or pass a numpy array with indices / boolean
equation = {
    0: "all",
    1: "intercept",
}

model = OnlineDistributionalRegression(
    distribution=Normal(),
    method="ols",
    equation=equation,
    fit_intercept=True,
)

model.fit(X, y)

# %%
# Now we can use the diagnostic tools
# Each of them can be called from the estimator
# and will return the display object

WormPlotDisplay.from_estimator(
    estimator=model,
    X=X,
    y=y,
)

PITHistogramDisplay.from_estimator(
    estimator=model,
    X=X,
    y=y,
)

QQDisplay.from_estimator(
    estimator=model,
    X=X,
    y=y,
)

# %%
# We can also plot them together
# in a single figure by passing the axes to the from_estimator method
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
QQDisplay.from_estimator(
    estimator=model,
    X=X,
    y=y,
    ax=axes[0],
)
WormPlotDisplay.from_estimator(
    estimator=model,
    X=X,
    y=y,
    ax=axes[1],
)
PITHistogramDisplay.from_estimator(
    estimator=model,
    X=X,
    y=y,
    ax=axes[2],
)
plt.tight_layout()
plt.show(block=False)
