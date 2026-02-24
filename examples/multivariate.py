# This script reproduces the multivariate example from the paper:
# Muschinski, Thomas, et al. "Cholesky-based multivariate Gaussian regression." Econometrics and Statistics 29 (2024): 261-281.
# It fits a multivariate distributional regression model to simulated data and visualizes the results.
# The results correspond to Figure 1 in the paper. However, we visualize the the results for the covariance matrix instead of the Cholesky factors.
# Qualitatively, the results are very similar.

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
from sklearn.preprocessing import SplineTransformer

from ondil.distributions import (
    MultivariateNormalInverseCholesky,
    MultivariateNormalInverseModifiedCholesky,
)
from ondil.estimators import MultivariateOnlineDistributionalRegressionPath

np.set_printoptions(precision=3, suppress=True)

# Dimension of the data D=3
# Initial training size M=1000
# On my laptop
# For 10k samples and D=10, it takes a two-three mins.
# For 10k samples and D=3, it takes a few seconds.

D = 3
M = 1000


# Define a function to compute the true parameters
def compute_true(x, D=3):
    M = len(x)
    DD = (D // 3 + 1) * 3

    true_mu = np.zeros((M, DD))
    true_D = np.zeros((M, DD, DD))
    true_T = np.zeros((M, DD, DD))

    for i in range(0, D + 1, 3):
        true_mu[:, i + 0] = 1
        true_mu[:, i + 1] = 1 + x
        true_mu[:, i + 2] = 1 + x**2

        true_D[:, i + 0, i + 0] = np.exp(-2)
        true_D[:, i + 1, i + 1] = np.exp(-2 + x)
        true_D[:, i + 2, i + 2] = np.exp(-2 + x**2)

        true_T[:, i + 0, i + 1] = (1 + x**2) / 4
        true_T[:, i + 0, i + 2] = 0
        true_T[:, i + 1, i + 2] = (3 + x) / 4

    true_T[:, range(DD), range(DD)] = -1
    true_T = true_T[:, :D, :D]

    true_D = true_D[:, :D, :D]
    true_mu = true_mu[:, :D]

    return true_mu, true_D, true_T


distribution = MultivariateNormalInverseCholesky()
equation = {
    0: {d: "all" for d in range(D)},
    1: {k: "all" for k in range((D * (D + 1)) // 2)},
}

r = 0
x = st.uniform(-1, 2).rvs((M), random_state=1 + r)
true_mu, true_D, true_T = compute_true(x, D=D)
true_cov = np.linalg.inv(true_T @ np.linalg.inv(true_D) @ true_T.transpose(0, 2, 1))

corr = true_cov / (
    true_cov.diagonal(axis1=1, axis2=2)[..., None] ** 0.5
    @ true_cov.diagonal(axis1=1, axis2=2)[:, None, :] ** 0.5
)

y = np.zeros((M, D))
for m in range(M):
    y[m, :] = st.multivariate_normal(true_mu[m], true_cov[m]).rvs(1)

transformer = SplineTransformer(
    n_knots=4, degree=3, include_bias=False, extrapolation="linear"
)
transformer.fit(np.expand_dims(x, -1))
X = transformer.transform(np.expand_dims(x, -1))

# Model
estimator_chol = MultivariateOnlineDistributionalRegressionPath(
    distribution=distribution,
    equation=equation,
    method={i: "ols" for i in range(D)},
    early_stopping=False,
    early_stopping_criteria="bic",
    iteration_along_diagonal=False,
    max_regularisation_size=None,
    verbose=2,
)

# Fit the estimator
estimator_chol.fit(X, y)


# Reproduce Figure
grid_x = np.linspace(-1, 1, 100)
grid_X = transformer.transform(np.expand_dims(grid_x, -1))
grid_mu, grid_D, grid_T = compute_true(grid_x, D=D)
grid_cov = np.linalg.inv(grid_T @ np.linalg.inv(grid_D) @ grid_T.transpose(0, 2, 1))

grid_pred = estimator_chol.predict_distribution_parameters(grid_X)
grid_pred_mu = grid_pred[0]
grid_pred_cov = np.linalg.inv(grid_pred[1] @ grid_pred[1].transpose(0, 2, 1))
grid_pred = estimator_chol.predict_distribution_parameters(grid_X)
grid_pred_mu = grid_pred[0]
grid_pred_cov = np.linalg.inv(grid_pred[1] @ grid_pred[1].transpose(0, 2, 1))
diag_idx = np.diag_indices(D)
triu_idx = np.triu_indices(D, k=1)

# Plot for grid_pred_mu vs grid_mu
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].plot(grid_x, grid_pred_mu)
axs[0].plot(grid_x, grid_mu, color="black", ls=":")
axs[0].set_title("Predicted vs True Mean")
axs[0].legend(["Predicted Mean", "True Mean"])

# Plot for diagonal elements of grid_pred_cov vs grid_cov
for i in range(D):
    axs[1].plot(grid_x, grid_pred_cov[:, diag_idx[0][i], diag_idx[1][i]])
    axs[1].plot(
        grid_x, grid_cov[:, diag_idx[0][i], diag_idx[1][i]], color="black", ls=":"
    )
axs[1].set_title("Predicted vs True Covariance (Diagonal Elements)")
axs[1].legend(["Predicted Covariance", "True Covariance"])

# Plot for upper triangular elements of grid_pred_cov vs grid_cov
for i in range(len(triu_idx[0])):
    axs[2].plot(grid_x, grid_pred_cov[:, triu_idx[0][i], triu_idx[1][i]])
    axs[2].plot(
        grid_x, grid_cov[:, triu_idx[0][i], triu_idx[1][i]], color="black", ls=":"
    )
axs[2].set_title("Predicted vs True Covariance (Upper Triangular Elements)")
axs[2].legend(["Predicted Covariance", "True Covariance"])

plt.tight_layout()
# plt.savefig(f"./docs/assets/figure_multivariate_cd_{D}_{M}.png", dpi=300)
# plt.show(block=False)

# Now fit the same model but using the modified Cholesky parameterization
distribution_mcd = MultivariateNormalInverseModifiedCholesky()
equation_mcd = {
    0: {d: "all" for d in range(D)},
    1: {d: "all" for d in range(D)},
    2: {k: "all" for k in range((D * (D - 1)) // 2)},
}

estimator_mcd = MultivariateOnlineDistributionalRegressionPath(
    distribution=distribution_mcd,
    equation=equation_mcd,
    method={i: "ols" for i in range(D)},
    early_stopping=False,
    early_stopping_criteria="bic",
    iteration_along_diagonal=False,
    max_iterations_inner=10,
    max_iterations_outer=10,
    max_regularisation_size=3,
    verbose=3,
    debug=False,
)
# Fit the estimator
estimator_mcd.fit(X, y)

# Updates work the same as with univariate
estimator_mcd.update(X[[-1], :], y[[-1], :])

# Create the same figure for the modified Cholesky based model
grid_x = np.linspace(-1, 1, 100)
grid_X = transformer.transform(np.expand_dims(grid_x, -1))
grid_mu, grid_D, grid_T = compute_true(grid_x, D=D)
grid_cov = np.linalg.inv(grid_T @ np.linalg.inv(grid_D) @ grid_T.transpose(0, 2, 1))

grid_pred = estimator_mcd.predict_distribution_parameters(grid_X)
grid_pred_scipy = estimator_mcd.distribution.theta_to_scipy(grid_pred)
grid_pred_mu = grid_pred_scipy["mean"]
grid_pred_cov = grid_pred_scipy["cov"]

diag_idx = np.diag_indices(D)
triu_idx = np.triu_indices(D, k=1)

fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Plot for grid_pred_mu vs grid_mu
axs[0].plot(grid_x, grid_pred_mu)
axs[0].plot(grid_x, grid_mu, color="black", ls=":")
axs[0].set_title("Predicted vs True Mean")
axs[0].legend(["Predicted Mean", "True Mean"])

# Plot for diagonal elements of grid_pred_cov vs grid_cov
for i in range(D):
    axs[1].plot(grid_x, grid_pred_cov[:, diag_idx[0][i], diag_idx[1][i]])
    axs[1].plot(
        grid_x, grid_cov[:, diag_idx[0][i], diag_idx[1][i]], color="black", ls=":"
    )
axs[1].set_title("Predicted vs True Covariance (Diagonal Elements)")
axs[1].legend(["Predicted Covariance", "True Covariance"])

# Plot for upper triangular elements of grid_pred_cov vs grid_cov
for i in range(len(triu_idx[0])):
    axs[2].plot(grid_x, grid_pred_cov[:, triu_idx[0][i], triu_idx[1][i]])
    axs[2].plot(
        grid_x, grid_cov[:, triu_idx[0][i], triu_idx[1][i]], color="black", ls=":"
    )
axs[2].set_title("Predicted vs True Covariance (Upper Triangular Elements)")
axs[2].legend(["Predicted Covariance", "True Covariance"])

plt.tight_layout()
# plt.savefig(f"./docs/assets/figure_multivariate_mcd_{D}_{M}.png", dpi=300)
# plt.show(block=False)
