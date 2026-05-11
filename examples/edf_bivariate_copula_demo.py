# %%
# ruff: noqa: E402
"""Quick bivariate-copula simulation for the new EDF-based IC.

Generates data from a bivariate Gaussian copula whose Fisher-z-transformed
correlation is a sparse linear function of X, fits
``MultivariateOnlineDistributionalRegressionPath`` with a LASSO path
(so ``compute_edf`` kicks in), and prints the IC / EDF diagnostics
stored on ``_model_selection`` so you can eyeball the new behavior.

Run from the repo root so the local ``src/ondil`` is importable, e.g.::

    python examples/edf_bivariate_copula_demo.py
"""

import os
import sys

# Make the in-tree package importable when running from the repo root
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import numpy as np
import scipy.stats as st

from ondil.distributions import BivariateCopulaNormal
from ondil.estimators import MultivariateOnlineDistributionalRegressionPath
from ondil.links import FisherZLink, ParameterToKendallsTau
from ondil.methods import LassoPath

np.set_printoptions(precision=4, suppress=True)
rng = np.random.default_rng(42)


# -------------------------------------------------------------------------
# 1. Simulate bivariate Gaussian copula data
# -------------------------------------------------------------------------
# True data-generating process:
#   eta_i = 0.5 + 0.8 * x1_i - 0.6 * x3_i     (sparse: x2, x4, x5 = 0)
#   rho_i = tanh(eta_i / 2)                   (inverse Fisher-z scaled)
#   (z1_i, z2_i) ~ N(0, [[1, rho_i], [rho_i, 1]])
#   u_i = (Phi(z1_i), Phi(z2_i))
N = 2000
P = 5
X = rng.standard_normal((N, P))
beta_true = np.array([0.8, 0.0, -0.6, 0.0, 0.0])  # intercept handled by model
intercept_true = 0.5
eta = intercept_true + X @ beta_true
rho = np.tanh(eta / 2.0)  # in (-1, 1)

# Draw correlated latents per row, then push through the normal CDF
z1 = rng.standard_normal(N)
z2 = rho * z1 + np.sqrt(np.maximum(1.0 - rho**2, 1e-12)) * rng.standard_normal(N)
u = np.column_stack([st.norm.cdf(z1), st.norm.cdf(z2)])

print(f"Simulated N={N} bivariate Gaussian copula observations.")
print(f"  Target support: features {np.where(beta_true != 0)[0].tolist()}")
print(f"  rho range: [{rho.min():.3f}, {rho.max():.3f}]\n")


# -------------------------------------------------------------------------
# 2. Fit with a LASSO path (so compute_edf is exercised)
# -------------------------------------------------------------------------
distribution = BivariateCopulaNormal(
    link=FisherZLink(),
    param_link=ParameterToKendallsTau(),
)

# Single dependence parameter (n_params=1, 1 element) - the exact scope
# of the new EDF branch in _fit_model_selection / _update_model_selection.
equation = {0: {0: np.arange(P)}}

estimator = MultivariateOnlineDistributionalRegressionPath(
    distribution=distribution,
    equation=equation,
    method=LassoPath(lambda_n=100, lambda_eps=1e-3),
    ic="bic",
    fit_intercept=True,
    regularize_intercept=False,
    scale_inputs=False,
    verbose=0,
    max_iterations_inner=5,
    max_iterations_outer=10,
    early_stopping=False,
)

estimator.fit(X, u)


# -------------------------------------------------------------------------
# 3. Inspect what the new EDF branch produced
# -------------------------------------------------------------------------
# _model_selection[param][adr_step][element] = {"ll":..., "non_zero":..., "ic":...}
# With the new code, "non_zero" for the CopulaMixin-scalar case now holds
# the EDF vector (one value per lambda) instead of the raw |A| count.
ms = estimator._model_selection[0][0][0]
edf_path = np.asarray(ms["non_zero"])
ic_path = np.asarray(ms["ic"])
ll_path = np.asarray(ms["ll"])
opt_idx = int(np.argmin(ic_path))

print("Path diagnostics (param=0, adr=0, element=0):")
print(f"  lambda_n           = {len(edf_path)}")
print(f"  EDF range          = [{edf_path.min():.3f}, {edf_path.max():.3f}]")
print(f"  EDF at opt lambda  = {edf_path[opt_idx]:.3f}")
print(f"  BIC at opt lambda  = {ic_path[opt_idx]:.3f}")
print(f"  opt path index     = {opt_idx}\n")

# Head of the path to eyeball monotonicity
print("First 10 path points (lambda decreasing left->right):")
print(f"  edf = {edf_path[:10].round(3)}")
print(f"  ic  = {ic_path[:10].round(3)}")
print()

# Selected coefficients
beta_hat = estimator.coef_[0][0][0]  # param=0, element=0, adr=0, row=0 (only)
print("Selected coefficients (index 0 = intercept):")
for j, bj in enumerate(beta_hat):
    tag = "  <- selected" if bj != 0 else ""
    truth = (
        intercept_true if j == 0 else (beta_true[j - 1] if j - 1 < P else 0.0)
    )
    print(f"  beta[{j}] = {bj: .4f}   (truth {truth: .2f}){tag}")
print()


# -------------------------------------------------------------------------
# 4. Sanity check: as alpha -> 1 (pure LASSO), EDF == |active set|.
#    Verified via compute_edf directly.
# -------------------------------------------------------------------------
method = estimator._method[0][0]
beta_path = estimator.coef_path_[0][0][0]
edf_vec = method.compute_edf(
    x_gram=estimator._x_gram[0][0][0],
    beta_path=beta_path,
    is_regularized=estimator.is_regularized_[0][0],
)
nz_vec = np.sum(beta_path != 0, axis=1)
match = np.allclose(edf_vec, nz_vec.astype(float), atol=1e-8)
print(f"Pure-LASSO identity (EDF == |A|): {match}")
if not match:
    diffs = edf_vec - nz_vec.astype(float)
    print(f"  max abs difference: {np.max(np.abs(diffs)):.2e}")
