# Author: Christian Jobelius Schulz
# License: GPL-3.0
# SPDX-License-Identifier: GPL-3.0-only
#
# Portions of this file -- the log-likelihood, score and Hessian expressions
# for the Gumbel copula, together with the h-function, its inverse and
# `qcondgum` -- were translated into Python from the C sources of
# the R package VineCopula:
#
#     https://cran.r-project.org/package=VineCopula
#     Thomas Nagler, Ulf Schepsmeier, Jakob Stoeber, Eike Christian Brechmann,
#     Benedikt Graeler, Tobias Erhardt and others
#     SPDX-License-Identifier: GPL-2.0-or-later OR GPL-3.0-or-later
#
# Redistributing those derivations under GPL-3.0-only is consistent with the
# upstream dual GPL-2 | GPL-3 option. See THIRD_PARTY_NOTICES.md of the
# replication package.

from typing import Dict

import numpy as np
import scipy.stats as st

from ..base import BivariateCopulaMixin, CopulaMixin, Distribution, LinkFunction
from ..links import GumbelParameterToKendallsTau, Log
from ..robust_math import UMAX, UMIN
from ..types import ParameterShapes


class BivariateCopulaGumbel(CopulaMixin, Distribution, BivariateCopulaMixin):
    corresponding_gamlss: str = None
    parameter_names = {0: "theta"}
    parameter_support = {0: (1, np.inf)}
    distribution_support = (0, 1)
    n_params = len(parameter_names)
    parameter_shape = {0: ParameterShapes.SCALAR}

    def __init__(
        self,
        link: LinkFunction = Log(),
        param_link: LinkFunction = GumbelParameterToKendallsTau(),
        family_code: int = 41,
    ):
        super().__init__(
            links={0: link},
            param_links={0: param_link},
        )
        self.family_code = family_code  # gamCopula family code (401, 402, 403, 404)
        self.is_multivariate = True
        self._regularization_allowed = {0: False}

    @staticmethod
    def fitted_elements(dim: int):
        return {0: 1}

    def theta_to_params(self, theta) -> np.ndarray:
        val = theta[0]
        return np.asarray(val).reshape(-1, 1)

    def set_initial_guess(self, y, theta, param):
        return theta

    def dl1_dp1(self, y: np.ndarray, theta: Dict, param: int = 0):
        """Return the first derivatives wrt to the parameter.

        Args:
            y (np.ndarray): Y values of shape n x d
            theta (Dict): Dict with {0 : fitted theta}
            param (int, optional): Which parameter derivatives to return. Defaults to 0.

        Returns:
            derivative: The 1st derivatives.
        """
        theta_param = self.theta_to_params(theta)
        return _derivative_1st(y, theta_param, self.family_code)

    def dl2_dp2(self, y: np.ndarray, theta: Dict, param: int = 0, clip=False):
        """Return the second derivatives wrt to the parameter.

        Args:
            y (np.ndarray): Y values of shape n x d
            theta (Dict): Dict with {0 : fitted theta}
            param (int, optional): Which parameter derivatives to return. Defaults to 0.

        Returns:
            derivative: The 2nd derivatives.
        """
        theta_param = self.theta_to_params(theta)
        return _derivative_2nd(y, theta_param, self.family_code)

    def element_score(self, y: np.ndarray, theta: Dict, param: int = 0, k: int = 0):
        return self.element_dl1_dp1(y=y, theta=theta, param=param, k=k)

    def element_hessian(self, y: np.ndarray, theta: Dict, param: int = 0, k: int = 0):
        return self.element_dl2_dp2(y=y, theta=theta, param=param, k=k)

    def element_dl1_dp1(
        self, y: np.ndarray, theta: Dict, param: int = 0, k: int = 0, clip=False
    ):
        theta_param = self.theta_to_params(theta)

        deriv = _derivative_1st(
            y,
            theta_param,
            self.family_code,
        )
        return deriv

    def element_dl2_dp2(
        self, y: np.ndarray, theta: Dict, param: int = 0, k: int = 0, clip=False
    ):
        theta_param = self.theta_to_params(theta)
        deriv = _derivative_2nd(y, theta_param)
        return deriv

    def dl2_dpp(self, y: np.ndarray, theta: Dict, param: int = 0):
        raise NotImplementedError("Not implemented.")

    def initial_values(self, y, param=0):
        M = y.shape[0]
        # Compute the empirical Kendall's tau and convert to Gumbel parameter
        tau = st.kendalltau(y[:, 0], y[:, 1]).correlation
        return np.full((M, 1), tau)

    def cdf(self, y, theta):
        raise NotImplementedError("Not implemented")

    def ppf(self, q, theta):
        raise NotImplementedError("Not implemented")

    def rvs(self, size, theta, family_code=None):
        """
        Generate random samples from the bivariate normal copula.

        Args:
            size (int): Number of samples to generate.
            theta (dict or np.ndarray): Correlation parameter(s).

        Returns:
            np.ndarray: Samples of shape (size, 2) in (0, 1).
        """
        # Generate standard normal samples

        z1 = np.random.uniform(size=size)
        z2 = np.random.uniform(size=size)

        x = self.hinv(z1, z2, theta, un=2, family_code=family_code)

        return np.column_stack([z2, x])

    def pdf(self, y, theta):
        return np.exp(self.logpdf(y, theta))

    def logcdf(self, y, theta):
        raise NotImplementedError("Not implemented")

    def logpdf(self, y, theta):
        theta_param = self.theta_to_params(theta)
        return _log_likelihood(y, theta_param, self.family_code)

    def logpmf(self, y, theta):
        raise NotImplementedError("Not implemented")

    def pmf(self, y, theta):
        raise NotImplementedError("Not implemented")

    def calculate_conditional_initial_values(
        self, y: np.ndarray, theta: Dict[int, np.ndarray]
    ) -> Dict[int, np.ndarray]:
        raise NotImplementedError("Not implemented")

    def hfunc(
        self, u: np.ndarray, v: np.ndarray, theta: np.ndarray, un: int, family_code: int
    ) -> np.ndarray:
        """
        Conditional distribution function h(u|v) for the bivariate Gumbel copula.
        Translated from the C sources of the R package VineCopula.

        Args:
            u (np.ndarray): Array of shape (n,) with values in (0, 1).
            v (np.ndarray): Array of shape (n,) with values in (0, 1).
            theta (np.ndarray or float): Gumbel parameter(s), shape (n,) or scalar.
            un (int): Determines which conditional to compute (0 for h(u|v), 1 for h(v|u)).

        Returns:
            np.ndarray: Array of shape (n,) with conditional probabilities.
        """

        theta = np.asarray(
            theta
        ).copy()  # <- prevents in-place mutation of caller's array
        u = np.clip(u, self.UMIN, self.UMAX).reshape(-1, 1)
        v = np.clip(v, self.UMIN, self.UMAX).reshape(-1, 1)

        # Get rotations for all samples
        rotation = get_effective_rotation(theta, family_code)
        # Apply rotation transformations vectorized
        u_rot, v_rot = u.copy(), v.copy()

        # 180° rotation (survival)
        mask_1 = rotation == 1
        u_rot[mask_1] = 1 - u[mask_1]
        v_rot[mask_1] = 1 - v[mask_1]

        if un == 1:
            # 90° rotation
            mask_2 = rotation == 2
            u_rot[mask_2] = 1 - u[mask_2]
            theta[mask_2] = -theta[mask_2]

            # 270° rotation
            mask_3 = rotation == 3
            v_rot[mask_3] = 1 - v[mask_3]
            theta[mask_3] = -theta[mask_3]
        else:
            # 90° rotation
            mask_2 = rotation == 2
            v_rot[mask_2] = 1 - v[mask_2]
            theta[mask_2] = -theta[mask_2]

            # 270° rotation
            mask_3 = rotation == 3
            u_rot[mask_3] = 1 - u[mask_3]
            theta[mask_3] = -theta[mask_3]

        log_u = np.log(u_rot)
        log_v = np.log(v_rot)

        t1 = (-log_u) ** theta
        t2 = (-log_v) ** theta
        sum_t = t1 + t2

        copula_val = np.exp(-(sum_t ** (1.0 / theta)))
        h = -(copula_val * (sum_t ** (1.0 / theta - 1.0)) * t2) / (v_rot * log_v)

        if un == 1:
            h[mask_1] = 1 - h[mask_1]  # 180° rotation
            h[mask_2] = 1 - h[mask_2]  # 270° rotation
        else:
            h[mask_1] = 1 - h[mask_1]  # 180° rotation
            h[mask_3] = 1 - h[mask_3]  # 270° rotation

        return h.squeeze()

    def hinv(
        self, u: np.ndarray, v: np.ndarray, theta: np.ndarray, un: int, family_code: int
    ) -> np.ndarray:
        """
        Inverse conditional distribution function h^(-1)(u|v) for the bivariate Gumbel copula.
        Translated from the C sources of the R package VineCopula.

        Args:
            u (np.ndarray): Array of shape (n,) with values in (0, 1).
            v (np.ndarray): Array of shape (n,) with values in (0, 1).
            theta (np.ndarray or float): Gumbel parameter(s), shape (n,) or scalar.
            un (int): Determines which conditional to compute.

        Returns:
            np.ndarray: Array of shape (n,) with inverse conditional probabilities.
        """
        # Apply clipping using masks
        u_mask_low = u < self.UMIN
        u_mask_high = u > self.UMAX
        v_mask_low = v < self.UMIN
        v_mask_high = v > self.UMAX

        u = np.where(u_mask_low, self.UMIN, u)
        u = np.where(u_mask_high, self.UMAX, u)
        v = np.where(v_mask_low, self.UMIN, v)
        v = np.where(v_mask_high, self.UMAX, v)
        u = u.reshape(-1, 1)
        v = v.reshape(-1, 1)

        rotation = get_effective_rotation(theta, family_code)
        # Apply rotation transformations vectorized
        u_rot, v_rot = u.copy(), v.copy()
        hinv = np.zeros_like(u)
        # 180° rotation (survival)
        mask_1 = rotation == 1
        u_rot[mask_1] = 1 - u[mask_1]
        v_rot[mask_1] = 1 - v[mask_1]

        if un == 1:
            # 90° rotation
            mask_2 = rotation == 2
            u_rot[mask_2] = 1 - u[mask_2]
            theta[mask_2] = -theta[mask_2]

            # 270° rotation
            mask_3 = rotation == 3
            v_rot[mask_3] = 1 - v[mask_3]
            theta[mask_3] = -theta[mask_3]
        else:
            # 90° rotation
            mask_2 = rotation == 2
            v_rot[mask_2] = 1 - v[mask_2]
            theta[mask_2] = -theta[mask_2]

            # 270° rotation
            mask_3 = rotation == 3
            u_rot[mask_3] = 1 - u[mask_3]
            theta[mask_3] = -theta[mask_3]

        hinv = qcondgum(u_rot, v_rot, theta).reshape(-1, 1)

        h_mask_low = hinv < 0
        h_mask_high = hinv > 1
        hinv = np.where(h_mask_low, 0, hinv)
        hinv = np.where(h_mask_high, 1, hinv)

        if un == 1:
            hinv[mask_1] = 1 - hinv[mask_1]  # 180° rotation
            hinv[mask_2] = 1 - hinv[mask_2]  # 270° rotation
        else:
            hinv[mask_1] = 1 - hinv[mask_1]  # 180° rotation
            hinv[mask_3] = 1 - hinv[mask_3]  # 270° rotation

        return hinv.squeeze()

    def get_regularization_size(self, dim: int) -> int:
        return dim


def qcondgum(q: np.ndarray, u: np.ndarray, de: np.ndarray) -> np.ndarray:
    """
    Quantile function for conditional Gumbel copula.
    Translated from C implementation.

    Args:
        q (np.ndarray): Quantile values in (0, 1)
        u (np.ndarray): Conditioning variable in (0, 1)
        de (np.ndarray): Gumbel parameter (theta)

    Returns:
        np.ndarray: Conditional quantiles
    """
    q = np.asarray(q).reshape(-1, 1)
    u = np.asarray(u).reshape(-1, 1)
    de = np.asarray(de).reshape(-1, 1)

    p = 1 - q
    z1 = -np.log(np.maximum(u, UMIN))
    de1 = de - 1.0

    # Protect log(1-p) and log(z1)
    con = (
        np.log(np.maximum(1.0 - p, UMIN))
        - z1
        + (1.0 - de) * np.log(np.maximum(z1, UMIN))
    )

    # Initial guess
    a = np.power(2.0 * np.power(np.maximum(z1, UMIN), de), 1.0 / np.maximum(de, UMIN))
    mxdif = np.ones_like(a)
    iter_count = np.zeros_like(a, dtype=int)
    dif = 0.1 * np.ones_like(a)

    max_iter = 20
    tol = 1e-6

    while np.any((mxdif > tol) & (iter_count < max_iter)):
        mask = (mxdif > tol) & (iter_count < max_iter)

        # Protect log(a)
        g = a + de1 * np.log(np.maximum(a, UMIN)) + con
        gp = 1.0 + de1 / np.maximum(a, UMIN)

        # Check for NaN values
        nan_mask = mask & (np.isnan(g) | np.isnan(gp) | np.isnan(g / gp))
        valid_mask = mask & ~nan_mask

        # Handle NaN case (for de > 50)
        dif[nan_mask] /= -2.0

        # Normal Newton-Raphson step
        dif[valid_mask] = g[valid_mask] / gp[valid_mask]

        a -= dif
        iter_count += mask.astype(int)

        # Ensure a > z1
        inner_it = 0
        while np.any((a <= z1) & mask) and inner_it < 20:
            adjust_mask = (a <= z1) & mask
            dif[adjust_mask] /= 2.0
            a[adjust_mask] += dif[adjust_mask]
            inner_it += 1

        mxdif = np.abs(dif)

    # z2 = a * (1 - (z1/a)**de)**(1/de), factored so the exponent stays
    # negative; forming a**de and z1**de separately overflows for large de.
    la = np.log(np.maximum(a, UMIN))
    lz1 = np.log(np.maximum(z1, UMIN))
    ratio = np.exp(np.minimum(de * (lz1 - la), 0.0))
    z2 = a * np.exp(np.log1p(-np.minimum(ratio, 1.0 - UMIN)) / np.maximum(de, UMIN))
    out = np.exp(-z2)

    return out.squeeze()


##########################################################
### Functions for the derivatives and log-likelihood ####
##########################################################
def get_effective_rotation(theta_values: np.ndarray, family_code: int) -> np.ndarray:
    """
    Vectorized version of get_effective_rotation().
    Accepts an array of theta_values and returns corresponding rotations.

    Args:
        theta_values (np.ndarray): Copula parameter values (any shape)
        family_code (int): Family code (401–404)

    Returns:
        np.ndarray: Effective rotations (same shape as theta_values)
    """

    theta_values = np.asarray(theta_values)

    rot = np.empty_like(theta_values, dtype=int)

    if family_code == 41:
        rot[:] = np.where(theta_values > 0, 0, 2)
    elif family_code == 42:
        rot[:] = np.where(theta_values > 0, 0, 3)
    elif family_code == 43:
        rot[:] = np.where(theta_values > 0, 1, 2)
    elif family_code == 44:
        rot[:] = np.where(theta_values > 0, 1, 3)
    else:
        raise ValueError(
            f"Unsupported family code: {family_code}. Supported codes: 41, 42, 43, 44."
        )

    return rot


def _log_likelihood(y, theta, family_code=41):
    """
    Log-likelihood for the Gumbel copula.
    """

    theta = np.asarray(theta).copy()  # <- prevents in-place mutation of caller's array
    y = np.clip(y, UMIN, UMAX)
    u = y[:, 0].reshape(-1, 1)
    v = y[:, 1].reshape(-1, 1)

    rotation = get_effective_rotation(theta, family_code)

    u_rot, v_rot = u.copy(), v.copy()

    # 180° rotation (survival)
    mask_1 = rotation == 1
    u_rot[mask_1] = 1 - u[mask_1]
    v_rot[mask_1] = 1 - v[mask_1]

    # 90° rotation
    mask_2 = rotation == 2
    u_rot[mask_2] = 1 - u[mask_2]
    theta[mask_2] = -theta[mask_2]

    # 270° rotation
    mask_3 = rotation == 3
    v_rot[mask_3] = 1 - v[mask_3]
    theta[mask_3] = -theta[mask_3]

    # Gumbel copula log-likelihood following C implementation
    log_u = np.log(u_rot)
    log_v = np.log(v_rot)

    # s carried in log space; the powers over/underflow for large theta
    logs = np.logaddexp(theta * np.log(-log_u), theta * np.log(-log_v))
    A = np.exp(logs / theta)

    f = (
        -A
        + (2.0 / theta - 2.0) * logs
        + (theta - 1.0) * np.log(np.maximum(np.abs(log_u * log_v), UMIN))
        - np.log(np.maximum(u_rot * v_rot, UMIN))
        + np.log1p(np.maximum((theta - 1.0) / A, -1 + UMIN))
    )

    # Handle numerical limits
    XINFMAX = 700.0  # Approximate maximum for exp
    mask_high = f > XINFMAX
    mask_low = f < np.log(np.finfo(float).tiny)

    f[mask_high] = np.log(XINFMAX)
    f[mask_low] = np.log(np.finfo(float).tiny)

    f = np.where(f == 0, 1e-2, f)
    return f.squeeze()


def _derivative_1st(y, theta, family_code=41):
    """
    First derivative of the Gumbel copula log-likelihood with respect to theta.
    Robust version: clips/guards domains to avoid NaN/inf.
    """
    y = np.asarray(y, dtype=float)
    theta = np.asarray(theta, dtype=float).copy()

    # ---- small constants ----
    eps = 1e-12  # generic
    eps_theta = 1e-8  # keep away from 0 for 1/theta, 1/theta^2
    max_log_arg = 1e300  # cap arguments before log if needed
    sign = np.ones_like(theta)

    u = np.clip(y[:, 0], eps, 1.0 - eps).reshape(-1, 1)
    v = np.clip(y[:, 1], eps, 1.0 - eps).reshape(-1, 1)

    rotation = get_effective_rotation(theta, family_code)

    u_rot, v_rot = u.copy(), v.copy()

    # 180° rotation (survival)
    mask_1 = rotation == 1
    u_rot[mask_1] = 1.0 - u[mask_1]
    v_rot[mask_1] = 1.0 - v[mask_1]
    sign[mask_1] = 1.0

    mask_2 = rotation == 2
    u_rot[mask_2] = 1.0 - u[mask_2]
    theta[mask_2] = -theta[mask_2]
    sign[mask_2] = -1.0

    mask_3 = rotation == 3
    v_rot[mask_3] = 1.0 - v[mask_3]
    theta[mask_3] = -theta[mask_3]
    sign[mask_3] = -1.0

    # ---- guard theta away from 0 (preserve sign) ----
    theta = np.where(
        np.abs(theta) < eps_theta,
        np.sign(theta + 0.0) * eps_theta + (theta == 0) * eps_theta,
        theta,
    )
    # logs of u_rot, v_rot are negative because u_rot,v_rot in (0,1)
    t1 = np.log(u_rot)  # < 0
    t3 = np.log(v_rot)  # < 0
    # -t1, -t3 are positive, but guard anyway
    mt1 = np.maximum(-t1, eps)
    mt3 = np.maximum(-t3, eps)
    # d(log c)/d(theta). The common factors exp(-A), s**(2(1/theta-1)) and
    # (log u log v)**(theta-1) cancel analytically and are dropped rather than
    # evaluated, since each over/underflows for large theta.
    L1 = np.log(mt1)
    L3 = np.log(mt3)
    a1 = theta * L1
    a3 = theta * L3
    logs = np.logaddexp(a1, a3)
    w1 = np.exp(a1 - logs)
    w3 = np.exp(a3 - logs)
    D = w1 * L1 + w3 * L3
    A = np.exp(logs / theta)
    dA = A * (D / theta - logs / theta**2)

    denom_A = A + theta - 1.0
    denom_A = np.where(
        np.abs(denom_A) < eps, np.sign(denom_A + 0.0) * eps + (denom_A == 0) * eps,
        denom_A,
    )

    deriv = (
        -dA
        + (dA + 1.0) / denom_A
        - logs / theta**2
        + (1.0 / theta - 2.0) * D
        + L1
        + L3
    )
    deriv = np.where(np.isfinite(deriv), deriv, 0.0)
    deriv *= sign
    return deriv.squeeze()


def _derivative_2nd(y, theta, family_code=41):
    """
    Second derivative of the Gumbel copula log-likelihood with respect to theta.
    Robust version: guards theta, logs/powers, and divisions to avoid NaN/inf.
    """
    y = np.asarray(y, dtype=float)
    theta = np.asarray(theta, dtype=float).copy()

    eps = 1e-12
    eps_theta = 1e-8

    u = np.clip(y[:, 0], eps, 1.0 - eps).reshape(-1, 1)
    v = np.clip(y[:, 1], eps, 1.0 - eps).reshape(-1, 1)
    rotation = get_effective_rotation(theta, family_code)
    u_rot, v_rot = u.copy(), v.copy()

    # 180° rotation (survival)
    mask_1 = rotation == 1
    u_rot[mask_1] = 1.0 - u_rot[mask_1]
    v_rot[mask_1] = 1.0 - v_rot[mask_1]

    # 90° rotation
    mask_2 = rotation == 2
    v_rot[mask_2] = 1.0 - v_rot[mask_2]
    theta[mask_2] = -theta[mask_2]

    # 270° rotation
    mask_3 = rotation == 3
    u_rot[mask_3] = 1.0 - u_rot[mask_3]
    theta[mask_3] = -theta[mask_3]

    # guard theta away from 0 for 1/theta, 1/theta^k
    theta = np.where(
        np.abs(theta) < eps_theta,
        np.sign(theta + 0.0) * eps_theta + (theta == 0) * eps_theta,
        theta,
    )
    # d^2 c / d theta^2 = c * ((dl/dth)^2 + d^2 l/dth^2), l = log c. Built from
    # log s, D = dlog(s)/dth and Dd = d2log(s)/dth2, all numerically stable.
    t3 = np.log(np.maximum(u_rot, eps))
    t5 = np.log(np.maximum(v_rot, eps))
    mt3 = np.maximum(-t3, eps)
    mt5 = np.maximum(-t5, eps)

    L1 = np.log(mt3)
    L3 = np.log(mt5)
    logs = np.logaddexp(theta * L1, theta * L3)
    w1 = np.exp(theta * L1 - logs)
    w3 = np.exp(theta * L3 - logs)
    D = w1 * L1 + w3 * L3
    Dd = w1 * L1 * L1 + w3 * L3 * L3 - D * D  # a variance, so >= 0

    th2 = theta * theta
    th3 = th2 * theta
    G = logs / theta
    Gd = D / theta - logs / th2
    Gdd = Dd / theta - 2.0 * D / th2 + 2.0 * logs / th3
    A = np.exp(G)
    Ad = A * Gd
    Add = A * (Gd * Gd + Gdd)

    B = A + theta - 1.0
    B = np.where(np.abs(B) < eps, np.sign(B + 0.0) * eps + (B == 0) * eps, B)

    logc = (
        -A
        + np.log(np.abs(B))
        + (1.0 / theta - 2.0) * logs
        + (theta - 1.0) * (L1 + L3)
        - t3
        - t5
    )
    l1 = -Ad + (Ad + 1.0) / B - logs / th2 + (1.0 / theta - 2.0) * D + L1 + L3
    l2 = (
        -Add
        + Add / B
        - (Ad + 1.0) ** 2 / (B * B)
        - 2.0 * D / th2
        + 2.0 * logs / th3
        + (1.0 / theta - 2.0) * Dd
    )

    deriv = np.exp(logc) * (l1 * l1 + l2)
    # if anything still went non-finite, neutralize those entries
    deriv = np.where(np.isfinite(deriv), deriv, 0.0)

    return deriv.squeeze()
