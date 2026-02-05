# Author: Your Name
# MIT Licence
from typing import Dict

import numpy as np
import scipy.stats as st

from ..base import BivariateCopulaMixin, CopulaMixin, Distribution, LinkFunction
from ..links import KendallsTauToParameterGumbel, Log
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
        param_link: LinkFunction = KendallsTauToParameterGumbel(),
        family_code: int = 41,
    ):
        super().__init__(
            links={0: link},
            param_links={0: param_link},
        )
        self.family_code = family_code  # gamCopula family code (401, 402, 403, 404)
        self.is_multivariate = True
        self._regularization_allowed = {0: False}

    @property
    def rotation(self):
        """Return the effective rotation based on family code in degrees."""
        return {
            401: 0,
            402: 90,
            403: 180,
            404: 270,
        }[self.family_code]

    @staticmethod
    def fitted_elements(dim: int):
        return {0: 1}

    def theta_to_params(self, theta) -> np.ndarray:
        val = theta[0]
        return np.asarray(val).reshape(-1, 1)

    def set_initial_guess(self, theta, param):
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
        Implementation based on vinecopulib package.

        Args:
            u (np.ndarray): Array of shape (n,) with values in (0, 1).
            v (np.ndarray): Array of shape (n,) with values in (0, 1).
            theta (np.ndarray or float): Gumbel parameter(s), shape (n,) or scalar.
            un (int): Determines which conditional to compute (0 for h(u|v), 1 for h(v|u)).

        Returns:
            np.ndarray: Array of shape (n,) with conditional probabilities.
        """

        UMIN = 1e-12
        UMAX = 1 - 1e-12

        theta = np.asarray(
            theta
        ).copy()  # <- prevents in-place mutation of caller's array
        u = np.clip(u, UMIN, UMAX).reshape(-1, 1)
        v = np.clip(v, UMIN, UMAX).reshape(-1, 1)

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
        Implementation based on vinecopulib package.

        Args:
            u (np.ndarray): Array of shape (n,) with values in (0, 1).
            v (np.ndarray): Array of shape (n,) with values in (0, 1).
            theta (np.ndarray or float): Gumbel parameter(s), shape (n,) or scalar.
            un (int): Determines which conditional to compute.

        Returns:
            np.ndarray: Array of shape (n,) with inverse conditional probabilities.
        """
        UMIN = 1e-12
        UMAX = 1 - 1e-12

        # Apply clipping using masks
        u_mask_low = u < UMIN
        u_mask_high = u > UMAX
        v_mask_low = v < UMIN
        v_mask_high = v > UMAX

        u = np.where(u_mask_low, UMIN, u)
        u = np.where(u_mask_high, UMAX, u)
        v = np.where(v_mask_low, UMIN, v)
        v = np.where(v_mask_high, UMAX, v)
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
    UMIN = 1e-12

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

    # Final calculation with numerical protection
    z2 = np.power(
        np.maximum(
            np.power(np.maximum(a, UMIN), de) - np.power(np.maximum(z1, UMIN), de), UMIN
        ),
        1.0 / np.maximum(de, UMIN),
    )
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
    UMIN = 1e-12
    UMAX = 1 - 1e-12

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

    t1 = (-log_u) ** theta + (-log_v) ** theta

    f = (
        -(t1 ** (1.0 / theta))
        + (2.0 / theta - 2.0) * np.log(np.maximum(t1, UMIN))
        + (theta - 1.0) * np.log(np.maximum(np.abs(log_u * log_v), UMIN))
        - np.log(np.maximum(u_rot * v_rot, UMIN))
        + np.log1p(np.maximum((theta - 1.0) * t1 ** (-1.0 / theta), -1 + UMIN))
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
    theta = np.where(np.abs(theta) < eps_theta, np.sign(theta + 0.0) * eps_theta + (theta == 0) * eps_theta, theta)
    # logs of u_rot, v_rot are negative because u_rot,v_rot in (0,1)
    t1 = np.log(u_rot)                      # < 0
    t3 = np.log(v_rot)                      # < 0
    # -t1, -t3 are positive, but guard anyway
    mt1 = np.maximum(-t1, eps)
    mt3 = np.maximum(-t3, eps)
    # power terms: (-log u)^theta, (-log v)^theta
    t2 = np.power(mt1, theta)
    t4 = np.power(mt3, theta)
    t5 = t2 + t4
    t5_safe = np.maximum(t5, eps)           # for log(t5), 1/t5, power(t5, ...)
    t6 = 1.0 / theta
    t7 = np.power(t5_safe, t6)
    t8 = theta * theta
    t10 = np.log(np.minimum(t5_safe, max_log_arg))
    t11 = (1.0 / t8) * t10
    # log(-log u), log(-log v)
    t12 = np.log(mt1)
    t14 = np.log(mt3)
    t16 = t2 * t12 + t4 * t14
    t18 = 1.0 / t5_safe

    t20 = -t11 + t6 * t16 * t18
    # exp(-t7) (safe enough; t7>=0)
    t22 = np.exp(-t7)
    t23 = -1.0 + t6
    # power(t5, 2*t23) can blow if t5~0 and exponent negative -> guard with t5_safe
    t24 = np.power(t5_safe, 2.0 * t23)
    t25 = t22 * t24
    t27 = t1 * t3                          # positive (neg*neg), but guard
    t27_safe = np.maximum(t27, eps)

    t28 = theta - 1.0
    t29 = np.power(t27_safe, t28)
    t30 = np.power(t5_safe, -t6)
    t31 = t28 * t30
    t32 = 1.0 + t31
    # avoid division by ~0 later
    t32_safe = np.where(np.abs(t32) < eps, np.sign(t32 + 0.0) * eps + (t32 == 0) * eps, t32)

    t34 = 1.0 / u_rot
    t35 = 1.0 / v_rot
    t36 = t34 * t35
    t37 = t29 * t32_safe * t36
    t45 = t25 * t29
    t46 = np.log(t27_safe)

    # ---- validity mask (stronger) ----
    mask = (
        np.isfinite(theta)
        & np.isfinite(t5_safe)
        & np.isfinite(t27_safe)
        & np.isfinite(t22)
        & np.isfinite(t24)
        & np.isfinite(t29)
        & np.isfinite(t32_safe)
        & (t5_safe > 0)
        & (t27_safe > 0)
        & (np.abs(t32_safe) > 0)
    )
    deriv = np.zeros_like(theta)

    # denominator: t22*t24*t29*t32 (guarded)
    denom = t22 * t24 * t29 * t32_safe
    denom_safe = np.where(np.abs(denom) < eps, np.sign(denom + 0.0) * eps + (denom == 0) * eps, denom)
    numer = (
        -t7 * t20 * t25 * t37
        + t25 * (-2.0 * t11 + 2.0 * t23 * t16 * t18) * t37
        + t45 * t46 * t32_safe * t36
        + t45 * (t30 - t31 * t20) * t34 * t35
    )
    deriv[mask] = (numer[mask] / denom_safe[mask]) * u_rot[mask] * v_rot[mask]
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
    # core logs (u_rot,v_rot already clipped away from 0/1, but keep safe)
    t3 = np.log(np.maximum(u_rot, eps))
    t5 = np.log(np.maximum(v_rot, eps))
    mt3 = np.maximum(-t3, eps)
    mt5 = np.maximum(-t5, eps)
    t4 = np.power(mt3, theta)
    t6 = np.power(mt5, theta)
    t7 = t4 + t6
    t7s = np.maximum(t7, eps)
    t8 = 1.0 / theta
    t9 = np.power(t7s, t8)
    t10 = theta * theta
    t11 = 1.0 / t10
    t12 = np.log(t7s)
    t13 = t11 * t12
    t14 = np.log(mt3)
    t16 = np.log(mt5)
    t18 = t4 * t14 + t6 * t16
    t20 = 1.0 / t7s

    t22 = -t13 + t8 * t18 * t20
    t23 = t22 * t22
    t25 = np.exp(-t9)
    # divisions by u_rot, v_rot safe because clipped, but keep max
    inv_u = 1.0 / np.maximum(u_rot, eps)
    inv_v = 1.0 / np.maximum(v_rot, eps)
    t27 = t25 * inv_u
    t29 = inv_v

    t30 = -1.0 + t8
    t31 = np.power(t7s, 2.0 * t30)
    t32 = t29 * t31
    t33 = t3 * t5
    t33s = np.maximum(np.abs(t33), eps)
    t34 = theta - 1.0
    t35 = np.power(t33s, t34)
    t36 = np.power(t7s, -t8)
    t37 = t34 * t36
    t38 = 1.0 + t37
    t38s = np.where(np.abs(t38) < eps, np.sign(t38 + 0.0) * eps + (t38 == 0) * eps, t38)
    t39 = t35 * t38s
    t40 = t32 * t39
    t44 = (1.0 / t10) * (1.0 / theta) * t12
    t47 = t11 * t18 * t20
    t49 = t14 * t14
    t51 = t16 * t16
    t53 = t4 * t49 + t6 * t51
    t56 = t18 * t18
    t58 = t7s * t7s
    t59 = 1.0 / np.maximum(t58, eps)
    t61 = 2.0 * t44 - 2.0 * t47 + t8 * t53 * t20 - t8 * t56 * t59
    t65 = t9 * t9
    t70 = t9 * t22 * t27
    t74 = -2.0 * t13 + 2.0 * t30 * t18 * t20
    t75 = t74 * t35
    t80 = np.log(t33s)
    t87 = t36 - t37 * t22
    t88 = t35 * t87
    t17 = t27 * t29
    t15 = t74 * t74
    t2 = t31 * t35
    t1 = t80 * t80

    deriv = (
        -t9 * t23 * t27 * t40
        - t9 * t61 * t27 * t40
        + t65 * t23 * t27 * t40
        - 2.0 * t70 * t32 * t75 * t38s
        - 2.0 * t70 * t32 * t35 * t80 * t38s
        - 2.0 * t70 * t32 * t88
        + t17 * t31 * t15 * t39
        + t17
        * t31
        * (4.0 * t44 - 4.0 * t47 + 2.0 * t30 * t53 * t20 - 2.0 * t30 * t56 * t59)
        * t39
        + 2.0 * t27 * t32 * t75 * t80 * t38s
        + 2.0 * t17 * t31 * t74 * t88
        + t17 * t2 * t1 * t38s
        + 2.0 * t17 * t2 * t80 * t87
        + t17 * t2 * (-2.0 * t36 * t22 + t37 * t23 - t37 * t61)
    )
    # if anything still went non-finite, neutralize those entries
    deriv = np.where(np.isfinite(deriv), deriv, 0.0)

    return deriv.squeeze()
