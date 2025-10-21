# Author: Your Name
# MIT Licence
from typing import Dict

import numpy as np
import scipy.stats as st

from ..base import BivariateCopulaMixin, CopulaMixin, Distribution, LinkFunction
from ..links import KendallsTauToParameterClayton, Log
from ..types import ParameterShapes


class BivariateCopulaClayton(BivariateCopulaMixin, CopulaMixin, Distribution):
    """
    Bivariate Clayton copula distribution class.
    """

    corresponding_gamlss: str = None
    parameter_names = {0: "theta"}
    parameter_support = {0: (1e-6, np.inf)}
    distribution_support = (0, 1)
    n_params = len(parameter_names)
    parameter_shape = {0: ParameterShapes.SCALAR}

    def __init__(
        self,
        link: LinkFunction = Log(),
        param_link: LinkFunction = KendallsTauToParameterClayton(),
        family_code: int = 301,
    ):
        super().__init__(links={0: link}, param_links={0: param_link}, rotation=0)
        self.family_code = family_code
        self.is_multivariate = True
        self._adr_lower_diag = {0: False}
        self._regularization_allowed = {0: False}
        self._regularization = ""
        self._scoring = "fisher"

    @staticmethod
    def fitted_elements(dim: int):
        return {0: int(dim * (dim - 1) // 2)}

    @property
    def param_structure(self):
        return self._param_structure

    @staticmethod
    def set_theta_element(theta: dict, value: np.ndarray, param: int, k: int) -> dict:
        theta[param] = value
        return theta

    def theta_to_params(self, theta):
        chol = theta[0]
        return chol

    def theta_to_scipy_params(self, theta: np.ndarray) -> dict:
        return {"theta": theta}

    def set_initial_guess(self, theta, param):
        return theta

    def initial_values(self, y, param=0):
        M = y.shape[0]
        # Compute the empirical Pearson correlation for each sample
        # y is expected to be (M, 2)
        tau = st.kendalltau(y[:, 0], y[:, 1]).correlation
        chol = np.full((M, 1), tau)
        return chol

    def param_conditional_likelihood(
        self, y: np.ndarray, theta: dict, eta: np.ndarray, param: int
    ) -> np.ndarray:
        fitted = self.flat_to_cube(eta, param=param)
        fitted = self.link_inverse(fitted, param=param)
        return self.log_likelihood(y, theta={**theta, param: fitted})

    def theta_to_scipy(self, theta: dict):
        return {"theta": theta}

    def logpdf(self, y, theta):
        theta_logpdf = self.theta_to_params(theta)
        result = _log_likelihood(y, theta_logpdf, self.family_code)
        return result

    def pdf(self, y, theta):
        return np.exp(self.logpdf(y, theta))

    def dl1_dp1(self, y, theta, param=0):
        theta = self.theta_to_params(theta)
        return _derivative_1st(y, theta, self.family_code)

    def dl2_dp2(self, y, theta, param=0, clip=False):
        """
        Second derivative with proper chain rule matching gamBiCopFit.R
        """
        theta = self.theta_to_params(theta)
        return _derivative_2nd(y, theta, self.family_code)

    def element_score(self, y, theta, param=0, k=0):
        return self.element_dl1_dp1(y, theta, param, k)

    def element_hessian(self, y, theta, param=0, k=0):
        return self.element_dl2_dp2(y, theta, param, k)

    def element_dl1_dp1(self, y, theta, param=0, k=0, clip=False):
        # Apply proper chain rule like dl1_dp1
        theta_params = self.theta_to_params(theta)
        deriv = _derivative_1st(y, chol=theta_params, family_code=self.family_code)
        return deriv

    def element_dl2_dp2(self, y, theta, param=0, k=0, clip=False):
        theta_params = self.theta_to_params(theta)
        deriv = _derivative_2nd(y, theta=theta_params, family_code=self.family_code)
        return deriv

    def dl2_dpp(self, y, theta, param=0):
        raise NotImplementedError("Not implemented for Clayton copula.")

    def calculate_conditional_initial_values(self, y: np.ndarray, theta: dict) -> dict:
        raise NotImplementedError("Not implemented for Clayton copula.")

    def cdf(self, y, theta):
        raise NotImplementedError("Not implemented for Clayton copula.")

    def ppf(self, q, theta):
        raise NotImplementedError("Not implemented for Clayton copula.")

    def rvs(self, size, theta):
        theta = self.theta_to_params(theta)
        u = np.random.uniform(size=size)
        w = np.random.uniform(size=size)
        v = (w ** (-theta / (1 + theta)) * (u ** (-theta) - 1) + 1) ** (-1 / theta)
        return np.column_stack((u, v))

    def logcdf(self, y, theta):
        raise NotImplementedError("Not implemented for Clayton copula.")

    def logpmf(self, y, theta):
        raise NotImplementedError("Not implemented for Clayton copula.")

    def pmf(self, y, theta):
        raise NotImplementedError("Not implemented for Clayton copula.")

    def get_regularization_size(self, dim: int) -> int:
        return dim

    def get_effective_rotation(
        theta_values: np.ndarray, family_code: int
    ) -> np.ndarray:
        """
        Vectorized version of get_effective_rotation().
        Accepts an array of theta_values and returns corresponding rotations.

        Args:
            theta_values (np.ndarray): Copula parameter values (any shape)
            family_code (int): Family code (301–304)

        Returns:
            np.ndarray: Effective rotations (same shape as theta_values)
        """
        theta_values = np.asarray(theta_values)
        rot = np.empty_like(theta_values, dtype=int)

        if family_code == 301:
            rot[:] = np.where(theta_values > 0, 0, 2)
        elif family_code == 302:
            rot[:] = np.where(theta_values > 0, 0, 3)
        elif family_code == 303:
            rot[:] = np.where(theta_values > 0, 1, 2)
        elif family_code == 304:
            rot[:] = np.where(theta_values > 0, 1, 3)
        else:
            raise ValueError(
                f"Unsupported family code: {family_code}. Supported codes: 301, 302, 303, 304."
            )

        return rot

    def hfunc(
        self, u: np.ndarray, v: np.ndarray, theta: np.ndarray, un: int, family_code: int
    ) -> np.ndarray:

        UMIN = 1e-12
        UMAX = 1 - 1e-12

        theta = np.asarray(
            theta
        ).copy()  # <- prevents in-place mutation of caller's array

        u = np.clip(u, UMIN, UMAX).reshape(-1, 1)
        v = np.clip(v, UMIN, UMAX).reshape(-1, 1)

        # Swap u and v if un == 1
        if un == 1:
            u, v = v, u

        # Get rotations for all samples
        rotation = get_effective_rotation(theta, family_code)

        # Apply rotation transformations vectorized
        u_rot, v_rot = u.copy(), v.copy()

        # 180° rotation (survival)
        mask_1 = rotation == 1
        u_rot[mask_1] = 1 - u[mask_1]
        v_rot[mask_1] = 1 - v[mask_1]

        # 90° rotation
        mask_2 = rotation == 2
        v_rot[mask_2] = 1 - v[mask_2]
        theta[mask_2] = -theta[mask_2]

        # 270° rotation
        mask_3 = rotation == 3
        v_rot[mask_3] = 1 - v[mask_3]
        theta[mask_3] = -theta[mask_3]

        # Vectorized conditional distribution computation
        t1 = v_rot ** (-theta - 1)
        t2 = u_rot ** (-theta) + v_rot ** (-theta) - 1
        t2 = np.maximum(t2, UMIN)
        t3 = -1.0 - 1.0 / theta
        h = t1 * (t2**t3)

        mask_4 = theta < 1e-4
        h[mask_4] = u[mask_4]

        # Apply rotation-specific transformations
        h[mask_3] = 1 - h[mask_3]  # 270° rotation
        h = np.clip(h, UMIN, UMAX)
        return h.squeeze()


##########################################################
# Helper functions for the log-likelihood and derivatives #
##########################################################


def get_effective_rotation(theta_values: np.ndarray, family_code: int) -> np.ndarray:
    """
    Vectorized version of get_effective_rotation().
    Accepts an array of theta_values and returns corresponding rotations.

    Args:
        theta_values (np.ndarray): Copula parameter values (any shape)
        family_code (int): Family code (301–304)

    Returns:
        np.ndarray: Effective rotations (same shape as theta_values)
    """
    theta_values = np.asarray(theta_values)
    rot = np.empty_like(theta_values, dtype=int)

    if family_code == 301:
        rot[:] = np.where(theta_values > 0, 0, 2)
    elif family_code == 302:
        rot[:] = np.where(theta_values > 0, 0, 3)
    elif family_code == 303:
        rot[:] = np.where(theta_values > 0, 1, 2)
    elif family_code == 304:
        rot[:] = np.where(theta_values > 0, 1, 3)
    else:
        raise ValueError(
            f"Unsupported family code: {family_code}. Supported codes: 301, 302, 303, 304."
        )

    return rot


def _log_likelihood(y, theta, family_code=None):

    theta = np.asarray(theta).copy()  # <- prevents in-place mutation of caller's array

    u = y[:, 0].reshape(-1, 1)
    v = y[:, 1].reshape(-1, 1)

    rotation = get_effective_rotation(theta, family_code)

    # Apply rotation transformations to data vectorized
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

    # Vectorized computation for all samples at once
    valid_mask = (theta != 0) & (theta >= 1e-10)

    t5 = u_rot ** (-theta)
    t6 = v_rot ** (-theta)
    t7 = np.maximum(t5 + t6 - 1.0, 1e-12)

    f = np.where(
        valid_mask,
        np.log1p(theta)
        - (1.0 + theta) * np.log(np.maximum(u_rot * v_rot, 1e-12))
        - (2.0 + 1.0 / theta) * np.log(t7),
        0,
    )

    # Apply bounds
    XINFMAX = 700
    DBL_MIN = 2.2e-308
    f = np.clip(f, np.log(DBL_MIN), XINFMAX)
    return f.squeeze()  # Always 1D


def _derivative_1st(y, chol, family_code):
    """
    Computes the first derivative of the bivariate Clayton copula log-likelihood
    with respect to theta, supporting rotations via copula_code.

    Args:
        y (np.ndarray): Input data of shape (M, 2)
        theta (np.ndarray or float): Copula parameter(s), shape (M,) or scalar
        copula_code (int): Copula family code (3=Clayton, 13/23/33=rotated)

    Returns:
        np.ndarray: First derivative, shape (M,)
    """
    chol = np.asarray(chol).copy()  # <- prevents in-place mutation of caller's array
    sign = np.ones_like(chol)

    y = y.copy()

    u = np.clip(y[:, 0], 1e-12, 1 - 1e-12).reshape(-1, 1)
    v = np.clip(y[:, 1], 1e-12, 1 - 1e-12).reshape(-1, 1)

    rotation = get_effective_rotation(chol, family_code)

    u_rot, v_rot = u.copy(), v.copy()

    # 180° rotation (survival)
    mask_1 = rotation == 1
    u_rot[mask_1] = 1 - u_rot[mask_1]
    v_rot[mask_1] = 1 - v_rot[mask_1]
    sign[mask_1] = 1.0

    mask_2 = rotation == 2
    u_rot[mask_2] = 1 - u_rot[mask_2]
    chol[mask_2] = -chol[mask_2]
    sign[mask_2] = -1.0

    mask_3 = rotation == 3
    v_rot[mask_3] = 1 - v_rot[mask_3]
    chol[mask_3] = -chol[mask_3]
    sign[mask_3] = -1.0

    t4 = np.log(u_rot * v_rot)
    t5 = u_rot ** (-chol)
    t6 = v_rot ** (-chol)
    t7 = t5 + t6 - 1.0
    t8 = np.log(t7)
    t9 = chol**2
    t14 = np.log(u_rot)
    t16 = np.log(v_rot)
    result = (
        1.0 / (1.0 + chol)
        - t4
        + t8 / t9
        + (1.0 / chol + 2.0) * (t5 * t14 + t6 * t16) / t7
    )
    deriv = sign * result

    return deriv.squeeze()


def _derivative_2nd(y, theta, family_code):
    """
    Second derivative of Clayton copula PDF w.r.t. parameter theta.
    Based on diff2PDF_mod from VineCopula C code.

    Args:
        y: array of shape (n, 2) - copula data [u, v]
        theta: array of shape (n,) or scalar - copula parameters
        copula: array of shape (n,) or scalar - copula family codes

    Returns:
        np.ndarray: second derivative values for each observation
    """

    # Constants for numerical stability
    UMIN = 1e-12
    UMAX = 1 - 1e-12
    theta = np.asarray(theta).copy()  # <- prevents in-place mutation of caller's array

    # Extract u and v from the y matrix
    u = np.clip(y[:, 0], UMIN, UMAX).reshape(-1, 1)
    v = np.clip(y[:, 1], UMIN, UMAX).reshape(-1, 1)

    rotation = get_effective_rotation(theta, family_code)

    u_rot, v_rot = u.copy(), v.copy()

    # 180° rotation (survival)
    mask_1 = rotation == 1
    u_rot[mask_1] = 1 - u[mask_1]
    v_rot[mask_1] = 1 - v[mask_1]

    mask_2 = rotation == 2
    v_rot[mask_2] = 1 - v[mask_2]
    theta[mask_2] = -theta[mask_2]

    mask_3 = rotation == 3
    v_rot[mask_3] = 1 - v[mask_3]
    theta[mask_3] = -theta[mask_3]

    # Basic terms (matching C variable names)
    t1 = u_rot * v_rot
    t2 = -theta - 1.0
    t3 = np.power(t1, t2)
    t4 = np.log(t1)

    t6 = np.power(u_rot, -theta)
    t7 = np.power(v_rot, -theta)
    t8 = t6 + t7 - 1.0

    t10 = -2.0 - 1.0 / theta
    t11 = np.power(t8, t10)

    # Higher order terms
    t15 = theta * theta
    t16 = 1.0 / t15
    t17 = np.log(t8)

    t19 = np.log(u_rot)
    t21 = np.log(v_rot)

    t24 = -t6 * t19 - t7 * t21

    t26 = 1.0 / t8
    t27 = t16 * t17 + t10 * t24 * t26

    t30 = -t2 * t3
    t32 = t4 * t4
    t14 = t27 * t27
    t13 = t19 * t19
    t12 = t21 * t21
    t9 = t24 * t24
    t5 = t8 * t8

    # The complete second derivative expression from C code
    term1 = -2.0 * t3 * t4 * t11
    term2 = 2.0 * t3 * t11 * t27
    term3 = t30 * t32 * t11
    term4 = -2.0 * t30 * t4 * t11 * t27
    term5 = t30 * t11 * t14

    # Additional correction terms
    t67 = 2.0 / (t15 * theta)  # Triple derivative term
    t70 = t6 * t13 + t7 * t12  # Second log derivative terms
    t74 = t9 / t5  # Ratio correction

    correction = (
        t30 * t11 * (-t67 * t17 + 2.0 * t16 * t24 * t26 + t10 * t70 * t26 - t10 * t74)
    )

    deriv = term1 + term2 + term3 + term4 + term5 + correction

    return deriv.squeeze()


def _log_likelihood_old(y, theta, family_code=None):
    M = y.shape[0]
    f = np.empty(M)
    u = y[:, 0]
    v = y[:, 1]

    # Handle edge cases like in C code

    u_valid = u if hasattr(u, "__len__") else u
    v_valid = v if hasattr(v, "__len__") else v

    for m in range(M):
        if M == 1:
            theta_valid = theta
        else:
            theta_valid = theta[m]

        rotation = get_effective_rotation(theta_valid, family_code)

        # Apply rotation transformations to data
        u_rot, v_rot = u_valid[m], v_valid[m]
        if rotation == 1:  # 180° rotation (survival)
            u_rot = 1 - u_valid[m]
            v_rot = 1 - v_valid[m]
        elif rotation == 2:  # 90° rotation
            u_rot = 1 - u_valid[m]
            theta_valid = -theta_valid
        elif rotation == 3:  # 270° rotation
            # u_rot stays the same
            v_rot = 1 - v_valid[m]
            theta_valid = -theta_valid

        if theta_valid == 0:
            f[m] = 0
        elif theta_valid < 1e-10:
            f[m] = 0
        else:
            t5 = u_rot ** (-theta_valid)
            t6 = v_rot ** (-theta_valid)
            t7 = t5 + t6 - 1.0
            t7 = np.maximum(t7, 1e-12)

            f[m] = (
                +np.log1p(theta_valid)  # log1p instead of log(theta + 1)
                - (1.0 + theta_valid)
                * np.log(
                    np.maximum(u_rot * v_rot, 1e-12)
                )  # log(u*v) with numerical protection
                - (2.0 + 1.0 / theta_valid) * np.log(t7)
            )

            # Apply bounds like C code
            XINFMAX = 700  # Approximate value for log overflow protection
            DBL_MIN = 2.2e-308
            f[m] = np.where(f[m] > XINFMAX, XINFMAX, f[m])
            f[m] = np.where(f[m] < np.log(DBL_MIN), np.log(DBL_MIN), f[m])

    return f  # Always 1D


def _derivative_2nd_old(y, theta, family_code):
    """
    Second derivative of Clayton copula PDF w.r.t. parameter theta.
    Based on diff2PDF_mod from VineCopula C code.

    Args:
        y: array of shape (n, 2) - copula data [u, v]
        theta: array of shape (n,) or scalar - copula parameters
        copula: array of shape (n,) or scalar - copula family codes

    Returns:
        np.ndarray: second derivative values for each observation
    """

    # Constants for numerical stability
    UMIN = 1e-12
    UMAX = 1 - 1e-12

    # Extract u and v from the y matrix
    u = np.clip(y[:, 0], UMIN, UMAX)
    v = np.clip(y[:, 1], UMIN, UMAX)

    # Ensure arrays - FIX: Handle scalar theta properly
    u = np.atleast_1d(u)
    v = np.atleast_1d(v)

    # Fix the theta handling
    if np.isscalar(theta):
        theta = np.full(len(u), theta)
    else:
        theta = np.atleast_1d(theta)

    n = len(u)
    out = np.zeros(n)

    for i in range(n):
        theta_i = theta[i]  # Now this will work since theta is always an array
        rotation = get_effective_rotation(theta_i, family_code)

        if rotation == 0:  # Standard Clayton
            uu, vv = u[i], v[i]
            th = theta_i
        elif rotation == 1:  # 180° rotated Clayton (survival)
            uu, vv = 1 - u[i], 1 - v[i]
            th = theta_i
        elif rotation == 2:  # 90° rotated Clayton
            uu, vv = u[i], 1 - v[i]
            th = -theta_i
        elif rotation == 3:  # 270° rotated Clayton
            uu, vv = u[i], 1 - v[i]
            th = -theta_i
        else:
            raise NotImplementedError("Copula family not implemented.")

        # Basic terms (matching C variable names)
        t1 = uu * vv
        t2 = -th - 1.0
        t3 = np.power(t1, t2)
        t4 = np.log(t1)

        t6 = np.power(uu, -th)
        t7 = np.power(vv, -th)
        t8 = t6 + t7 - 1.0

        t10 = -2.0 - 1.0 / th
        t11 = np.power(t8, t10)

        # Higher order terms
        t15 = th * th
        t16 = 1.0 / t15
        t17 = np.log(t8)

        t19 = np.log(uu)
        t21 = np.log(vv)

        t24 = -t6 * t19 - t7 * t21

        t26 = 1.0 / t8
        t27 = t16 * t17 + t10 * t24 * t26

        t30 = -t2 * t3
        t32 = t4 * t4
        t14 = t27 * t27
        t13 = t19 * t19
        t12 = t21 * t21
        t9 = t24 * t24
        t5 = t8 * t8

        # The complete second derivative expression from C code
        term1 = -2.0 * t3 * t4 * t11
        term2 = 2.0 * t3 * t11 * t27
        term3 = t30 * t32 * t11
        term4 = -2.0 * t30 * t4 * t11 * t27
        term5 = t30 * t11 * t14

        # Additional correction terms
        t67 = 2.0 / (t15 * th)  # Triple derivative term
        t70 = t6 * t13 + t7 * t12  # Second log derivative terms
        t74 = t9 / t5  # Ratio correction

        correction = (
            t30
            * t11
            * (-t67 * t17 + 2.0 * t16 * t24 * t26 + t10 * t70 * t26 - t10 * t74)
        )

        result = term1 + term2 + term3 + term4 + term5 + correction
        out[i] = result

    return out


def _derivative_1st_old(y, theta, family_code):
    """
    Computes the first derivative of the bivariate Clayton copula log-likelihood
    with respect to theta, supporting rotations via copula_code.

    Args:
        y (np.ndarray): Input data of shape (M, 2)
        theta (np.ndarray or float): Copula parameter(s), shape (M,) or scalar
        copula_code (int): Copula family code (3=Clayton, 13/23/33=rotated)

    Returns:
        np.ndarray: First derivative, shape (M,)
    """
    M = y.shape[0]
    deriv = np.empty((M,), dtype=np.float64)
    rotation = np.empty((M,), dtype=np.float64)
    u = np.clip(y[:, 0], 1e-12, 1 - 1e-12)
    v = np.clip(y[:, 1], 1e-12, 1 - 1e-12)

    for m in range(M):
        th = theta[m] if hasattr(theta, "__len__") else theta

        rotation[m] = get_effective_rotation(th, family_code)

        # Handle rotations
        if rotation[m] == 0:  # Standard Clayton
            uu, vv = u[m], v[m]
            th = th
            sign = 1
        elif rotation[m] == 1:  # 180° rotated Clayton (survival)
            (
                uu,
                vv,
            ) = (
                1 - u[m],
                1 - v[m],
            )
            sign = 1

        elif rotation[m] == 2:  # 90° rotated Clayton
            uu, vv = 1 - u[m], v[m]
            th = -th
            sign = -1.0
        elif rotation[m] == 3:  # 270° rotated Clayton
            uu, vv = u[m], 1 - v[m]
            th = -th
            sign = -1.0
        else:
            raise NotImplementedError("Copula family not implemented.")

        t4 = np.log(uu * vv)
        t5 = uu ** (-th)
        t6 = vv ** (-th)
        t7 = t5 + t6 - 1.0
        t8 = np.log(t7)
        t9 = th**2
        t14 = np.log(uu)
        t16 = np.log(vv)
        result = (
            1.0 / (1.0 + th)
            - t4
            + t8 / t9
            + (1.0 / th + 2.0) * (t5 * t14 + t6 * t16) / t7
        )
        deriv[m] = sign * result

    return deriv


def hfunc_old(
    self, u: np.ndarray, v: np.ndarray, theta: np.ndarray, un: int, family_code: int
) -> np.ndarray:
    M = u.shape[0]
    UMIN = 1e-12
    UMAX = 1 - 1e-12

    u = np.clip(u, UMIN, UMAX)
    v = np.clip(v, UMIN, UMAX)

    # Swap u and v if un == 1
    if un == 1:
        u, v = v, u

    h = np.empty(M)
    for m in range(M):
        th = theta[m] if hasattr(theta, "__len__") else theta

        rotation = get_effective_rotation(th, family_code)

        # Apply rotation transformations to data
        u_rot, v_rot = u[m], v[m]
        if rotation == 1:  # 180° rotation (survival)
            u_rot = 1 - u[m]
            v_rot = 1 - v[m]
        elif rotation == 2:  # 90° rotation
            u_rot = 1 - u[m]
            th = -th
        elif rotation == 3:  # 270° rotation
            # u_rot stays the same
            u_rot = 1 - u[m]
            th = -th

        # Conditional distribution function for Clayton copula
        # h(v|u) = ∂C(u,v)/∂v = (u^{-θ-1}) * (u^{-θ} + v^{-θ} - 1)^{-1/θ - 1}
        t1 = v_rot ** (-th - 1)
        t2 = u_rot ** (-th) + v_rot ** (-th) - 1
        t2 = np.maximum(t2, UMIN)
        t3 = -1.0 - 1.0 / th
        h[m] = t1 * (t2**t3)

    if rotation == 3:  # 270° rotation
        h = 1 - np.clip(h, UMIN, UMAX)
    else:
        h = np.clip(h, UMIN, UMAX)
    return h
