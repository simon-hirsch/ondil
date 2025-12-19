# Author Simon Hirsch
# MIT Licence
import math
from math import exp, fabs, log
from typing import Dict, List, Tuple

import numpy as np
import scipy.special as sp
import scipy.stats as st

from ..base import BivariateCopulaMixin, CopulaMixin, Distribution, LinkFunction
from ..links import FisherZLink, KendallsTauToParameter, LogShiftTwo
from ..types import ParameterShapes

UMIN = 1e-12
UMAX = 1 - 1e-12


class BivariateCopulaStudentT(BivariateCopulaMixin, CopulaMixin, Distribution):
    corresponding_gamlss: str = None
    parameter_names = {0: "rho", 1: "nu"}
    parameter_support = {0: (-1, 1), 1: (2, np.inf)}
    distribution_support = (-1, 1)
    n_params = len(parameter_names)
    parameter_shape = {
        0: ParameterShapes.SCALAR,
        1: ParameterShapes.SCALAR,
    }

    def __init__(
        self,
        link_1: LinkFunction = FisherZLink(),
        link_2: LinkFunction = LogShiftTwo(),
        param_link_1: LinkFunction = KendallsTauToParameter(),
        param_link_2: LinkFunction = KendallsTauToParameter(),
        family_code: int = 2,
    ):
        super().__init__(
            links={0: link_1, 1: link_2},
            param_links={0: param_link_1, 1: param_link_2},
        )
        self.family_code = family_code
        self.is_multivariate = True
        self._regularization_allowed = {0: False, 1: False}

    @staticmethod
    def fitted_elements(dim: int):
        return {0: 1, 1: 1}

    def theta_to_params(self, theta):
        return theta[0], theta[1]  # default nu = 4

    def set_initial_guess(self, theta, param):
        return theta

    def dl1_dp1(self, y: np.ndarray, theta: Dict, param: int = 0):
        """Return the first derivatives wrt to the parameter.

        Args:
            y (np.ndarray): Y values of shape n x d
            theta (Dict): Dict with parameters
            param (int, optional): Which parameter derivatives to return. Defaults to 0.

        Returns:
            derivative: The 1st derivatives.
        """
        rho, nu = self.theta_to_params(theta)

        if param == 0:  # derivative wrt rho
            deriv = _derivative_1st_rho(y=y, rho=rho, nu=nu)
        else:  # derivative wrt nu
            deriv = _derivative_1st_nu(y=y, rho=rho, nu=nu)

        return deriv

    def dl2_dp2(self, y: np.ndarray, theta: Dict, param: int = 0, clip=False):
        """Return the second derivatives wrt to the parameter.

        Args:
            y (np.ndarray): Y values of shape n x d
            theta (Dict): Dict with parameters
            param (int, optional): Which parameter derivatives to return. Defaults to 0.

        Returns:
            derivative: The 2nd derivatives.
        """
        rho, nu = self.theta_to_params(theta)

        if param == 0:  # derivative wrt rho
            deriv = _derivative_2nd_rho(y=y, rho=rho, nu=nu)
        else:  # derivative wrt nu
            deriv = _derivative_2nd_nu(y=y, rho=rho, nu=nu)

        return deriv

    def element_score(self, y: np.ndarray, theta: Dict, param: int = 0, k: int = 0):
        return self.element_dl1_dp1(y=y, theta=theta, param=param, k=k)

    def element_hessian(self, y: np.ndarray, theta: Dict, param: int = 0, k: int = 0):
        return self.element_dl2_dp2(y=y, theta=theta, param=param, k=k)

    def element_dl1_dp1(
        self, y: np.ndarray, theta: Dict, param: int = 0, k: int = 0, clip=False
    ):
        rho, nu = self.theta_to_params(theta)
        if param == 0:
            deriv = _derivative_1st_rho(y, rho, nu)
        else:
            deriv = _derivative_1st_nu(y, rho, nu)
        return deriv

    def element_dl2_dp2(
        self, y: np.ndarray, theta: Dict, param: int = 0, k: int = 0, clip=False
    ):
        rho, nu = self.theta_to_params(theta)

        if param == 0:
            deriv = _derivative_2nd_rho(y, rho, nu)
        else:
            deriv = _derivative_2nd_nu(y, rho, nu)
        return deriv

    def dl2_dpp(self, y: np.ndarray, theta: Dict, param: int = 0):
        raise NotImplementedError("Not implemented.")

    def initial_values(self, y, param=0):
        M = y.shape[0]
        if param == 0:  # rho
            tau = st.kendalltau(y[:, 0], y[:, 1]).correlation
            rho = np.full((M, 1), tau)
            return rho
        else:  # nu
            nu = np.full((M, 1), 10)  # default degrees of freedom
            return nu

    def cdf(self, y, theta):
        raise NotImplementedError("Not implemented")

    def ppf(self, q, theta):
        raise NotImplementedError("Not implemented")

    def rvs(self, size, theta):
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

        x = self.hinv(z1, z2, theta, un=2)

        return x

    def pdf(self, y, theta):
        return np.exp(self.logpdf(y, theta))

    def logcdf(self, y, theta):
        raise NotImplementedError("Not implemented")

    def logpdf(self, y, theta):
        rho, nu = self.theta_to_params(theta)
        return np.log(_log_likelihood_t(y, rho, nu))

    def logpmf(self, y, theta):
        raise NotImplementedError("Not implemented")

    def pmf(self, y, theta):
        raise NotImplementedError("Not implemented")

    def calculate_conditional_initial_values(
        self, y: np.ndarray, theta: Dict[int, np.ndarray]
    ) -> Dict[int, np.ndarray]:
        raise NotImplementedError("Not implemented")

    def hfunc(
        self, u: np.ndarray, v: np.ndarray, theta: Dict, un: int, family_code=2
    ) -> np.ndarray:
        """
        Conditional distribution function h(u|v) for the bivariate t copula.

        Args:
            u (np.ndarray): Array of shape (n,) with values in (0, 1).
            v (np.ndarray): Array of shape (n,) with values in (0, 1).
            theta (Dict): Parameters {0: rho, 1: nu}.
            un (int): Conditioning variable (1 or 2).

        Returns:
            np.ndarray: Array of shape (n,) with conditional probabilities.
        """

        rho, nu = self.theta_to_params(theta)

        # Apply clipping using masks
        u_mask_low = u < UMIN
        u_mask_high = u > UMAX
        v_mask_low = v < UMIN
        v_mask_high = v > UMAX

        u = np.where(u_mask_low, UMIN, u)
        u = np.where(u_mask_high, UMAX, u)
        v = np.where(v_mask_low, UMIN, v)
        v = np.where(v_mask_high, UMAX, v)

        # Swap u and v if un == 2
        if un == 1:
            u, v = v, u

        # Handle edge cases
        h = np.where((v == 0) | (u == 0), 0, np.nan)
        h = np.where(v == 1, u, h).reshape(-1, 1)

        qt_u = np.array([st.t.ppf(u[i], df=nu[i]) for i in range(len(u))]).reshape(
            -1, 1
        )
        qt_v = np.array([st.t.ppf(v[i], df=nu[i]) for i in range(len(v))]).reshape(
            -1, 1
        )

        denom = np.sqrt((nu + qt_v**2) * (1 - rho**2) / (nu + 1))
        x = (qt_u - rho * qt_v) / denom

        # Use masks to handle finite and infinite cases
        finite_mask = np.isfinite(x)
        neg_mask = ~finite_mask & ((qt_u - rho * qt_v) < 0)
        pos_mask = ~finite_mask & ((qt_u - rho * qt_v) >= 0)

        h = np.where(
            finite_mask,
            np.array([st.t.cdf(x[i, 0], df=nu[i]) for i in range(len(x))]).reshape(
                -1, 1
            ),
            h,
        )
        h = np.where(neg_mask, 0, h)
        h = np.where(pos_mask, 1, h)

        return h.squeeze()

    def hinv(
        self, u: np.ndarray, v: np.ndarray, theta: np.ndarray, un: int, family_code=2
    ) -> np.ndarray:
        """
        Inverse conditional distribution function h^(-1)(u|v) for the bivariate normal copula.

        Args:
            u (np.ndarray): Array of shape (n,) with values in (0, 1).
            v (np.ndarray): Array of shape (n,) with values in (0, 1).
            theta (np.ndarray or float): Correlation parameter(s), shape (n,) or scalar.
            un (int): Determines which conditional to compute.

        Returns:
            np.ndarray: Array of shape (n,) with inverse conditional probabilities.
        """

        # Apply clipping using masks
        u_mask_low = u < UMIN
        u_mask_high = u > UMAX
        v_mask_low = v < UMIN
        v_mask_high = v > UMAX

        u = np.where(u_mask_low, UMIN, u)
        u = np.where(u_mask_high, UMAX, u)
        v = np.where(v_mask_low, UMIN, v)
        v = np.where(v_mask_high, UMAX, v)

        rho, nu = self.theta_to_params(theta)

        qt_u = st.t.ppf(u, df=nu + 1.0).reshape(-1, 1)
        qt_v = st.t.ppf(v, df=nu).reshape(-1, 1)

        mu = rho * qt_v
        var = ((nu + qt_v**2) * (1.0 - rho**2)) / (nu + 1.0)
        hinv = st.t.cdf((np.sqrt(var) * qt_u + mu), df=nu).reshape(-1, 1)

        # Clip output for numerical stability
        # Ensure results are in [0,1] using masks

        h_mask_low = hinv < 0
        h_mask_high = hinv > 1
        hinv = np.where(h_mask_low, 0, hinv)
        hinv = np.where(h_mask_high, 1, hinv)

        return hinv.squeeze()

    def get_regularization_size(self, dim: int) -> int:
        return dim


##########################################################
### Functions for the Student-t copula derivatives #####
##########################################################


def stable_gamma_division(x1, x2):
    """Stable computation of gamma(x1)/gamma(x2)"""
    # Handle scalar inputs by converting to arrays

    x1_arr = np.asarray(x1, dtype=float)
    x2_arr = np.asarray(x2, dtype=float)

    # Broadcast to common shape
    x1_b, x2_b = np.broadcast_arrays(x1_arr, x2_arr)
    result = np.empty_like(x1_b, dtype=float)

    it = np.nditer(
        [x1_b, x2_b, result],
        flags=["multi_index"],
        op_flags=[["readonly"], ["readonly"], ["writeonly"]],
    )
    for xi, yi, out in it:
        x1_i = float(xi)
        x2_i = float(yi)

        a1 = math.fmod(max(x1_i, x2_i), 1.0)
        a2 = max(x1_i, x2_i) - a1
        b1 = math.fmod(min(x1_i, x2_i), 1.0)
        b2 = min(x1_i, x2_i) - b1

        s = 1.0
        if a1 == 0.0 and b1 == 0.0:
            i = 1
            while i < int(b2):
                s *= ((a1 + a2) - float(i)) / ((b1 + b2) - float(i))
                i += 1
            i = int(b2)
            while i < int(a2):
                s *= (a1 + a2) - float(i)
                i += 1
        elif a1 > 0.0 and b1 == 0.0:
            i = 1
            while i < int(b2):
                s *= ((a1 + a2) - float(i)) / ((b1 + b2) - float(i))
                i += 1
            i = int(b2)
            while i <= int(a2):
                s *= (a1 + a2) - float(i)
                i += 1
            s *= float(sp.gamma(a1))
        elif a1 == 0.0 and b1 > 0.0:
            i = 1
            while i <= int(b2):
                s *= ((a1 + a2) - float(i)) / ((b1 + b2) - float(i))
                i += 1
            i = int(b2) + 1
            while i < int(a2):
                s *= (a1 + a2) - float(i)
                i += 1
            s /= float(sp.gamma(b1))
        elif a1 > 0.0 and b1 > 0.0:
            i = 1
            while i <= int(b2):
                s *= ((a1 + a2) - float(i)) / ((b1 + b2) - float(i))
                i += 1
            i = int(b2) + 1
            while i <= int(a2):
                s *= (a1 + a2) - float(i)
                i += 1
            s *= float(sp.gamma(a1)) / float(sp.gamma(b1))

        if x2_i > x1_i:
            s = 1.0 / s

        out[...] = s

    result = result if result.shape != () else float(result)

    return result


def _log_likelihood_t(y, rho, nu):
    """Log-likelihood for bivariate t copula"""

    y_clipped = np.clip(y, UMIN, UMAX)

    t1 = st.t.ppf(y_clipped[:, 0], df=nu).reshape(-1, 1)
    t2 = st.t.ppf(y_clipped[:, 1], df=nu).reshape(-1, 1)

    # Bivariate t copula density (following C code structure)
    # f = StableGammaDivision((nu+2)/2, nu/2) / (nu*pi*sqrt(1-rho^2)*dt(t1,nu)*dt(t2,nu))
    #     * (1 + (t1^2 + t2^2 - 2*rho*t1*t2)/(nu*(1-rho^2)))^(-(nu+2)/2)

    # Calculate the gamma ratio using stable division
    gamma_ratio = stable_gamma_division((nu + 2.0) / 2.0, nu / 2.0)

    # Calculate t distribution PDFs (dt in C code)
    dt1 = st.t.pdf(t1, df=nu)
    dt2 = st.t.pdf(t2, df=nu)

    # Calculate the quadratic form in the exponent
    quad_form = (t1 * t1 + t2 * t2 - 2.0 * rho * t1 * t2) / (nu * (1 - rho**2))

    # Calculate the full density
    f = (
        gamma_ratio
        / (nu * np.pi * np.sqrt(1 - rho**2) * dt1 * dt2)
        * np.power(1.0 + quad_form, -(nu + 2.0) / 2.0)
    )

    f[f <= 0] = 1e-16

    return f.squeeze()


def _derivative_1st_rho(y, rho, nu):
    """First derivative wrt rho for t copula"""

    y_clipped = np.clip(y, UMIN, UMAX)

    t1 = st.t.ppf(y_clipped[:, 0], df=nu).reshape(-1, 1)
    t2 = st.t.ppf(y_clipped[:, 1], df=nu).reshape(-1, 1)

    t3 = -(nu + 2.0) / 2.0
    t10 = nu * (1.0 - rho * rho)
    t4 = -2.0 * t1 * t2 / t10
    t11 = t1 * t1 + t2 * t2 - 2.0 * rho * t1 * t2
    t5 = 2.0 * t11 * rho / t10 / (1.0 - rho * rho)
    t6 = 1.0 + (t11 / t10)
    t7 = rho / (1.0 - rho * rho)
    deriv = t3 * (t4 + t5) / t6 + t7

    return deriv.squeeze()


def _derivative_1st_rho_l(y, rho, nu):
    """First derivative wrt rho for t copula"""

    y_clipped = np.clip(y, UMIN, UMAX)

    c = _log_likelihood_t(y_clipped, rho, nu).reshape(-1, 1)
    t1 = st.t.ppf(y_clipped[:, 0], df=nu).reshape(-1, 1)
    t2 = st.t.ppf(y_clipped[:, 1], df=nu).reshape(-1, 1)

    # Calculate current likelihood
    t3 = -(nu + 2.0) / 2.0
    t10 = nu * (1.0 - rho * rho)
    t4 = -2.0 * t1 * t2 / t10
    t11 = t1 * t1 + t2 * t2 - 2.0 * rho * t1 * t2
    t5 = 2.0 * t11 * rho / t10 / (1.0 - rho * rho)
    t6 = 1.0 + (t11 / t10)
    t7 = rho / (1.0 - rho * rho)
    deriv = c * (t3 * (t4 + t5) / t6 + t7)

    return deriv.squeeze()


def _derivative_1st_nu(y, rho, nu):
    """First derivative wrt nu for t copula"""

    eps = np.finfo(float).eps
    y_clipped = np.clip(y, eps, 1 - eps)

    u = np.array([st.t.ppf(y_clipped[i, 0], df=nu[i]) for i in range(len(y))]).reshape(
        -1, 1
    )
    v = np.array([st.t.ppf(y_clipped[i, 1], df=nu[i]) for i in range(len(y))]).reshape(
        -1, 1
    )

    # Follow C code structure exactly
    t1 = sp.digamma((nu + 1.0) / 2.0)
    t2 = sp.digamma(nu / 2.0)
    t3 = 0.5 * np.log(1.0 - rho * rho)
    t4 = (nu - 2.0) / (2.0 * nu)
    t5 = 0.5 * np.log(nu)
    t6 = -t1 + t2 + t3 - t4 - t5
    t10 = (nu + 2.0) / 2.0

    x1 = u
    x2 = v

    # Derivatives of quantile function wrt nu
    out1 = _diff_quantile_nu(x1, nu)
    out2 = _diff_quantile_nu(x2, nu)

    t7 = 1.0 + 2.0 * x1 * out1
    t8 = 1.0 + 2.0 * x2 * out2
    t9 = (nu + 1.0) / 2.0 * (t7 / (nu + x1 * x1) + t8 / (nu + x2 * x2))

    M = nu * (1.0 - rho * rho) + x1 * x1 + x2 * x2 - 2.0 * rho * x1 * x2
    t11 = (
        1.0
        - rho * rho
        + 2.0 * x1 * out1
        + 2.0 * x2 * out2
        - 2.0 * rho * (x1 * out2 + x2 * out1)
    )
    t12 = 0.5 * np.log((nu + x1 * x1) * (nu + x2 * x2))
    t13 = 0.5 * np.log(M)

    deriv = t6 + t9 + t12 - t10 * t11 / M - t13

    return deriv.squeeze()


def _derivative_1st_nu_l(y, rho, nu):
    """First derivative wrt nu for t copula"""

    eps = np.finfo(float).eps
    y_clipped = np.clip(y, eps, 1 - eps)

    u = np.array([st.t.ppf(y_clipped[i, 0], df=nu[i]) for i in range(len(y))]).reshape(
        -1, 1
    )
    v = np.array([st.t.ppf(y_clipped[i, 1], df=nu[i]) for i in range(len(y))]).reshape(
        -1, 1
    )

    # Calculate current likelihood
    c = _log_likelihood_t(y_clipped, rho, nu).reshape(-1, 1)

    # Follow C code structure exactly
    t1 = sp.digamma((nu + 1.0) / 2.0)
    t2 = sp.digamma(nu / 2.0)
    t3 = 0.5 * np.log(1.0 - rho * rho)
    t4 = (nu - 2.0) / (2.0 * nu)
    t5 = 0.5 * np.log(nu)
    t6 = -t1 + t2 + t3 - t4 - t5
    t10 = (nu + 2.0) / 2.0

    x1 = u
    x2 = v

    # Derivatives of quantile function wrt nu
    out1 = _diff_quantile_nu(x1, nu)
    out2 = _diff_quantile_nu(x2, nu)

    t7 = 1.0 + 2.0 * x1 * out1
    t8 = 1.0 + 2.0 * x2 * out2
    t9 = (nu + 1.0) / 2.0 * (t7 / (nu + x1 * x1) + t8 / (nu + x2 * x2))

    M_val = nu * (1.0 - rho * rho) + x1 * x1 + x2 * x2 - 2.0 * rho * x1 * x2
    t11 = (
        1.0
        - rho * rho
        + 2.0 * x1 * out1
        + 2.0 * x2 * out2
        - 2.0 * rho * (x1 * out2 + x2 * out1)
    )
    t12 = 0.5 * np.log((nu + x1 * x1) * (nu + x2 * x2))
    t13 = 0.5 * np.log(M_val)

    deriv = c * (t6 + t9 + t12 - t10 * t11 / M_val - t13)

    return deriv.squeeze()


def _derivative_2nd_rho(y, rho, nu):
    """Second derivative wrt rho for t copula"""

    y_clipped = np.clip(y, UMIN, UMAX)

    u = np.array([st.t.ppf(y_clipped[i, 0], df=nu[i]) for i in range(len(y))]).reshape(
        -1, 1
    )
    v = np.array([st.t.ppf(y_clipped[i, 1], df=nu[i]) for i in range(len(y))]).reshape(
        -1, 1
    )

    # Calculate current likelihood
    c = _log_likelihood_t(y_clipped, rho, nu).reshape(-1, 1)
    c = np.exp(np.log(c))
    # Get first derivative
    diff = _derivative_1st_rho_l(y_clipped, rho, nu).reshape(-1, 1)

    t1 = u
    t2 = v
    t4 = 1.0 - rho * rho
    M_val = nu * t4 + t1 * t1 + t2 * t2 - 2.0 * rho * t1 * t2

    t3 = -(nu + 1.0) * (1.0 + rho * rho) / t4 / t4
    t5 = (nu + 2.0) * nu / M_val
    t6 = 2.0 * (nu + 2.0) * np.power(nu * rho + t1 * t2, 2.0) / M_val / M_val
    t7 = diff / c

    deriv = c * (t3 + t5 + t6 + t7 * t7)
    return deriv.squeeze()


def _derivative_2nd_nu(y, rho, nu):
    """Second derivative wrt nu for t copula"""

    eps = np.finfo(float).eps
    y_clipped = np.clip(y, eps, 1 - eps)

    u = np.array([st.t.ppf(y_clipped[i, 0], df=nu[i]) for i in range(len(y))]).reshape(
        -1, 1
    )
    v = np.array([st.t.ppf(y_clipped[i, 1], df=nu[i]) for i in range(len(y))]).reshape(
        -1, 1
    )

    # Calculate current likelihood
    c = _log_likelihood_t(y_clipped, rho, nu).reshape(-1, 1)
    c = np.exp(np.log(c))

    # Get first derivative
    diff_nu = _derivative_1st_nu_l(y_clipped, rho, nu).reshape(-1, 1)
    x1 = u
    x2 = v

    # Common terms
    t1 = (nu + 1.0) / 2.0
    t2 = nu / 2.0
    t23 = nu * nu
    t3 = 1.0 / t23
    t4 = 1.0 / (2.0 * nu)
    t5 = 0.5 * sp.polygamma(1, t1)  # trigamma
    t6 = 1.0 - rho * rho
    t9 = 0.5 * sp.polygamma(1, t2)  # trigamma
    t10 = -t5 + t9 - t3 - t4

    # Get derivatives of quantile functions
    out1 = _diff_quantile_nu(x1, nu)
    out2 = _diff_quantile_nu(x2, nu)

    M_val = nu * t6 + x1 * x1 + x2 * x2 - 2.0 * rho * x1 * x2

    t8 = x1 * out2 + out1 * x2
    M_nu = t6 + 2.0 * x1 * out1 + 2.0 * x2 * out2 - 2.0 * rho * t8

    t24 = x1 * x1
    t25 = x2 * x2

    t11 = 1.0 + 2.0 * x1 * out1
    t12 = nu + t24
    t13 = t11 / t12

    t14 = 1.0 + 2.0 * x2 * out2
    t15 = nu + t25
    t16 = t14 / t15

    # Second derivatives of quantile functions (simplified approximation)
    out3 = diff2_x_nu(x1, nu)  # Second derivative of quantile function wrt nu
    out4 = diff2_x_nu(x2, nu)  # Second derivative of quantile function wrt nu

    t17 = 2.0 * out1 * out1 + 2.0 * x1 * out3
    t18 = t17 / t12

    t19 = 2.0 * out2 * out2 + 2.0 * x2 * out4
    t20 = t19 / t15

    t21 = t13 * t13
    t22 = t16 * t16

    M_nu_nu = (
        2.0 * out1 * out1
        + 2.0 * x1 * out3
        + 2.0 * out2 * out2
        + 2.0 * x2 * out4
        - 4.0 * rho * out1 * out2
        - 2.0 * rho * (x2 * out3 + x1 * out4)
    )
    deriv = (
        c
        * (
            t10
            + 0.5 * (t13 + t16)
            + t1 * (t18 - t21 + t20 - t22)
            + 0.5 * t13
            + 0.5 * t16
            - M_nu / M_val
            - (nu / 2.0 + 1.0) * (M_nu_nu / M_val - M_nu * M_nu / M_val / M_val)
        )
        + diff_nu * diff_nu / c
    )

    return deriv.squeeze()


# Make st.beta a function (alias to scipy.special.beta) so your function works unchanged
try:
    import scipy.special as _spsp

    st.beta = _spsp.beta
except Exception:
    pass


def trigamma(x: float) -> float:
    return float(sp.polygamma(1, x))


# ---------------- Incomplete Beta helpers (direct translations) ----------------


def incompleBeta_an1_bn1_p(
    x: float, p: float, q: float
) -> Tuple[List[float], List[float]]:
    t2 = 1.0 / (1.0 - x)
    t3 = x * t2
    t4 = q - 1.0
    t5 = p + 1.0
    t9 = t5 * t5
    t19 = q * x * t2
    t20 = 2.0 * t19
    t21 = 4.0 * q
    t27 = p * q
    t28 = p - 2.0 - t19
    t31 = 1.0 / q
    t32 = (t20 + t21 + 2.0 * (t19 + 2.0 * q) * (p - 1.0) + t27 * t28) * t31
    t33 = 1.0 / p
    t34 = p + 2.0
    t35 = 1.0 / t34
    t36 = t33 * t35
    t40 = (t20 + t21 + q * t28 + t27) * t31
    t42 = p * p
    t43 = 1.0 / t42
    t44 = t43 * t35
    t46 = t34 * t34
    t47 = 1.0 / t46
    t48 = t33 * t47
    an0 = t3 * t4 / t5
    an1 = -t3 * t4 / t9
    an2 = 2.0 * t3 * t4 / t9 / t5
    bn0 = t32 * t36
    bn1 = t40 * t36 - t32 * t44 - t32 * t48
    bn2 = (
        2.0 * t36
        - 2.0 * t40 * t44
        - 2.0 * t40 * t48
        + 2.0 * t32 / t42 / p * t35
        + 2.0 * t32 * t43 * t47
        + 2.0 * t32 * t33 / t46 / t34
    )
    return [an0, an1, an2], [bn0, bn1, bn2]


def incompleBeta_an1_bn1_q(
    x: float, p: float, q: float
) -> Tuple[List[float], List[float]]:
    t2 = 1.0 / (1.0 - x)
    t3 = x * t2
    t6 = 1.0 / (p + 1.0)
    t11 = q * x * t2
    t16 = p - 1.0
    t19 = p * q
    t20 = p - 2.0 - t11
    t22 = 2.0 * t11 + 4.0 * q + 2.0 * (t11 + 2.0 * q) * t16 + t19 * t20
    t23 = 1.0 / q
    t27 = 1.0 / (p + 2.0)
    t28 = 1.0 / p * t27
    t36 = 2.0 * t3 + 4.0 + 2.0 * (t3 + 2.0) * t16 + p * t20 - t19 * t3
    t39 = q * q
    t40 = 1.0 / t39
    an0 = t3 * (q - 1.0) * t6
    an1 = t3 * t6
    an2 = 0.0
    bn0 = t22 * t23 * t28
    bn1 = t36 * t23 * t28 - t22 * t40 * t28
    bn2 = -2.0 * t3 * t23 * t27 - 2.0 * t36 * t40 * t28 + 2.0 * t22 / t39 / q * t28
    return [an0, an1, an2], [bn0, bn1, bn2]


def incompleBeta_an_bn_p(
    x: float, p: float, q: float, n: int
) -> Tuple[List[float], List[float]]:
    t1 = x * x
    t2 = 1.0 - x
    t3 = t2 * t2
    t5 = t1 / t3
    t6 = n - 1.0
    t9 = t5 * t6 * (p + q + n - 2.0)
    t10 = p + n - 1.0
    t11 = q - n
    t12 = t10 * t11
    t13 = 2.0 * n
    t14 = p + t13 - 3.0
    t15 = 1.0 / t14
    t16 = p + t13 - 2.0
    t17 = t16 * t16
    t18 = 1.0 / t17
    t19 = t15 * t18
    t20 = p + t13 - 1.0
    t21 = 1.0 / t20
    t26 = t5 * t6 * t10
    t27 = t11 * t15
    t28 = t18 * t21
    t29 = t27 * t28
    t32 = t14 * t14
    t33 = 1.0 / t32
    t34 = t33 * t18
    t39 = 1.0 / t17 / t16
    t40 = t15 * t39
    t45 = t20 * t20
    t46 = 1.0 / t45
    t55 = t11 * t33 * t28
    t59 = t27 * t39 * t21
    t63 = t27 * t18 * t46
    t88 = t17 * t17

    t105 = (
        2.0 * t5 * t6 * t29
        - 2.0 * t26 * t55
        - 4.0 * t26 * t59
        - 2.0 * t26 * t63
        - 2.0 * t9 * t55
        - 4.0 * t9 * t59
        - 2.0 * t9 * t63
        + 2.0 * t9 * t12 / t32 / t14 * t18 * t21
        + 4.0 * t9 * t12 * t33 * t39 * t21
        + 2.0 * t9 * t12 * t34 * t46
        + 6.0 * t9 * t12 * t15 / t88 * t21
        + 4.0 * t9 * t12 * t40 * t46
        + 2.0 * t9 * t12 * t19 / t45 / t20
    )

    t108 = q * x / t2
    t110 = t108 + 2.0 * q
    t111 = n * n
    t118 = p * q
    t119 = p - 2.0 - t108
    t122 = 1.0 / q
    t123 = (2.0 * t110 * t111 + 2.0 * t110 * (p - 1.0) * n + t118 * t119) * t122
    t124 = 1.0 / t16
    t125 = p + t13
    t126 = 1.0 / t125
    t127 = t124 * t126
    t133 = (2.0 * t110 * n + q * t119 + t118) * t122
    t135 = t18 * t126
    t137 = t125 * t125
    t138 = 1.0 / t137
    t139 = t124 * t138

    an0 = t9 * t12 * t19 * t21
    an1 = (
        t26 * t29
        + t9 * t29
        - t9 * t12 * t34 * t21
        - 2.0 * t9 * t12 * t40 * t21
        - t9 * t12 * t19 * t46
    )
    an2 = t105
    bn0 = t123 * t127
    bn1 = t133 * t127 - t123 * t135 - t123 * t139
    bn2 = (
        2.0 * t127
        - 2.0 * t133 * t135
        - 2.0 * t133 * t139
        + 2.0 * t123 * t39 * t126
        + 2.0 * t123 * t18 * t138
        + 2.0 * t123 * t124 / t137 / t125
    )
    return [an0, an1, an2], [bn0, bn1, bn2]


def incompleBeta_an_bn_q(
    x: float, p: float, q: float, n: int
) -> Tuple[List[float], List[float]]:
    t1 = x * x
    t2 = 1.0 - x
    t3 = t2 * t2
    t5 = t1 / t3
    t6 = n - 1.0
    t9 = t5 * t6 * (p + q + n - 2.0)
    t10 = p + n - 1.0
    t11 = q - n
    t13 = 2.0 * n
    t15 = 1.0 / (p + t13 - 3.0)
    t16 = p + t13 - 2.0
    t17 = t16 * t16
    t18 = 1.0 / t17
    t21 = 1.0 / (p + t13 - 1.0)
    t28 = t18 * t21
    t32 = t10 * t15 * t28
    t39 = 1.0 / t2
    t40 = q * x * t39
    t42 = t40 + 2.0 * q
    t43 = n * n
    t46 = p - 1.0
    t50 = p * q
    t51 = p - 2.0 - t40
    t53 = 2.0 * t42 * t43 + 2.0 * t42 * t46 * n + t50 * t51
    t54 = 1.0 / q
    t56 = 1.0 / t16
    t58 = 1.0 / (p + t13)
    t59 = t56 * t58
    t61 = x * t39
    t62 = t61 + 2.0
    t70 = 2.0 * t62 * t43 + 2.0 * t62 * t46 * n + p * t51 - t50 * t61
    t73 = q * q
    t74 = 1.0 / t73

    an0 = t9 * t10 * t11 * t15 * t18 * t21
    an1 = t5 * t6 * t10 * t11 * t15 * t28 + t9 * t32
    an2 = 2.0 * t5 * t6 * t32
    bn0 = t53 * t54 * t59
    bn1 = t70 * t54 * t59 - t53 * t74 * t59
    bn2 = (
        -2.0 * p * x * t39 * t54 * t56 * t58
        - 2.0 * t70 * t74 * t59
        + 2.0 * t53 / t73 / q * t59
    )
    return [an0, an1, an2], [bn0, bn1, bn2]


# ------------------------------ inbeder ------------------------------


def inbeder(
    x_in: float,
    p_in: float,
    q_in: float,
    err: float = 1e-12,
    minappx: int = 3,
    maxappx: int = 200,
) -> Tuple[float, float, float]:
    """
    Incomplete Beta I(x; p, q) and derivatives wrt p (first, second).
    Returns (I, dI/dp, d²I/dp²).
    """
    EPS = 1e-12
    if x_in > p_in / (p_in + q_in):
        x = 1.0 - x_in
        p = q_in
        q = p_in
        flipped = True
    else:
        x = x_in
        p = p_in
        q = q_in
        flipped = False

    lbet = sp.betaln(p, q)
    pa = sp.digamma(p)
    pa1 = trigamma(p)
    pb = sp.digamma(q)
    pb1 = trigamma(q)
    pab = sp.digamma(p + q)
    pab1 = trigamma(p + q)
    x = max(EPS, min(1.0 - EPS, x))
    p = max(EPS, p)
    omx = 1.0 - x
    logx = log(x)
    logomx = log(omx)

    c = [0.0, 0.0, 0.0]
    c[0] = p * logx + (q - 1.0) * logomx - lbet - log(p)
    c0 = np.exp(c[0])
    if flipped:
        c[1] = logomx - pb + pab
        c[2] = c[1] * c[1] - pb1 + pab1
    else:
        c[1] = logx - 1.0 / p - pa + pab
        c[2] = c[1] * c[1] + 1.0 / (p * p) - pa1 + pab1

    an = [0.0, 0.0, 0.0]
    bn = [0.0, 0.0, 0.0]
    an1 = [1.0, 0.0, 0.0]
    an2 = [1.0, 0.0, 0.0]
    bn1 = [1.0, 0.0, 0.0]
    bn2 = [0.0, 0.0, 0.0]
    der_old = [0.0, 0.0, 0.0]
    dan = [0.0, 0.0, 0.0]
    dbn = [0.0, 0.0, 0.0]
    dr = [0.0, 0.0, 0.0]
    d1 = [0.0, 0.0, 0.0]

    n = 0
    der = [0.0, 0.0, 0.0]

    while True:
        n += 1
        if n == 1:
            if flipped:
                an[:], bn[:] = incompleBeta_an1_bn1_q(x, p, q)
            else:
                an[:], bn[:] = incompleBeta_an1_bn1_p(x, p, q)
        else:
            if flipped:
                an[:], bn[:] = incompleBeta_an_bn_q(x, p, q, n)
            else:
                an[:], bn[:] = incompleBeta_an_bn_p(x, p, q, n)

        dan[0] = an[0] * an2[0] + bn[0] * an1[0]
        dbn[0] = an[0] * bn2[0] + bn[0] * bn1[0]
        dan[1] = an[1] * an2[0] + an[0] * an2[1] + bn[1] * an1[0] + bn[0] * an1[1]
        dbn[1] = an[1] * bn2[0] + an[0] * bn2[1] + bn[1] * bn1[0] + bn[0] * bn1[1]
        dan[2] = (
            an[2] * an2[0]
            + 2 * an[1] * an2[1]
            + an[0] * an2[2]
            + bn[2] * an1[0]
            + 2 * bn[1] * an1[1]
            + bn[0] * an1[2]
        )
        dbn[2] = (
            an[2] * bn2[0]
            + 2 * an[1] * bn2[1]
            + an[0] * bn2[2]
            + bn[2] * bn1[0]
            + 2 * bn[1] * bn1[1]
            + bn[0] * bn1[2]
        )

        Rn = dan[0]
        if fabs(dbn[0]) > fabs(dan[0]):
            Rn = dbn[0]
        for i in range(3):
            an1[i] /= Rn
            bn1[i] /= Rn
        dan[1] /= Rn
        dan[2] /= Rn
        dbn[1] /= Rn
        dbn[2] /= Rn

        if fabs(dbn[0]) > fabs(dan[0]):
            dan[0] = dan[0] / dbn[0]
            dbn[0] = 1.0
        else:
            dbn[0] = dbn[0] / dan[0]
            dan[0] = 1.0

        dr[0] = dan[0] / dbn[0]
        Rn = dr[0]
        dbn0_sq = dbn[0] * dbn[0]
        dr[1] = (dan[1] - Rn * dbn[1]) / dbn[0]
        dr[2] = (-2 * dan[1] * dbn[1] + 2 * Rn * dbn[1] * dbn[1]) / dbn0_sq + (
            dan[2] - Rn * dbn[2]
        ) / dbn[0]

        an2[:] = an1[:]
        an1[:] = dan[:]
        bn2[:] = bn1[:]
        bn1[:] = dbn[:]

        pr = 0.0
        if dr[0] > 0.0:
            pr = exp(c[0] + log(dr[0]))
        der[0] = pr
        der[1] = pr * c[1] + c0 * dr[1]
        der[2] = pr * c[2] + 2 * c0 * c[1] * dr[1] + c0 * dr[2]

        for i in range(3):
            denom = max(err, fabs(der[i]))
            d1[i] = fabs(der_old[i] - der[i]) / denom
            der_old[i] = der[i]
        d = max(d1)

        if n < minappx:
            d = 1.0
        if n >= maxappx:
            d = 0.0

        if d <= err:
            break

    if flipped:
        der[0] = 1.0 - der[0]
        der[1] = -der[1]
        der[2] = -der[2]

    return tuple(der)


# ------------------------------ Derivative helpers ------------------------------
def diff_t_nu_nu(x, nu):
    """
    Vectorized translation of C diff_t_nu_nu.
    Accepts scalar or array-like x and nu. Returns array broadcast to common shape.
    """
    x_arr = np.asarray(x, dtype=float)
    nu_arr = np.asarray(nu, dtype=float)
    x_b, nu_b = np.broadcast_arrays(x_arr, nu_arr)

    # Numerics
    nu_eff = np.clip(nu_b, 1e-6, np.inf)
    abs_x = np.abs(x_b)
    sign = np.where(x_b < 0.0, -1.0, 1.0)

    xmax = nu_eff / (nu_eff + abs_x * abs_x)
    t1 = 1.0 / (abs_x * abs_x + nu_eff)  # 1 / (x^2 + nu)
    t2 = 0.5 * nu_eff  # nu/2
    t4 = 0.5 * (nu_eff + 1.0)  # (nu+1)/2

    # Compute Ipp from inbeder (scalar routine) only where needed
    Ipp = np.empty_like(x_b, dtype=float)
    flat_xmax = np.ravel(xmax)
    flat_t2 = np.ravel(t2)
    flat_Ipp = np.empty_like(flat_xmax, dtype=float)
    for i in range(flat_xmax.size):
        # inbeder returns (I, dI/dp, d2I/dp2); we need the third
        flat_Ipp[i] = inbeder(float(flat_xmax[i]), float(flat_t2[i]), 0.5)[2]
    Ipp[...] = flat_Ipp.reshape(x_b.shape)

    # Remaining terms (vectorized)
    t5 = np.power(nu_eff, (nu_eff / 2.0) - 1.0) * abs_x
    t6 = np.power(t1, t4)
    t7 = sp.beta(t2, 0.5)
    t8 = t5 * t6
    t9 = nu_eff * t1

    t11 = sp.digamma(0.5 * nu_eff)
    t12 = sp.digamma(0.5 * nu_eff + 0.5)
    t13 = t11 - t12
    t14 = 1.0 / t7

    t10 = -t1 * t4 + (t2 - 1.0) / nu_eff + 0.5 * np.log(t1) + 0.5 * np.log(nu_eff)

    out = -0.125 * Ipp + t8 * t14 * (-0.25 * np.log(t9) + 0.5 * t13 - 0.5 * t10)
    out *= sign

    if out.size == 1:
        return float(out)
    return out


def diff_dt_nu(x, nu):
    """
    Vectorized translation of C diff_dt_nu.
    """
    x_b, nu_b = np.broadcast_arrays(np.asarray(x, float), np.asarray(nu, float))
    nu_eff = np.clip(nu_b, 1e-6, np.inf)

    t1 = (nu_eff + 1.0) / 2.0
    t2 = sp.digamma(t1)
    t3 = sp.beta(0.5 * nu_eff, 0.5)
    t4 = np.sqrt(nu_eff)
    t6 = sp.digamma(0.5 * nu_eff)

    t10 = -0.5 / t3 / t4 * (t6 - t2 + 1.0 / nu_eff)
    t11 = 1.0 + (x_b * x_b) / nu_eff
    t13 = np.power(t11, -t1)
    t14 = 1.0 / t3 / t4
    t15 = np.log(t11)
    t16 = -t1 * x_b * x_b / (nu_eff * nu_eff) / t11

    out = t10 * t13 + t14 * (t13 * (-0.5 * t15 - t16))
    if out.size == 1:
        return float(out)
    return out


def diff_dt_x(x, nu):
    """
    Vectorized translation of C diff_dt_x.
    """
    x_b, nu_b = np.broadcast_arrays(np.asarray(x, float), np.asarray(nu, float))
    nu_eff = np.clip(nu_b, 1e-6, np.inf)

    t2 = (nu_eff + 1.0) / nu_eff
    t3 = np.sqrt(nu_eff)
    t4 = 1.0 / (t3 * sp.beta(0.5 * nu_eff, 0.5))
    t5 = 1.0 + (x_b * x_b) / nu_eff
    t6 = (nu_eff + 3.0) / 2.0
    t7 = np.power(t5, -t6)
    out = -t4 * t2 * x_b * t7

    if out.size == 1:
        return float(out)
    return out


def _diff_quantile_nu(x, nu):
    """
    Derivative of t quantile function wrt degrees of freedom (vectorized).
    Accepts scalar or array-like x and nu. Returns array broadcast to the common shape.
    """
    # Broadcast inputs
    x_arr = np.asarray(x, dtype=float)
    nu_arr = np.asarray(nu, dtype=float)
    x_b, nu_b = np.broadcast_arrays(x_arr, nu_arr)

    # Prepare output
    out = np.zeros_like(x_b, dtype=float)

    # Masks and numerics
    EPS_X = 1e-14
    TINY = 1e-300

    abs_x = np.abs(x_b)
    sign = np.where(x_b < 0.0, -1.0, 1.0)
    mask = abs_x > EPS_X  # where derivative is not trivially 0

    if not np.any(mask):
        return float(0.0) if out.size == 1 else out

    # Clamp nu for stability
    nu_eff = np.maximum(nu_b, 1e-8)

    # Common terms (vectorized)
    t_pdf = st.t.pdf(abs_x, df=nu_eff)
    t_pdf = np.maximum(t_pdf, TINY)  # avoid division by zero

    t2 = 0.5 * nu_eff
    t4 = 0.5 * (nu_eff + 1.0)
    denom = abs_x * abs_x + nu_eff
    t6 = np.power(1.0 / denom, t4)
    t5 = np.power(nu_eff, t2 - 1.0) * abs_x
    t7 = sp.beta(t2, 0.5)

    xmax = nu_eff / (nu_eff + abs_x * abs_x)

    # Compute dI/dp via inbeder for masked positions
    idxs = np.flatnonzero(mask)
    dIdp_vals = np.empty(idxs.size, dtype=float)
    for i, j in enumerate(idxs):
        dIdp_vals[i] = inbeder(float(xmax.flat[j]), float(t2.flat[j]), 0.5)[1]

    dIdp = np.zeros_like(x_b, dtype=float)
    dIdp.flat[idxs] = dIdp_vals

    # Final result
    res = 0.5 / t_pdf * (0.5 * dIdp + (t5 * t6) / t7)
    res *= sign
    out[mask] = res[mask]

    if out.size == 1:
        return float(out)
    return out


def diff2_x_nu(x, nu):
    """
    Vectorized translation of C diff2_x_nu:
        out = (-t5*t4^2 - t2 - 2*t3*t4) / t1
    where:
        t1 = t.pdf(x; nu)
        t2 = diff_t_nu_nu(x, nu)
        t3 = diff_dt_nu(x, nu)
        t4 = _diff_quantile_nu(x, nu)
        t5 = diff_dt_x(x, nu)
    """
    x_b, nu_b = np.broadcast_arrays(np.asarray(x, float), np.asarray(nu, float))

    t1 = st.t.pdf(x_b, df=nu_b)
    TINY = 1e-300
    denom = np.maximum(t1, TINY)  # avoid division by zero in a vectorized way

    t2 = diff_t_nu_nu(x_b, nu_b)
    t3 = diff_dt_nu(x_b, nu_b)
    t4 = _diff_quantile_nu(x_b, nu_b)
    t5 = diff_dt_x(x_b, nu_b)

    out = (-t5 * t4 * t4 - t2 - 2.0 * t3 * t4) / denom
    if out.size == 1:
        return float(out)
    return out
