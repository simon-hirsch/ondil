# Author Simon Hirsch
# MIT Licence
from typing import Dict

import numpy as np
import scipy.special as sp
import scipy.stats as st

from ..base import BivariateCopulaMixin, CopulaMixin, Distribution, LinkFunction
from ..links import FisherZLink, KendallsTauToParameter, LogShiftTwo
from ..types import ParameterShapes


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
    ):
        super().__init__(
            links={0: link_1, 1: link_2},
            param_links={0: param_link_1, 1: param_link_2},
        )
        self.is_multivariate = True
        self._regularization_allowed = {0: False, 1: False}

    @staticmethod
    def fitted_elements(dim: int):
        return {0: 1, 1: 1}

    def theta_to_params(self, theta):
        if len(theta) > 1:
            return theta[0], theta[1]  # rho, nu
        else:
            return theta[0], 4.0  # default nu = 4

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
        Generate random samples from the bivariate t copula.

        Args:
            size (int): Number of samples to generate.
            theta (dict or tuple): (rho, nu) parameters.

        Returns:
            np.ndarray: Samples of shape (size, 2) in (0, 1).
        """
        rho, nu = self.theta_to_params(theta)

        # Generate from bivariate t distribution
        z = np.random.multivariate_normal([0, 0], [[1, rho], [rho, 1]], size=size)
        w = np.random.gamma(nu / 2, 2 / nu, size=size)

        # Scale by chi-squared random variable
        t_samples = z / np.sqrt(w[:, np.newaxis])

        # Transform to uniform marginals using the t CDF
        u = st.t.cdf(t_samples[:, 0], df=nu)
        v = st.t.cdf(t_samples[:, 1], df=nu)

        return np.column_stack((u, v))

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

    def hfunc(self, u: np.ndarray, v: np.ndarray, theta: Dict, un: int) -> np.ndarray:
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
        M = u.shape[0]
        UMIN = 1e-12
        UMAX = 1 - 1e-12

        u = np.clip(u, UMIN, UMAX)
        v = np.clip(v, UMIN, UMAX)

        # Swap u and v if un == 2
        if un == 2:
            u, v = v, u

        # Handle edge cases
        h = np.where((v == 0) | (u == 0), 0, np.nan)
        h = np.where(v == 1, u, h)

        qt_u = st.t.ppf(u, df=nu)
        qt_v = st.t.ppf(v, df=nu)

        for m in range(M):
            rho_m = rho[m] if hasattr(rho, "__len__") else rho
            nu_m = nu[m] if hasattr(nu, "__len__") else nu

            denom = np.sqrt((nu_m + qt_v[m] ** 2) * (1 - rho_m**2) / (nu_m + 1))
            x = (qt_u[m] - rho_m * qt_v[m]) / denom

            if np.isfinite(x):
                h[m] = st.t.cdf(x, df=nu_m + 1)
            elif (qt_u[m] - rho_m * qt_v[m]) < 0:
                h[m] = 0
            else:
                h[m] = 1

        # Clip output for numerical stability
        h = np.clip(h, UMIN, UMAX)
        return h

    def get_regularization_size(self, dim: int) -> int:
        return dim


##########################################################
### Functions for the Student-t copula derivatives #####
##########################################################


def stable_gamma_division(x1, x2):
    """Stable computation of gamma(x1)/gamma(x2)"""
    if x1 <= 0 or x2 <= 0:
        return float(sp.gamma(x1)) / float(sp.gamma(x2))

    a1 = max(x1, x2) % 1.0
    a2 = max(x1, x2) - a1
    b1 = min(x1, x2) % 1.0
    b2 = min(x1, x2) - b1
    sum_val = 1.0

    if a1 == 0.0 and b1 == 0.0:
        for i in range(1, int(b2)):
            sum_val *= ((a1 + a2) - i) / ((b1 + b2) - i)
        for i in range(int(b2), int(a2)):
            sum_val *= (a1 + a2) - i
    elif a1 > 0.0 and b1 == 0.0:
        for i in range(1, int(b2)):
            sum_val *= ((a1 + a2) - i) / ((b1 + b2) - i)
        for i in range(int(b2), int(a2) + 1):  # Fixed: should be <= in C
            sum_val *= (a1 + a2) - i
        sum_val *= float(sp.gamma(a1))
    elif a1 == 0.0 and b1 > 0.0:
        for i in range(1, int(b2) + 1):
            sum_val *= ((a1 + a2) - i) / ((b1 + b2) - i)
        for i in range(int(b2) + 1, int(a2)):
            sum_val *= (a1 + a2) - i
        sum_val /= float(sp.gamma(b1))
    elif a1 > 0.0 and b1 > 0.0:
        for i in range(1, int(b2) + 1):
            sum_val *= ((a1 + a2) - i) / ((b1 + b2) - i)
        for i in range(int(b2) + 1, int(a2) + 1):
            sum_val *= (a1 + a2) - i
        sum_val *= float(sp.gamma(a1)) / float(sp.gamma(b1))

    if x2 > x1:
        sum_val = 1.0 / sum_val
    return sum_val


def _log_likelihood_t(y, rho, nu):
    """Log-likelihood for bivariate t copula"""
    M = y.shape[0]
    f = np.empty(M)
    UMIN = 1e-12
    UMAX = 1 - 1e-12
    y_clipped = np.clip(y, UMIN, UMAX)
    u = np.array(
        [
            st.t.ppf(
                y_clipped[i, 0], df=nu[i] if hasattr(nu, "__len__") else nu
            ).squeeze()
            for i in range(M)
        ]
    )
    v = np.array(
        [
            st.t.ppf(
                y_clipped[i, 1], df=nu[i] if hasattr(nu, "__len__") else nu
            ).squeeze()
            for i in range(M)
        ]
    )

    for m in range(M):
        rho_m = rho[m] if hasattr(rho, "__len__") else rho
        nu_m = nu[m] if hasattr(nu, "__len__") else nu

        t1 = u[m]
        t2 = v[m]

        # Bivariate t copula density (following C code structure)
        # f = StableGammaDivision((nu+2)/2, nu/2) / (nu*pi*sqrt(1-rho^2)*dt(t1,nu)*dt(t2,nu))
        #     * (1 + (t1^2 + t2^2 - 2*rho*t1*t2)/(nu*(1-rho^2)))^(-(nu+2)/2)

        # Calculate the gamma ratio using stable division
        gamma_ratio = stable_gamma_division((nu_m + 2.0) / 2.0, nu_m / 2.0)

        # Calculate t distribution PDFs (dt in C code)
        dt1 = st.t.pdf(t1, df=nu_m)
        dt2 = st.t.pdf(t2, df=nu_m)

        # Calculate the quadratic form in the exponent
        quad_form = (t1 * t1 + t2 * t2 - 2.0 * rho_m * t1 * t2) / (
            nu_m * (1 - rho_m**2)
        )

        # Calculate the full density
        f[m] = (
            gamma_ratio
            / (nu_m * np.pi * np.sqrt(1 - rho_m**2) * dt1 * dt2)
            * np.power(1.0 + quad_form, -(nu_m + 2.0) / 2.0)
        )

    f[f <= 0] = 1e-16
    return f


def _derivative_1st_rho(y, rho, nu):
    """First derivative wrt rho for t copula"""
    M = y.shape[0]
    deriv = np.empty((M, 1), dtype=np.float64)
    UMIN = 1e-12
    UMAX = 1 - 1e-12
    y_clipped = np.clip(y, UMIN, UMAX)
    u = np.array(
        [
            st.t.ppf(
                y_clipped[i, 0], df=nu[i] if hasattr(nu, "__len__") else nu
            ).squeeze()
            for i in range(M)
        ]
    )
    v = np.array(
        [
            st.t.ppf(
                y_clipped[i, 1], df=nu[i] if hasattr(nu, "__len__") else nu
            ).squeeze()
            for i in range(M)
        ]
    )

    for m in range(M):
        rho_m = rho[m] if hasattr(rho, "__len__") else rho
        nu_m = nu[m] if hasattr(nu, "__len__") else nu

        # Calculate current likelihood
        # c = _log_likelihood_t(y_clipped[m:m+1], rho_m, nu_m)
        t1 = u[m]
        t2 = v[m]
        t3 = -(nu_m + 2.0) / 2.0
        t10 = nu_m * (1.0 - rho_m * rho_m)
        t4 = -2.0 * t1 * t2 / t10
        t11 = t1 * t1 + t2 * t2 - 2.0 * rho_m * t1 * t2
        t5 = 2.0 * t11 * rho_m / t10 / (1.0 - rho_m * rho_m)
        t6 = 1.0 + (t11 / t10)
        t7 = rho_m / (1.0 - rho_m * rho_m)
        deriv[m] = t3 * (t4 + t5) / t6 + t7

    return deriv.squeeze()


def _derivative_1st_rho_l(y, rho, nu):
    """First derivative wrt rho for t copula"""
    M = y.shape[0]
    deriv = np.empty((M, 1), dtype=np.float64)
    UMIN = 1e-12
    UMAX = 1 - 1e-12
    y_clipped = np.clip(y, UMIN, UMAX)
    u = np.array(
        [
            st.t.ppf(
                y_clipped[i, 0], df=nu[i] if hasattr(nu, "__len__") else nu
            ).squeeze()
            for i in range(M)
        ]
    )
    v = np.array(
        [
            st.t.ppf(
                y_clipped[i, 1], df=nu[i] if hasattr(nu, "__len__") else nu
            ).squeeze()
            for i in range(M)
        ]
    )

    for m in range(M):
        rho_m = rho[m] if hasattr(rho, "__len__") else rho
        nu_m = nu[m] if hasattr(nu, "__len__") else nu

        # Calculate current likelihood
        c = _log_likelihood_t(y_clipped[m : m + 1], rho_m, nu_m)
        t1 = u[m]
        t2 = v[m]
        t3 = -(nu_m + 2.0) / 2.0
        t10 = nu_m * (1.0 - rho_m * rho_m)
        t4 = -2.0 * t1 * t2 / t10
        t11 = t1 * t1 + t2 * t2 - 2.0 * rho_m * t1 * t2
        t5 = 2.0 * t11 * rho_m / t10 / (1.0 - rho_m * rho_m)
        t6 = 1.0 + (t11 / t10)
        t7 = rho_m / (1.0 - rho_m * rho_m)
        deriv[m] = c * (t3 * (t4 + t5) / t6 + t7)

    return deriv.squeeze()


def _derivative_1st_nu(y, rho, nu):
    """First derivative wrt nu for t copula"""
    M = y.shape[0]
    deriv = np.empty((M, 1), dtype=np.float64)
    eps = np.finfo(float).eps
    y_clipped = np.clip(y, eps, 1 - eps)

    u = np.array(
        [
            st.t.ppf(
                y_clipped[i, 0], df=nu[i] if hasattr(nu, "__len__") else nu
            ).squeeze()
            for i in range(M)
        ]
    )
    v = np.array(
        [
            st.t.ppf(
                y_clipped[i, 1], df=nu[i] if hasattr(nu, "__len__") else nu
            ).squeeze()
            for i in range(M)
        ]
    )

    for m in range(M):
        rho_m = rho[m] if hasattr(rho, "__len__") else rho
        nu_m = nu[m] if hasattr(nu, "__len__") else nu

        # Follow C code structure exactly
        t1 = sp.digamma((nu_m + 1.0) / 2.0)
        t2 = sp.digamma(nu_m / 2.0)
        t3 = 0.5 * np.log(1.0 - rho_m * rho_m)
        t4 = (nu_m - 2.0) / (2.0 * nu_m)
        t5 = 0.5 * np.log(nu_m)
        t6 = -t1 + t2 + t3 - t4 - t5
        t10 = (nu_m + 2.0) / 2.0

        x1 = u[m]
        x2 = v[m]

        # Derivatives of quantile function wrt nu
        out1 = _diff_quantile_nu(x1, nu_m)
        out2 = _diff_quantile_nu(x2, nu_m)

        t7 = 1.0 + 2.0 * x1 * out1
        t8 = 1.0 + 2.0 * x2 * out2
        t9 = (nu_m + 1.0) / 2.0 * (t7 / (nu_m + x1 * x1) + t8 / (nu_m + x2 * x2))

        M = nu_m * (1.0 - rho_m * rho_m) + x1 * x1 + x2 * x2 - 2.0 * rho_m * x1 * x2
        t11 = (
            1.0
            - rho_m * rho_m
            + 2.0 * x1 * out1
            + 2.0 * x2 * out2
            - 2.0 * rho_m * (x1 * out2 + x2 * out1)
        )
        t12 = 0.5 * np.log((nu_m + x1 * x1) * (nu_m + x2 * x2))
        t13 = 0.5 * np.log(M)

        deriv[m] = t6 + t9 + t12 - t10 * t11 / M - t13
    return deriv.squeeze()


def _derivative_1st_nu_l(y, rho, nu):
    """First derivative wrt nu for t copula"""
    M = y.shape[0]
    deriv = np.empty((M, 1), dtype=np.float64)
    eps = np.finfo(float).eps
    y_clipped = np.clip(y, eps, 1 - eps)

    u = np.array(
        [
            st.t.ppf(
                y_clipped[i, 0], df=nu[i] if hasattr(nu, "__len__") else nu
            ).squeeze()
            for i in range(M)
        ]
    )
    v = np.array(
        [
            st.t.ppf(
                y_clipped[i, 1], df=nu[i] if hasattr(nu, "__len__") else nu
            ).squeeze()
            for i in range(M)
        ]
    )

    for m in range(M):
        rho_m = rho[m] if hasattr(rho, "__len__") else rho
        nu_m = nu[m] if hasattr(nu, "__len__") else nu

        # Calculate current likelihood
        c = _log_likelihood_t(y_clipped[m : m + 1], rho_m, nu_m)

        # Follow C code structure exactly
        t1 = sp.digamma((nu_m + 1.0) / 2.0)
        t2 = sp.digamma(nu_m / 2.0)
        t3 = 0.5 * np.log(1.0 - rho_m * rho_m)
        t4 = (nu_m - 2.0) / (2.0 * nu_m)
        t5 = 0.5 * np.log(nu_m)
        t6 = -t1 + t2 + t3 - t4 - t5
        t10 = (nu_m + 2.0) / 2.0

        x1 = u[m]
        x2 = v[m]

        # Derivatives of quantile function wrt nu
        out1 = _diff_quantile_nu(x1, nu_m)
        out2 = _diff_quantile_nu(x2, nu_m)

        t7 = 1.0 + 2.0 * x1 * out1
        t8 = 1.0 + 2.0 * x2 * out2
        t9 = (nu_m + 1.0) / 2.0 * (t7 / (nu_m + x1 * x1) + t8 / (nu_m + x2 * x2))

        M_val = nu_m * (1.0 - rho_m * rho_m) + x1 * x1 + x2 * x2 - 2.0 * rho_m * x1 * x2
        t11 = (
            1.0
            - rho_m * rho_m
            + 2.0 * x1 * out1
            + 2.0 * x2 * out2
            - 2.0 * rho_m * (x1 * out2 + x2 * out1)
        )
        t12 = 0.5 * np.log((nu_m + x1 * x1) * (nu_m + x2 * x2))
        t13 = 0.5 * np.log(M_val)

        deriv[m] = c * (t6 + t9 + t12 - t10 * t11 / M_val - t13)
    return deriv.squeeze()


# from autograd import grad
# import autograd.numpy as anp
# import numpy as np  # only for final .squeeze() shape, optional

# def _derivative_1st_nu(y, rho, nu):
#     """First derivative wrt nu for t copula using autograd.

#     Notes
#     -----
#     - `_log_likelihood_t(y_slice, rho_scalar, nu_scalar)` must be autograd-compatible.
#     - If `nu` or `rho` are vectors, they are used row-wise (one value per observation).
#     - Returns shape (M,) via .squeeze(), matching the original behavior.
#     """
#     y = anp.asarray(y)
#     M = y.shape[0]
#     deriv = anp.empty((M, 1))

#     rho_is_vec = hasattr(rho, "__len__")
#     nu_is_vec  = hasattr(nu, "__len__")

#     for m in range(M):
#         rho_m = rho[m] if rho_is_vec else rho
#         nu_m  = nu[m]  if nu_is_vec  else nu

#         # Define a scalar function of nu to differentiate
#         def scalar_loglik(nu_scalar):
#             # keep df positive for stability (soft clip)
#             nu_pos = anp.maximum(nu_scalar, 1e-8)
#             ll = _log_likelihood_t(y[m:m+1], rho_m, nu_pos)
#             ll = anp.asarray(ll)
#             # if ll is an array, sum to get a scalar
#             return anp.log(ll).sum()

#         d_ll_d_nu = grad(scalar_loglik)
#         deriv[m, 0] = d_ll_d_nu(nu_m)

#     return np.asarray(deriv).squeeze()


# def _derivative_1st_nu(y, rho, nu):
#     """First derivative wrt nu for t copula using numerical differentiation"""
#     M = y.shape[0]
#     deriv = np.empty((M, 1), dtype=np.float64)
#     eps = 1e-8  # Small epsilon for numerical differentiation

#     for m in range(M):
#         rho_m = rho[m] if hasattr(rho, '__len__') else rho
#         nu_m = nu[m] if hasattr(nu, '__len__') else nu

#         # Create parameter dictionaries for nu +/- epsilon
#         nu1 = nu_m - eps
#         nu2 = nu_m + eps

#         # Calculate log-likelihood at nu +/- epsilon
#         ll1 = np.log(_log_likelihood_t(y[m:m+1], rho_m, nu1))
#         ll2 = np.log(_log_likelihood_t(y[m:m+1], rho_m, nu2))

#         # Numerical derivative: (f(x+h) - f(x-h)) / (2*h)
#         deriv[m] = (ll2 - ll1) / (2.0 * eps)

#     return deriv.squeeze()


def _derivative_2nd_rho(y, rho, nu):
    """Second derivative wrt rho for t copula"""
    M = y.shape[0]
    deriv = np.empty((M, 1), dtype=np.float64)
    UMIN = 1e-12
    UMAX = 1 - 1e-12
    y_clipped = np.clip(y, UMIN, UMAX)

    u = np.array(
        [
            st.t.ppf(
                y_clipped[i, 0], df=nu[i] if hasattr(nu, "__len__") else nu
            ).squeeze()
            for i in range(M)
        ]
    )
    v = np.array(
        [
            st.t.ppf(
                y_clipped[i, 1], df=nu[i] if hasattr(nu, "__len__") else nu
            ).squeeze()
            for i in range(M)
        ]
    )

    for m in range(M):
        rho_m = rho[m] if hasattr(rho, "__len__") else rho
        nu_m = nu[m] if hasattr(nu, "__len__") else nu

        # Calculate current likelihood
        c = _log_likelihood_t(y_clipped[m : m + 1], rho_m, nu_m)
        c = np.exp(np.log(c))

        # Get first derivative
        diff = _derivative_1st_rho_l(y_clipped[m : m + 1], rho_m, nu_m)

        t1 = u[m]
        t2 = v[m]
        t4 = 1.0 - rho_m * rho_m
        M_val = nu_m * t4 + t1 * t1 + t2 * t2 - 2.0 * rho_m * t1 * t2

        t3 = -(nu_m + 1.0) * (1.0 + rho_m * rho_m) / t4 / t4
        t5 = (nu_m + 2.0) * nu_m / M_val
        t6 = 2.0 * (nu_m + 2.0) * np.power(nu_m * rho_m + t1 * t2, 2.0) / M_val / M_val
        t7 = diff / c

        deriv[m] = c * (t3 + t5 + t6 + t7 * t7)

    return deriv.squeeze()


def _derivative_2nd_nu(y, rho, nu):
    """Second derivative wrt nu for t copula"""
    M = y.shape[0]
    deriv = np.empty((M, 1), dtype=np.float64)
    eps = np.finfo(float).eps
    y_clipped = np.clip(y, eps, 1 - eps)

    u = np.array(
        [
            st.t.ppf(
                y_clipped[i, 0], df=nu[i] if hasattr(nu, "__len__") else nu
            ).squeeze()
            for i in range(M)
        ]
    )
    v = np.array(
        [
            st.t.ppf(
                y_clipped[i, 1], df=nu[i] if hasattr(nu, "__len__") else nu
            ).squeeze()
            for i in range(M)
        ]
    )

    for m in range(M):
        rho_m = rho[m] if hasattr(rho, "__len__") else rho
        nu_m = nu[m] if hasattr(nu, "__len__") else nu

        # Calculate current likelihood
        c = _log_likelihood_t(y_clipped[m : m + 1], rho_m, nu_m)

        # Get first derivative
        diff_nu = _derivative_1st_nu_l(y_clipped[m : m + 1], rho_m, nu_m)

        x1 = u[m]
        x2 = v[m]

        # Common terms
        t1 = (nu_m + 1.0) / 2.0
        t2 = nu_m / 2.0
        t23 = nu_m * nu_m
        t3 = 1.0 / t23
        t4 = 1.0 / (2.0 * nu_m)
        t5 = 0.5 * sp.polygamma(1, t1)  # trigamma
        t6 = 1.0 - rho_m * rho_m
        t9 = 0.5 * sp.polygamma(1, t2)  # trigamma
        t10 = -t5 + t9 - t3 - t4

        # Get derivatives of quantile functions
        out1 = _diff_quantile_nu(x1, nu_m)
        out2 = _diff_quantile_nu(x2, nu_m)

        M_val = nu_m * t6 + x1 * x1 + x2 * x2 - 2.0 * rho_m * x1 * x2

        t8 = x1 * out2 + out1 * x2
        M_nu = t6 + 2.0 * x1 * out1 + 2.0 * x2 * out2 - 2.0 * rho_m * t8

        t24 = x1 * x1
        t25 = x2 * x2

        t11 = 1.0 + 2.0 * x1 * out1
        t12 = nu_m + t24
        t13 = t11 / t12

        t14 = 1.0 + 2.0 * x2 * out2
        t15 = nu_m + t25
        t16 = t14 / t15

        # Second derivatives of quantile functions (simplified approximation)
        out3 = diff2_x_nu(x1, nu_m)  # Second derivative of quantile function wrt nu
        out4 = diff2_x_nu(x2, nu_m)  # Second derivative of quantile function wrt nu

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
            - 4.0 * rho_m * out1 * out2
            - 2.0 * rho_m * (x2 * out3 + x1 * out4)
        )

        deriv[m] = (
            c
            * (
                t10
                + 0.5 * (t13 + t16)
                + t1 * (t18 - t21 + t20 - t22)
                + 0.5 * t13
                + 0.5 * t16
                - M_nu / M_val
                - (nu_m / 2.0 + 1.0) * (M_nu_nu / M_val - M_nu * M_nu / M_val / M_val)
            )
            + diff_nu * diff_nu / c
        )

    return deriv.squeeze()


# def _derivative_2nd_nu(y, rho, nu):
#     """Second derivative wrt nu for t copula using numerical differentiation"""
#     M = y.shape[0]
#     deriv = np.empty((M, 1), dtype=np.float64)
#     eps = 1e-8  # Small epsilon for numerical differentiation

#     for m in range(M):
#         rho_m = rho[m] if hasattr(rho, '__len__') else rho
#         nu_m = nu[m] if hasattr(nu, '__len__') else nu

#         # Create parameter values for nu +/- epsilon
#         nu1 = nu_m - eps
#         nu2 = nu_m + eps

#         # Calculate log-likelihood at nu-eps, nu, and nu+eps
#         ll1 = np.log(_log_likelihood_t(y[m:m+1], rho_m, nu1))
#         ll2 = np.log(_log_likelihood_t(y[m:m+1], rho_m, nu_m))
#         ll3 = np.log(_log_likelihood_t(y[m:m+1], rho_m, nu2))

#         # Second derivative: (f(x+h) - 2*f(x) + f(x-h)) / h²
#         deriv[m] = (ll3 - 2.0 * ll2 + ll1) / (eps * eps)

#     return deriv.squeeze()


from math import exp, fabs, log, pow, sqrt
from typing import List, Sequence, Tuple

import numpy as np
import scipy.stats as st
from scipy.special import beta as betafunc
from scipy.special import betaln as lbeta
from scipy.special import digamma, polygamma

# Make st.beta a function (alias to scipy.special.beta) so your function works unchanged
try:
    import scipy.special as _spsp

    st.beta = _spsp.beta
except Exception:
    pass


def trigamma(x: float) -> float:
    return float(polygamma(1, x))


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

    lbet = lbeta(p, q)
    pa = digamma(p)
    pa1 = trigamma(p)
    pb = digamma(q)
    pb1 = trigamma(q)
    pab = digamma(p + q)
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


def diff_t_nu_nu(x: float, nu: float) -> float:
    """
    Translation of C diff_t_nu_nu.
    """
    x_help = x if x >= 0.0 else -x
    xmax = nu / (nu + x_help * x_help)

    t1 = 1.0 / (x_help * x_help + nu)
    t2 = nu / 2.0
    t3 = 0.5

    I, Ip, Ipp = inbeder(xmax, t2, t3)
    nu = np.clip(nu, 1e-6, np.inf)
    t4 = (nu + 1.0) / 2.0
    t5 = pow(nu, nu / 2.0 - 1.0) * x_help
    t6 = pow(t1, t4)
    t7 = betafunc(t2, 0.5)
    t8 = t5 * t6
    t9 = nu * t1

    t11 = digamma(0.5 * nu)
    t12 = digamma(0.5 * nu + 0.5)
    t13 = t11 - t12
    t14 = 1.0 / t7

    t10 = -t1 * t4 + (t2 - 1.0) / nu + 0.5 * log(t1) + 0.5 * log(nu)

    out = -0.125 * Ipp + t8 * t14 * (-0.25 * log(t9) + 0.5 * t13 - 0.5 * t10)
    if x < 0.0:
        out = -out
    return out


def diff_dt_nu(x: float, nu: float) -> float:
    """
    Translation of C diff_dt_nu.
    """
    t1 = (nu + 1.0) / 2.0
    t2 = digamma(t1)
    t3 = betafunc(nu * 0.5, 0.5)
    nu = max(nu, 1e-6)
    t4 = sqrt(nu)
    t6 = digamma(0.5 * nu)
    t10 = -0.5 / t3 / t4 * (t6 - t2 + 1.0 / nu)
    t11 = 1.0 + (x * x) / nu
    t13 = pow(t11, -t1)
    t14 = 1.0 / t3 / t4
    t15 = log(t11)
    t16 = -t1 * x * x / (nu * nu) / t11

    out = t10 * t13 + t14 * (t13 * (-0.5 * t15 - t16))
    return out


def diff_dt_x(x: float, nu: float) -> float:
    """
    Translation of C diff_dt_x.
    """
    t2 = (nu + 1.0) / nu
    nu = max(nu, 1e-6)
    t3 = sqrt(nu)
    t4 = 1.0 / (t3 * betafunc(nu * 0.5, 0.5))
    t5 = 1.0 + (x * x) / nu
    t6 = (nu + 3.0) / 2.0
    t7 = pow(t5, -t6)
    out = -t4 * t2 * x * t7
    return out


# ------------------------------ Your function (unchanged) ------------------------------


def _diff_quantile_nu(x, nu):
    """Derivative of t quantile function wrt degrees of freedom parameter"""
    x_help = np.abs(x)

    # Handle edge cases
    if np.isclose(x_help, 0):
        return 0.0

    # Calculate components based on the C code
    xmax = nu / (nu + x_help**2)

    # t distribution pdf at x_help (dt function in C)
    t1 = st.t.pdf(x_help, df=nu)

    # Parameters for incomplete beta function
    t2 = nu / 2.0
    t3 = 0.5

    # Incomplete beta function derivative using inbeder function
    inbeder_result = inbeder(xmax, t2, t3)
    inbeder_out_1 = inbeder_result[1]  # First derivative wrt p

    t4 = (nu + 1.0) / 2.0
    t5 = np.power(nu, nu / 2.0 - 1.0) * x_help
    t6 = np.power(1.0 / (x_help**2 + nu), t4)
    t7 = st.beta(nu / 2.0, 0.5)

    # Main calculation matching C code exactly
    result = 1.0 / (2.0 * t1) * (0.5 * inbeder_out_1 + (t5 * t6) / t7)

    # Apply sign correction
    if x < 0:
        result = -result

    return result


# ------------------------------ Top-level: diff2_x_nu ------------------------------


def diff2_x_nu(x: float, nu: float) -> float:
    """
    Translation of C diff2_x_nu:
        out = (-t5*t4^2 - t2 - 2*t3*t4) / t1
    where:
        t1 = t.pdf(x; nu)               (SciPy)
        t2 = diff_t_nu_nu(x, nu)
        t3 = diff_dt_nu(x, nu)
        t4 = _diff_quantile_nu(x, nu)   (your function)
        t5 = diff_dt_x(x, nu)
    """
    t1 = st.t.pdf(x, df=nu)
    t2 = diff_t_nu_nu(x, nu)
    t3 = diff_dt_nu(x, nu)
    t4 = _diff_quantile_nu(x, nu)
    t5 = diff_dt_x(x, nu)

    if t1 == 0.0:
        raise ZeroDivisionError("t.pdf(x, nu) returned 0; cannot divide.")

    out = (-t5 * t4 * t4 - t2 - 2.0 * t3 * t4) / t1
    return out
