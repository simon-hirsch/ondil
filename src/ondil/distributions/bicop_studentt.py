# Author: Christian Schulz
# License: GPL-3.0

import math
from typing import Dict, List, Tuple

import numpy as np
import scipy.special as sp
import scipy.stats as st
from numba import njit, prange
from scipy.optimize import minimize_scalar

from ..base import BivariateCopulaMixin, CopulaMixin, Distribution, LinkFunction
from ..links import FisherZLink, ParameterToKendallsTau, LogShiftTwo
from ..robust_math import UMAX, UMIN
from ..types import ParameterShapes


# Bounds for the 1D nu MLE. Below 2 the t-copula variance is undefined; above
# ~30 the t-copula is numerically indistinguishable from the Gaussian copula
# so further refinement is meaningless.
_NU_BOUNDS = (2.0, 30.0)
_NU_FALLBACK = 8.0


def _profile_mle_nu(
    y: np.ndarray,
    rho_values: np.ndarray,
    bounds: Tuple[float, float] = _NU_BOUNDS,
    xatol: float = 1e-4,
    maxiter: int = 100,
) -> float:
    """Derivative-free 1D MLE for nu at a given (possibly heterogeneous) rho.

    Uses Brent's bounded method. Cheap because each evaluation only touches
    ``_log_likelihood_t`` (two ``t.ppf`` + two ``t.pdf`` calls), avoiding the
    expensive ``inbeder``-based derivatives entirely.

    Parameters
    ----------
    y : np.ndarray of shape (n, 2)
        Pseudo-observations in (0, 1).
    rho_values : np.ndarray
        Per-observation rho values. Will be reshaped to (n, 1).
    bounds : tuple of float
        (nu_min, nu_max) for the optimizer.
    xatol : float
        Absolute tolerance on nu.
    maxiter : int
        Maximum number of iterations.

    Returns
    -------
    float
        Optimal nu, or a sane fallback if the optimizer fails.
    """
    n = y.shape[0]
    rho_arr = np.asarray(rho_values, dtype=float).reshape(-1, 1)
    if rho_arr.shape[0] == 1 and n > 1:
        rho_arr = np.full((n, 1), rho_arr[0, 0])

    def neg_loglik(nu: float) -> float:
        nu_arr = np.full((n, 1), nu)
        try:
            lik = _log_likelihood_t(y, rho_arr, nu_arr)
            if not np.all(np.isfinite(lik)) or np.any(lik <= 0):
                return 1e10
            return -np.sum(np.log(lik))
        except Exception:
            return 1e10

    result = minimize_scalar(
        neg_loglik,
        bounds=bounds,
        method="bounded",
        options={"xatol": xatol, "maxiter": maxiter},
    )

    if result.success and bounds[0] <= result.x <= bounds[1]:
        return float(result.x)
    return _NU_FALLBACK


class BivariateCopulaStudentT(BivariateCopulaMixin, CopulaMixin, Distribution):
    """Bivariate Student-t copula.

    Estimation strategy
    -------------------
    rho is fitted via IWLS using the analytic score and Hessian
    (``_derivative_1st_rho`` / ``_derivative_2nd_rho``).

    nu is fitted by a derivative-free 1D profile MLE at the current rho. This
    avoids the expensive ``inbeder``-based quantile derivatives that would
    otherwise be invoked through ``_derivative_1st_nu`` / ``_derivative_2nd_nu``
    on every IWLS step. Those derivatives are still implemented and exposed
    via ``dl1_dp1`` / ``dl2_dp2`` for callers that want score/Hessian access
    (e.g. standard errors), but the fitting path does not use them.

    The optimizer treats nu as a single global scalar (consistent with
    ``parameter_shape[1] = ParameterShapes.SCALAR``) and broadcasts the
    result back to shape (n, 1).
    """

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
        param_link_1: LinkFunction = ParameterToKendallsTau(),
        param_link_2: LinkFunction = ParameterToKendallsTau(),
        family_code: int = 2,
        nu_bounds: Tuple[float, float] = _NU_BOUNDS,
        nu_xatol: float = 1e-4,
        nu_maxiter: int = 100,
    ):
        super().__init__(
            links={0: link_1, 1: link_2},
            param_links={0: param_link_1, 1: param_link_2},
        )
        self.family_code = family_code
        self.is_multivariate = True
        self._regularization_allowed = {0: False, 1: False}

        # Configuration for the 1D nu optimizer.
        self.nu_bounds = nu_bounds
        self.nu_xatol = nu_xatol
        self.nu_maxiter = nu_maxiter

    @staticmethod
    def fitted_elements(dim: int):
        return {0: 1, 1: 1}

    def theta_to_params(self, theta):
        return theta[0].reshape(-1, 1), theta[1].reshape(-1, 1)

    def set_initial_guess(self, theta, param):
        return theta

    # ------------------------------------------------------------------ #
    # Score / Hessian (kept intact; used by IWLS for rho, available for  #
    # nu but not invoked during fitting).                                #
    # ------------------------------------------------------------------ #

    def dl1_dp1(self, y: np.ndarray, theta: Dict, param: int = 0):
        """First derivative of the log-likelihood wrt the requested parameter."""
        rho, nu = self.theta_to_params(theta)
        if param == 0:
            return _derivative_1st_rho(y=y, rho=rho, nu=nu)
        return _derivative_1st_nu(y=y, rho=rho, nu=nu)

    def dl2_dp2(self, y: np.ndarray, theta: Dict, param: int = 0, clip=False):
        """Second derivative of the log-likelihood wrt the requested parameter."""
        rho, nu = self.theta_to_params(theta)
        if param == 0:
            return _derivative_2nd_rho(y=y, rho=rho, nu=nu)
        return _derivative_2nd_nu(y=y, rho=rho, nu=nu)

    def element_score(self, y: np.ndarray, theta: Dict, param: int = 0, k: int = 0):
        return self.element_dl1_dp1(y=y, theta=theta, param=param, k=k)

    def element_hessian(self, y: np.ndarray, theta: Dict, param: int = 0, k: int = 0):
        return self.element_dl2_dp2(y=y, theta=theta, param=param, k=k)

    def element_dl1_dp1(
        self, y: np.ndarray, theta: Dict, param: int = 0, k: int = 0, clip=False
    ):
        rho, nu = self.theta_to_params(theta)
        if param == 0:
            return _derivative_1st_rho(y, rho, nu)
        return _derivative_1st_nu(y, rho, nu)

    def element_dl2_dp2(
        self, y: np.ndarray, theta: Dict, param: int = 0, k: int = 0, clip=False
    ):
        rho, nu = self.theta_to_params(theta)
        if param == 0:
            return _derivative_2nd_rho(y, rho, nu)
        return _derivative_2nd_nu(y, rho, nu)

    def dl2_dpp(self, y: np.ndarray, theta: Dict, param: int = 0):
        raise NotImplementedError("Not implemented.")

    # ------------------------------------------------------------------ #
    # Initial values                                                      #
    # ------------------------------------------------------------------ #

    def initial_values(self, y, param=0):
        """Initial parameter values for IWLS.

        - rho: closed-form from Kendall's tau (sin(tau * pi / 2)).
        - nu : 1D profile MLE at the tau-implied rho (derivative-free).
        """
        M = y.shape[0]
        tau = st.kendalltau(y[:, 0], y[:, 1]).correlation
        rho_init = float(np.sin(tau * np.pi / 2.0))

        if param == 0:
            return np.full((M, 1), rho_init)

        rho_arr = np.full((M, 1), rho_init)
        nu_init = _profile_mle_nu(
            y,
            rho_arr,
            bounds=self.nu_bounds,
            xatol=self.nu_xatol,
            maxiter=self.nu_maxiter,
        )
        return np.full((M, 1), nu_init)

    # ------------------------------------------------------------------ #
    # Per-iteration nu update (called by the outer fitting loop)         #
    # ------------------------------------------------------------------ #

    def update_nu(self, y: np.ndarray, theta: Dict) -> np.ndarray:
        """Re-optimize nu by 1D profile MLE at the current rho.

        Intended to be called once per outer IWLS iteration, after rho has
        been updated. Returns an (n, 1) array suitable for assignment back
        into ``theta[1]``.
        """
        rho, _ = self.theta_to_params(theta)
        nu_opt = _profile_mle_nu(
            y,
            rho,
            bounds=self.nu_bounds,
            xatol=self.nu_xatol,
            maxiter=self.nu_maxiter,
        )
        return np.full((y.shape[0], 1), nu_opt)

    def df_iteration(self, y: np.ndarray, rho_values: np.ndarray) -> float:
        """Backward-compatible alias used by some calling code.

        Returns a scalar nu (not broadcast). Use ``update_nu`` when you want
        an array of shape (n, 1).
        """
        return _profile_mle_nu(
            y,
            rho_values,
            bounds=self.nu_bounds,
            xatol=self.nu_xatol,
            maxiter=self.nu_maxiter,
        )

    # ------------------------------------------------------------------ #
    # Distribution interface                                             #
    # ------------------------------------------------------------------ #

    def cdf(self, y, theta):
        raise NotImplementedError("Not implemented")

    def ppf(self, q, theta):
        raise NotImplementedError("Not implemented")

    def rvs(self, size, theta):
        """Sample from the bivariate t copula via the inverse h-function."""
        z1 = np.random.uniform(size=size)
        z2 = np.random.uniform(size=size)
        return self.hinv(z1, z2, theta, un=2)

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

    # ------------------------------------------------------------------ #
    # h-function and inverse                                             #
    # ------------------------------------------------------------------ #

    def hfunc(
        self, u: np.ndarray, v: np.ndarray, theta: Dict, un: int, family_code=2
    ) -> np.ndarray:
        """Conditional distribution function h(u|v) for the bivariate t copula."""
        rho, nu = self.theta_to_params(theta)

        u_mask_low = u < self.UMIN
        u_mask_high = u > self.UMAX
        v_mask_low = v < self.UMIN
        v_mask_high = v > self.UMAX

        u = np.where(u_mask_low, self.UMIN, u)
        u = np.where(u_mask_high, self.UMAX, u)
        v = np.where(v_mask_low, self.UMIN, v)
        v = np.where(v_mask_high, self.UMAX, v)

        if un == 1:
            u, v = v, u

        h = np.where((v == 0) | (u == 0), 0, np.nan)
        h = np.where(v == 1, u, h).reshape(-1, 1)

        qt_u = st.t.ppf(u.reshape(-1, 1), df=nu + 1.0).reshape(-1, 1)
        qt_v = st.t.ppf(v.reshape(-1, 1), df=nu).reshape(-1, 1)

        denom = np.sqrt((nu + qt_v**2) * (1 - rho**2) / (nu + 1))
        x = (qt_u - rho * qt_v) / denom

        finite_mask = np.isfinite(x)
        neg_mask = ~finite_mask & ((qt_u - rho * qt_v) < 0)
        pos_mask = ~finite_mask & ((qt_u - rho * qt_v) >= 0)

        h = np.where(
            finite_mask,
            st.t.cdf(x[:, 0], df=nu).reshape(-1, 1),
            h,
        )
        h = np.where(neg_mask, 0, h)
        h = np.where(pos_mask, 1, h)

        return h.squeeze()

    def hinv(
        self, u: np.ndarray, v: np.ndarray, theta: np.ndarray, un: int, family_code=2
    ) -> np.ndarray:
        """Inverse conditional distribution function h^(-1)(u|v)."""
        u_mask_low = u < self.UMIN
        u_mask_high = u > self.UMAX
        v_mask_low = v < self.UMIN
        v_mask_high = v > self.UMAX

        u = np.where(u_mask_low, self.UMIN, u)
        u = np.where(u_mask_high, self.UMAX, u)
        v = np.where(v_mask_low, self.UMIN, v)
        v = np.where(v_mask_high, self.UMAX, v)

        rho, nu = self.theta_to_params(theta)

        qt_u = st.t.ppf(u, df=nu + 1.0).reshape(-1, 1)
        qt_v = st.t.ppf(v, df=nu).reshape(-1, 1)

        mu = rho * qt_v
        var = ((nu + qt_v**2) * (1.0 - rho**2)) / (nu + 1.0)
        hinv = st.t.cdf((np.sqrt(var) * qt_u + mu), df=nu).reshape(-1, 1)

        h_mask_low = hinv < 0
        h_mask_high = hinv > 1
        hinv = np.where(h_mask_low, 0, hinv)
        hinv = np.where(h_mask_high, 1, hinv)

        return hinv.squeeze()

    def get_regularization_size(self, dim: int) -> int:
        return dim


##########################################################
### Functions for the Student-t copula derivatives     ###
##########################################################


def stable_gamma_division(x1, x2):
    """Stable computation of gamma(x1)/gamma(x2)."""
    x1_arr = np.asarray(x1, dtype=float)
    x2_arr = np.asarray(x2, dtype=float)

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
    """Log-likelihood for bivariate t copula."""
    y_clipped = np.clip(y, UMIN, UMAX)
    nu1 = np.asarray(nu).ravel()
    t1 = st.t.ppf(y_clipped[:, 0], df=nu1).reshape(-1, 1)
    t2 = st.t.ppf(y_clipped[:, 1], df=nu1).reshape(-1, 1)

    gamma_ratio = nu / 2
    dt1 = st.t.pdf(t1[:, 0], df=nu1).reshape(-1, 1)
    dt2 = st.t.pdf(t2[:, 0], df=nu1).reshape(-1, 1)

    quad_form = (t1 * t1 + t2 * t2 - 2.0 * rho * t1 * t2) / (nu * (1 - rho**2))

    f = (
        gamma_ratio
        / (nu * np.pi * np.sqrt(1 - rho**2) * dt1 * dt2)
        * np.power(1.0 + quad_form, -(nu + 2.0) / 2.0)
    )

    f[f <= 0] = 1e-16
    return f.squeeze()


def _derivative_1st_rho(y, rho, nu):
    """First derivative wrt rho for t copula."""
    y_clipped = np.clip(y, UMIN, UMAX)

    nu1 = np.asarray(nu).ravel()
    t1 = st.t.ppf(y_clipped[:, 0], df=nu1).reshape(-1, 1)
    t2 = st.t.ppf(y_clipped[:, 1], df=nu1).reshape(-1, 1)

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
    """First derivative wrt rho times the likelihood."""
    y_clipped = np.clip(y, UMIN, UMAX)

    c = _log_likelihood_t(y_clipped, rho, nu).reshape(-1, 1)
    nu1 = np.asarray(nu).ravel()
    t1 = st.t.ppf(y_clipped[:, 0], df=nu1).reshape(-1, 1)
    t2 = st.t.ppf(y_clipped[:, 1], df=nu1).reshape(-1, 1)

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
    """First derivative wrt nu for t copula."""
    eps = np.finfo(float).eps
    y_clipped = np.clip(y, eps, 1 - eps)
    nu1 = np.asarray(nu).ravel()
    u = st.t.ppf(y_clipped[:, 0], df=nu1).reshape(-1, 1)
    v = st.t.ppf(y_clipped[:, 1], df=nu1).reshape(-1, 1)

    t1 = sp.digamma((nu + 1.0) / 2.0)
    t2 = sp.digamma(nu / 2.0)
    t3 = 0.5 * np.log(1.0 - rho * rho)
    t4 = (nu - 2.0) / (2.0 * nu)
    t5 = 0.5 * np.log(nu)
    t6 = -t1 + t2 + t3 - t4 - t5
    t10 = (nu + 2.0) / 2.0

    x1 = u
    x2 = v

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
    """First derivative wrt nu times the likelihood."""
    eps = np.finfo(float).eps
    y_clipped = np.clip(y, eps, 1 - eps)
    nu1 = np.asarray(nu).ravel()
    u = st.t.ppf(y_clipped[:, 0], df=nu1).reshape(-1, 1)
    v = st.t.ppf(y_clipped[:, 1], df=nu1).reshape(-1, 1)

    c = _log_likelihood_t(y_clipped, rho, nu).reshape(-1, 1)

    t1 = sp.digamma((nu + 1.0) / 2.0)
    t2 = sp.digamma(nu / 2.0)
    t3 = 0.5 * np.log(1.0 - rho * rho)
    t4 = (nu - 2.0) / (2.0 * nu)
    t5 = 0.5 * np.log(nu)
    t6 = -t1 + t2 + t3 - t4 - t5
    t10 = (nu + 2.0) / 2.0

    x1 = u
    x2 = v

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
    """Second derivative wrt rho for t copula."""
    y_clipped = np.clip(y, UMIN, UMAX)
    nu1 = np.asarray(nu).ravel()
    u = st.t.ppf(y_clipped[:, 0], df=nu1).reshape(-1, 1)
    v = st.t.ppf(y_clipped[:, 1], df=nu1).reshape(-1, 1)

    c = _log_likelihood_t(y_clipped, rho, nu).reshape(-1, 1)
    c = np.exp(np.log(c))
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
    """Second derivative wrt nu for t copula."""
    eps = np.finfo(float).eps
    y_clipped = np.clip(y, eps, 1 - eps)
    nu1 = np.asarray(nu).ravel()
    u = st.t.ppf(y_clipped[:, 0], df=nu1).reshape(-1, 1)
    v = st.t.ppf(y_clipped[:, 1], df=nu1).reshape(-1, 1)

    c = _log_likelihood_t(y_clipped, rho, nu).reshape(-1, 1)
    c = np.exp(np.log(c))

    diff_nu = _derivative_1st_nu_l(y_clipped, rho, nu).reshape(-1, 1)
    x1 = u
    x2 = v

    t1 = (nu + 1.0) / 2.0
    t2 = nu / 2.0
    t23 = nu * nu
    t3 = 1.0 / t23
    t4 = 1.0 / (2.0 * nu)
    t5 = 0.5 * sp.polygamma(1, t1)
    t6 = 1.0 - rho * rho
    t9 = 0.5 * sp.polygamma(1, t2)
    t10 = -t5 + t9 - t3 - t4

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

    out3 = diff2_x_nu(x1, nu)
    out4 = diff2_x_nu(x2, nu)

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


def trigamma(x: float) -> float:
    return float(sp.polygamma(1, x))


@njit(cache=True)
def _an_bn_1_p(x, p, q):
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
    return an0, an1, an2, bn0, bn1, bn2


@njit(cache=True)
def _an_bn_1_q(x, p, q):
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
    return an0, an1, an2, bn0, bn1, bn2


@njit(cache=True)
def _an_bn_n_p(x, p, q, n):
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
    return an0, an1, an2, bn0, bn1, bn2


@njit(cache=True)
def _an_bn_n_q(x, p, q, n):
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
    return an0, an1, an2, bn0, bn1, bn2


@njit(cache=True, parallel=True)
def _inbeder_core_vec(x, p, q, flipped, c0log, c0exp, c1, c2, err, minappx, maxappx):
    """Compiled iterative core for inbeder. See ``inbeder_vec_numba``."""
    n_elem = x.size
    der0 = np.empty(n_elem, dtype=np.float64)
    der1 = np.empty(n_elem, dtype=np.float64)
    der2 = np.empty(n_elem, dtype=np.float64)

    for idx in prange(n_elem):
        xi = x[idx]
        pi = p[idx]
        qi = q[idx]

        an1_0, an1_1, an1_2 = 1.0, 0.0, 0.0
        an2_0, an2_1, an2_2 = 1.0, 0.0, 0.0
        bn1_0, bn1_1, bn1_2 = 1.0, 0.0, 0.0
        bn2_0, bn2_1, bn2_2 = 0.0, 0.0, 0.0

        der_old0, der_old1, der_old2 = 0.0, 0.0, 0.0

        n = 0
        while True:
            n += 1

            if n == 1:
                if flipped[idx]:
                    an0, an1v, an2v, bn0, bn1v, bn2v = _an_bn_1_q(xi, pi, qi)
                else:
                    an0, an1v, an2v, bn0, bn1v, bn2v = _an_bn_1_p(xi, pi, qi)
            else:
                if flipped[idx]:
                    an0, an1v, an2v, bn0, bn1v, bn2v = _an_bn_n_q(xi, pi, qi, n)
                else:
                    an0, an1v, an2v, bn0, bn1v, bn2v = _an_bn_n_p(xi, pi, qi, n)

            dan0 = an0 * an2_0 + bn0 * an1_0
            dbn0 = an0 * bn2_0 + bn0 * bn1_0

            dan1 = an1v * an2_0 + an0 * an2_1 + bn1v * an1_0 + bn0 * an1_1
            dbn1 = an1v * bn2_0 + an0 * bn2_1 + bn1v * bn1_0 + bn0 * bn1_1

            dan2 = (
                an2v * an2_0
                + 2.0 * an1v * an2_1
                + an0 * an2_2
                + bn2v * an1_0
                + 2.0 * bn1v * an1_1
                + bn0 * an1_2
            )
            dbn2 = (
                an2v * bn2_0
                + 2.0 * an1v * bn2_1
                + an0 * bn2_2
                + bn2v * bn1_0
                + 2.0 * bn1v * bn1_1
                + bn0 * bn1_2
            )

            Rn = dan0
            if abs(dbn0) > abs(dan0):
                Rn = dbn0

            an1_0 /= Rn
            an1_1 /= Rn
            an1_2 /= Rn
            bn1_0 /= Rn
            bn1_1 /= Rn
            bn1_2 /= Rn
            dan1 /= Rn
            dan2 /= Rn
            dbn1 /= Rn
            dbn2 /= Rn

            if abs(dbn0) > abs(dan0):
                dan0 = dan0 / dbn0
                dbn0 = 1.0
            else:
                dbn0 = dbn0 / dan0
                dan0 = 1.0

            dr0 = dan0 / dbn0
            dbn0_sq = dbn0 * dbn0
            dr1 = (dan1 - dr0 * dbn1) / dbn0
            dr2 = (-2.0 * dan1 * dbn1 + 2.0 * dr0 * dbn1 * dbn1) / dbn0_sq + (
                dan2 - dr0 * dbn2
            ) / dbn0

            an2_0, an2_1, an2_2 = an1_0, an1_1, an1_2
            an1_0, an1_1, an1_2 = dan0, dan1, dan2
            bn2_0, bn2_1, bn2_2 = bn1_0, bn1_1, bn1_2
            bn1_0, bn1_1, bn1_2 = dbn0, dbn1, dbn2

            pr = 0.0
            if dr0 > 0.0:
                pr = np.exp(c0log[idx] + np.log(dr0))

            d0 = pr
            d1v = pr * c1[idx] + c0exp[idx] * dr1
            d2v = pr * c2[idx] + 2.0 * c0exp[idx] * c1[idx] * dr1 + c0exp[idx] * dr2

            denom0 = max(err, abs(d0))
            denom1 = max(err, abs(d1v))
            denom2 = max(err, abs(d2v))

            r0 = abs(der_old0 - d0) / denom0
            r1 = abs(der_old1 - d1v) / denom1
            r2 = abs(der_old2 - d2v) / denom2

            der_old0, der_old1, der_old2 = d0, d1v, d2v

            dmax = r0
            if r1 > dmax:
                dmax = r1
            if r2 > dmax:
                dmax = r2

            if n < minappx:
                dmax = 1.0
            if n >= maxappx:
                dmax = 0.0

            if dmax <= err:
                der0[idx] = d0
                der1[idx] = d1v
                der2[idx] = d2v
                break

    return der0, der1, der2


def inbeder_vec_numba(
    x_in,
    p_in,
    q_in,
    err: float = 1e-12,
    minappx: int = 3,
    maxappx: int = 200,
):
    """Vectorized + Numba-accelerated inbeder."""
    x_in = np.asarray(x_in, dtype=float)
    p_in = np.asarray(p_in, dtype=float)
    q_in = np.asarray(q_in, dtype=float)
    x_in, p_in, q_in = np.broadcast_arrays(x_in, p_in, q_in)

    shp = x_in.shape
    x0 = x_in.ravel()
    p0 = p_in.ravel()
    q0 = q_in.ravel()

    EPS = 1e-12
    flipped = x0 > (p0 / (p0 + q0))

    x = np.where(flipped, 1.0 - x0, x0)
    p = np.where(flipped, q0, p0)
    q = np.where(flipped, p0, q0)

    x = np.clip(x, EPS, 1.0 - EPS)
    p = np.maximum(p, EPS)

    lbet = sp.betaln(p, q)
    pa = sp.digamma(p)
    pa1 = sp.polygamma(1, p)
    pb = sp.digamma(q)
    pb1 = sp.polygamma(1, q)
    pab = sp.digamma(p + q)
    pab1 = sp.polygamma(1, p + q)

    omx = 1.0 - x
    logx = np.log(x)
    logomx = np.log(omx)

    c0log = p * logx + (q - 1.0) * logomx - lbet - np.log(p)
    c0exp = np.exp(c0log)

    c1 = np.empty_like(x)
    c2 = np.empty_like(x)

    nf = ~flipped
    f = flipped

    c1[nf] = logx[nf] - 1.0 / p[nf] - pa[nf] + pab[nf]
    c2[nf] = c1[nf] * c1[nf] + 1.0 / (p[nf] * p[nf]) - pa1[nf] + pab1[nf]

    c1[f] = logomx[f] - pb[f] + pab[f]
    c2[f] = c1[f] * c1[f] - pb1[f] + pab1[f]

    d0, d1v, d2v = _inbeder_core_vec(
        x, p, q, flipped, c0log, c0exp, c1, c2, err, minappx, maxappx
    )

    d0 = np.where(flipped, 1.0 - d0, d0)
    d1v = np.where(flipped, -d1v, d1v)
    d2v = np.where(flipped, -d2v, d2v)

    return d0.reshape(shp), d1v.reshape(shp), d2v.reshape(shp)


# ------------------------------ Derivative helpers ------------------------------
def diff_t_nu_nu(x, nu):
    """Vectorized translation of C diff_t_nu_nu."""
    x_arr = np.asarray(x, dtype=float)
    nu_arr = np.asarray(nu, dtype=float)
    x_b, nu_b = np.broadcast_arrays(x_arr, nu_arr)

    nu_eff = np.clip(nu_b, 1e-6, np.inf)
    abs_x = np.abs(x_b)
    sign = np.where(x_b < 0.0, -1.0, 1.0)

    xmax = nu_eff / (nu_eff + abs_x * abs_x)
    t1 = 1.0 / (abs_x * abs_x + nu_eff)
    t2 = 0.5 * nu_eff
    t4 = 0.5 * (nu_eff + 1.0)
    _, _, Ipp = inbeder_vec_numba(xmax, t2, 0.5)

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
    """Vectorized translation of C diff_dt_nu."""
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
    """Vectorized translation of C diff_dt_x."""
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
    """Derivative of t quantile function wrt degrees of freedom (vectorized)."""
    x_arr = np.asarray(x, dtype=float)
    nu_arr = np.asarray(nu, dtype=float)
    x_b, nu_b = np.broadcast_arrays(x_arr, nu_arr)

    out = np.zeros_like(x_b, dtype=float)

    EPS_X = 1e-14
    TINY = 1e-300

    abs_x = np.abs(x_b)
    sign = np.where(x_b < 0.0, -1.0, 1.0)
    mask = abs_x > EPS_X

    if not np.any(mask):
        return float(0.0) if out.size == 1 else out

    nu_eff = np.maximum(nu_b, 1e-8)

    t_pdf = st.t.pdf(abs_x, df=nu_eff)
    t_pdf = np.maximum(t_pdf, TINY)

    t2 = 0.5 * nu_eff
    t4 = 0.5 * (nu_eff + 1.0)
    denom = abs_x * abs_x + nu_eff
    t6 = np.power(1.0 / denom, t4)
    t5 = np.power(nu_eff, t2 - 1.0) * abs_x
    t7 = sp.beta(t2, 0.5)

    xmax = nu_eff / (nu_eff + abs_x * abs_x)

    _, dIdp, _ = inbeder_vec_numba(xmax, t2, 0.5)

    res = 0.5 / t_pdf * (0.5 * dIdp + (t5 * t6) / t7)
    res *= sign
    out[mask] = res[mask]

    if out.size == 1:
        return float(out)
    return out


def diff2_x_nu(x, nu):
    """Vectorized translation of C diff2_x_nu."""
    x_b, nu_b = np.broadcast_arrays(np.asarray(x, float), np.asarray(nu, float))

    t1 = st.t.pdf(x_b, df=nu_b)
    TINY = 1e-300
    denom = np.maximum(t1, TINY)

    t2 = diff_t_nu_nu(x_b, nu_b)
    t3 = diff_dt_nu(x_b, nu_b)
    t4 = _diff_quantile_nu(x_b, nu_b)
    t5 = diff_dt_x(x_b, nu_b)

    out = (-t5 * t4 * t4 - t2 - 2.0 * t3 * t4) / denom
    if out.size == 1:
        return float(out)
    return out