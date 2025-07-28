from typing import Tuple

import numpy as np
import scipy.special as spc
import scipy.stats as st
from scipy.integrate import quad
from scipy.optimize import root_scalar

from ..base import Distribution, LinkFunction
from ..links import Identity, Log


class SkewT1(Distribution):
    """
    Azzalini Skew-t type-1 Distribution for GAMLSS.

    Parameters:
    0 : Location (mu)
    1 : Scale (sigma)
    2 : Skewness (nu)
    3 : Tail behaviour (tau)
    """

    corresponding_gamlss: str = "ST1"

    parameter_names = {0: "mu", 1: "sigma", 2: "nu", 3: "tau"}
    parameter_support = {
        0: (-np.inf, np.inf),
        1: (np.nextafter(0, 1), np.inf),
        2: (-np.inf, np.inf),
        3: (np.nextafter(0, 1), np.inf),
    }
    distribution_support = (-np.inf, np.inf)

    def __init__(
        self,
        loc_link: LinkFunction = Identity(),
        scale_link: LinkFunction = Log(),
        skew_link: LinkFunction = Identity(),
        tail_link: LinkFunction = Log(),
        use_gamlss_init_values: bool = True,
    ) -> None:
        super().__init__(
            links={
                0: loc_link,
                1: scale_link,
                2: skew_link,
                3: tail_link,
            }
        )
        self.gamlss_init_values = use_gamlss_init_values

    def initial_values(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=np.float64).ravel()
        out = np.empty((y.shape[0], 4))
        if self.gamlss_init_values:
            # Match R GAMLSS ST1 initial values
            out[:, 0] = (y + np.mean(y)) / 2.0
            out[:, 1] = np.full_like(y, np.std(y) / 4.0)
            out[:, 2] = 0.1
            out[:, 3] = 5.0
        else:
            # Optionally: Use MLE or other start values
            out[:, 0] = np.mean(y)
            out[:, 1] = np.std(y)
            out[:, 2] = 0.0
            out[:, 3] = 1.0
        return out

    def calculate_conditional_initial_values(self, y, theta, param):
        theta = np.atleast_2d(theta).copy()
        n_obs, n_cols = theta.shape
        if n_cols < 4:
            defaults = self.initial_values(y)
            padded = np.zeros((n_obs, 4))
            padded[:, :n_cols] = theta
            padded[:, n_cols:] = defaults[:, n_cols:]
            theta = padded
        if param == 0:
            theta[:, 0] = (y + np.mean(y)) / 2.0
        elif param == 1:
            theta[:, 1] = np.full_like(y, np.std(y, ddof=1) / 4.0)
        elif param == 2:
            theta[:, 2] = 0.1
        elif param == 3:
            theta[:, 3] = 5.0
        return theta

    def theta_to_params(self, theta: np.ndarray):
        theta = np.atleast_2d(theta)
        return tuple(theta[:, k] for k in range(4))

    @staticmethod
    def _z(y, mu, sigma):
        return (y - mu) / sigma

    @staticmethod
    def _w(z, nu):
        return nu * z

    @staticmethod
    def _lambda(z, tau):
        return np.where(tau < 1e6, (tau + 1) / (tau + z**2), 1.0)

    def dl1_dp1(self, y: np.ndarray, theta: np.ndarray, param: int = 0) -> np.ndarray:
        self._validate_dln_dpn_inputs(y, theta, param)
        mu, sigma, nu, tau = self.theta_to_params(theta)
        z = self._z(y, mu, sigma)
        w = self._w(z, nu)
        lam = self._lambda(z, tau)
        t_pdf = st.t.pdf(w, tau)
        t_cdf = st.t.cdf(w, tau)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(t_cdf == 0, 0, t_pdf / t_cdf)

        if param == 0:
            return -ratio * nu / sigma + lam * z / sigma
        if param == 1:
            return -ratio * nu * z / sigma + (lam * z**2 - 1) / sigma
        if param == 2:
            with np.errstate(divide="ignore", invalid="ignore"):
                dwdv = np.where(nu == 0, 0.0, w / nu)
            return ratio * dwdv
        if param == 3:
            delta = 1e-3
            logcdf_plus = st.t.logcdf(w, tau + delta)
            logcdf_minus = st.t.logcdf(w, tau - delta)
            j = (logcdf_plus - logcdf_minus) / (2 * delta)
            out = (
                j
                + (
                    spc.digamma((tau + 1) / 2)
                    - spc.digamma(tau / 2)
                    - 1 / tau
                    - np.log1p(z**2 / tau)
                    + lam * z**2 / tau
                )
                / 2
            )
            return out
        raise ValueError("param must be 0, 1, 2, or 3")

    def dl2_dp2(self, y: np.ndarray, theta: np.ndarray, param: int = 0) -> np.ndarray:
        self._validate_dln_dpn_inputs(y, theta, param)
        g = self.dl1_dp1(y, theta, param)
        h = -g * g
        h = np.where(h < -1e-15, h, -1e-15)
        return h

    def dl2_dpp(
        self, y: np.ndarray, theta: np.ndarray, params: Tuple[int, int] = (0, 1)
    ) -> np.ndarray:
        self._validate_dln_dpn_inputs(y, theta, params)
        i, j = params
        if i == j:
            return self.dl2_dp2(y, theta, i)
        mu, sigma, nu, tau = self.theta_to_params(theta)
        z = self._z(y, mu, sigma)
        w = self._w(z, nu)
        lam = self._lambda(z, tau)
        t_pdf = st.t.pdf(w, tau)
        t_cdf = st.t.cdf(w, tau)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(t_cdf == 0, 0, t_pdf / t_cdf)

        def s(idx):
            if idx == 0:
                return -ratio * nu / sigma + lam * z / sigma
            if idx == 1:
                return -ratio * nu * z / sigma + (lam * z**2 - 1) / sigma
            if idx == 2:
                with np.errstate(divide="ignore", invalid="ignore"):
                    dwdv = np.where(nu == 0, 0.0, w / nu)
                return ratio * dwdv
            if idx == 3:
                delta = 1e-3
                logcdf_plus = st.t.logcdf(w, tau + delta)
                logcdf_minus = st.t.logcdf(w, tau - delta)
                j = (logcdf_plus - logcdf_minus) / (2 * delta)
                return (
                    j
                    + (
                        spc.digamma((tau + 1) / 2)
                        - spc.digamma(tau / 2)
                        - 1 / tau
                        - np.log1p(z**2 / tau)
                        + lam * z**2 / tau
                    )
                    / 2
                )
            raise ValueError

        h = -s(i) * s(j)
        h = np.where(h < -1e-15, h, -1e-15)
        return h

    def logpdf(self, y, theta):
        mu, sigma, nu, tau = self.theta_to_params(theta)
        z = self._z(y, mu, sigma)
        w = self._w(z, nu)
        loglik1 = st.t.logcdf(w, tau) + st.t.logpdf(z, tau) + np.log(2) - np.log(sigma)
        loglik2 = st.norm.logcdf(w) + st.norm.logpdf(z) + np.log(2) - np.log(sigma)
        loglik = np.where(tau < 1e6, loglik1, loglik2)
        return loglik

    def pdf(self, y, theta):
        return np.exp(self.logpdf(y, theta))

    def cdf(self, y, theta):
        mu, sigma, nu, tau = self.theta_to_params(theta)
        z = self._z(y, mu, sigma)
        nu = np.asarray(nu)
        tau = np.asarray(tau)

        def dST1(x, nu_i, tau_i):
            wx = nu_i * x
            if tau_i < 1e6:
                return np.exp(
                    st.t.logcdf(wx, tau_i) + st.t.logpdf(x, tau_i) + np.log(2)
                )
            else:
                return np.exp(st.norm.logcdf(wx) + st.norm.logpdf(x) + np.log(2))

        opts = dict(epsabs=1e-10, epsrel=1e-10)
        result = np.empty_like(z)
        for i, (zi, nu_i, tau_i) in enumerate(zip(z, nu, tau)):
            result[i], _ = quad(lambda xx: dST1(xx, nu_i, tau_i), -np.inf, zi, **opts)
        return result

    def logcdf(self, y, theta):
        return np.log(self.cdf(y, theta))

    def ppf(self, q, theta):
        mu, sigma, nu, tau = self.theta_to_params(theta)
        q = np.asarray(q, dtype=np.float64)
        out = np.empty_like(q)
        for i, (qi, mui, sigi, nui, taui) in enumerate(zip(q, mu, sigma, nu, tau)):
            if qi <= 0:
                out[i] = -np.inf
                continue
            if qi >= 1:
                out[i] = np.inf
                continue

            def root_fn(x):
                return (
                    self.cdf(np.array([x]), np.array([[mui, sigi, nui, taui]]))[0] - qi
                )

            interval = [mui - sigi, mui + sigi]
            while root_fn(interval[0]) > qi:
                interval[0] -= sigi
            while root_fn(interval[1]) < qi:
                interval[1] += sigi
            sol = root_scalar(
                root_fn, bracket=interval, method="brentq", xtol=1e-10, rtol=1e-10
            )
            out[i] = sol.root if sol.converged else np.nan
        return out

    def rvs(self, size, theta):
        rng = np.random.default_rng()
        u = rng.uniform(size=size)
        theta = np.atleast_2d(theta)
        if theta.shape[0] == 1 and size > 1:
            theta = np.tile(theta, (size, 1))
        return self.ppf(u, theta)

    def pmf(self, y, theta):
        raise NotImplementedError(
            "PMF is not implemented for continuous distributions."
        )

    def logpmf(self, y, theta):
        raise NotImplementedError(
            "logPMF is not implemented for continuous distributions."
        )
