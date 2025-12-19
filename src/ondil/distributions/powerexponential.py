from typing import Tuple

import numpy as np
from scipy.special import digamma, gamma, gammaln, polygamma
from scipy.stats import gamma as gamma_dist

from ..base import Distribution, LinkFunction
from ..links import Identity, Log
from ..types import ParameterShapes


class PowerExponential(Distribution):
    r"""
    Power Exponential distribution (GAMLSS: PE).

    The PDF is defined as:

    \\(f(y \\mid \\mu, \\sigma, \\nu) = \\left(\\frac{\\nu}{2 \\sigma c(\\nu)}\\right) \\exp\\left(-0.5 \\left|\\frac{y - \\mu}{\\sigma c(\\nu)}\\right|^{\\nu}\\right)\\)

    where \\(c(\\nu) = \\sqrt{2} \\left(\\frac{\\Gamma(1/\\nu)}{\\Gamma(3/\\nu)}\\right)^{1/2}\\) and \\(\\Gamma(\\cdot)\\) is the gamma function.

    The parameters are: $\\mu$ (location), $\\sigma$ (scale), and $\\nu$ (shape).


    """

    corresponding_gamlss = "PE"
    parameter_names = {0: "mu", 1: "sigma", 2: "nu"}
    parameter_support = {
        0: (-np.inf, np.inf),
        1: (np.nextafter(0, 1), np.inf),
        2: (np.nextafter(0, 1), np.inf),
    }
    distribution_support = (-np.inf, np.inf)
    parameter_shape = {
        0: ParameterShapes.SCALAR,
        1: ParameterShapes.SCALAR,
        2: ParameterShapes.SCALAR,
    }

    def __init__(
        self,
        loc_link: LinkFunction = Identity(),
        scale_link: LinkFunction = Log(),
        shape_link: LinkFunction = Log(),
    ) -> None:
        super().__init__(links={0: loc_link, 1: scale_link, 2: shape_link})

    def _log_c(self, nu):
        return 0.5 * (-(2 / nu) * np.log(2) + gammaln(1 / nu) - gammaln(3 / nu))

    def _c(self, nu):
        return np.exp(self._log_c(nu))

    def _z(self, y, mu, sigma):
        return (y - mu) / sigma

    def logpdf(self, y, theta):
        mu, sigma, nu = self.theta_to_params(theta)
        c = self._c(nu)
        z = self._z(y, mu, sigma)
        return (
            np.log(nu)
            - np.log(sigma)
            - self._log_c(nu)
            - 0.5 * np.abs(z / c) ** nu
            - (1 + 1 / nu) * np.log(2)
            - gammaln(1 / nu)
        )

    def pdf(self, y, theta):
        return np.exp(self.logpdf(y, theta))

    def cdf(self, y, theta):
        mu, sigma, nu = self.theta_to_params(theta)
        c = self._c(nu)
        z = self._z(y, mu, sigma)
        s = 0.5 * (np.abs(z / c) ** nu)
        sign_z = np.sign(z)
        return 0.5 * (1 + gamma_dist.cdf(s, a=1 / nu, scale=1) * sign_z)

    def logcdf(self, y, theta):
        return np.log(self.cdf(y, theta))

    def ppf(self, q, theta):
        mu, sigma, nu = self.theta_to_params(theta)
        c = self._c(nu)

        q = np.asarray(q)

        # Output container
        result = np.full_like(q, fill_value=np.nan, dtype=np.float64)

        # Handling boundary conditions explicitly
        result[q == 0] = -np.inf
        result[q == 1] = np.inf

        # Validating quantiles between 0 and 1 (excluding boundaries)
        valid = (q > 0) & (q < 1)
        p = q[valid]
        sign = np.sign(p - 0.5)
        gamma_q = gamma_dist.ppf(np.abs(2 * p - 1), a=1 / nu, scale=1)
        z = sign * (2 * gamma_q) ** (1 / nu) * c
        result[valid] = mu + sigma * z

        return result

    def calculate_conditional_initial_values(self, y, theta, param):
        if param == 0:
            # Initial guess for mu
            return np.full_like(y, np.mean(y))
        elif param == 1:
            # Initial guess for sigma
            return np.full_like(y, np.std(y))
        elif param == 2:
            # Initial guess for nu
            return np.full_like(y, 2.0)
        else:
            raise ValueError("Invalid parameter index.")

    def rvs(self, theta, n=1, random_state=None):
        rng = np.random.default_rng(random_state)
        p = rng.uniform(size=n)
        return self.ppf(p, theta)

    def dl1_dp1(self, y, theta, param):
        self._validate_dln_dpn_inputs(y, theta, param)
        mu, sigma, nu = self.theta_to_params(theta)
        z = self._z(y, mu, sigma)
        c = self._c(nu)

        if param == 0:
            return (np.sign(z) * nu / (2 * sigma * np.abs(z))) * (np.abs(z / c) ** nu)
        elif param == 1:
            return ((nu / 2) * (np.abs(z / c) ** nu) - 1) / sigma
        elif param == 2:
            logc_deriv = (1 / (2 * nu**2)) * (
                2 * np.log(2) - digamma(1 / nu) + 3 * digamma(3 / nu)
            )
            s = np.abs(z / c) ** nu
            return (
                (1 / nu)
                - 0.5 * np.log(np.abs(z / c)) * s
                + np.log(2) / (nu**2)
                + digamma(1 / nu) / (nu**2)
                + (-1 + (nu / 2) * s) * logc_deriv
            )

    def d2ldv2(self, y, theta):
        mu, sigma, nu = self.theta_to_params(theta)

        dlogc_dv = (1 / (2 * nu**2)) * (
            2 * np.log(2) - digamma(1 / nu) + 3 * digamma(3 / nu)
        )

        p = (1 + nu) / nu
        part1 = p * polygamma(1, p) + 2 * digamma(p) ** 2
        part2 = digamma(p) * (np.log(2) + 3 - 3 * digamma(3 / nu) - nu)
        part3 = -3 * digamma(3 / nu) * (1 + np.log(2))
        part4 = -(nu + np.log(2)) * np.log(2)
        part5 = -nu + (nu**4) * (dlogc_dv**2)

        d2ldv2 = -(part1 + part2 + part3 + part4 + part5) / (nu**3)
        d2ldv2 = np.where(d2ldv2 < -1e-15, d2ldv2, -1e-15)
        return d2ldv2

    def dl2_dp2(self, y, theta, param):
        self._validate_dln_dpn_inputs(y, theta, param)
        mu, sigma, nu = self.theta_to_params(theta)

        if param == 0:
            z = self._z(y, mu, sigma)
            c = self._c(nu)
            dldm = (np.sign(z) * nu / (2 * sigma * np.abs(z))) * (np.abs(z / c) ** nu)
            closed_form = -(nu**2 * gamma(2 - 1 / nu) * gamma(3 / nu)) / (
                (sigma * gamma(1 / nu)) ** 2
            )
            d2ldm2 = np.where(nu < 1.05, -(dldm**2), closed_form)
            return d2ldm2

        if param == 1:
            return -nu / (sigma**2)

        if param == 2:
            return self.d2ldv2(y, theta)

    def dl2_dpp(self, y, theta, params):
        self._validate_dl2_dpp_inputs(y, theta, params)
        if (set(params) == {0, 1}) or (set(params) == {0, 2}):
            return np.zeros_like(y)

        _, sigma, nu = self.theta_to_params(theta)
        return (1 / (2 * sigma)) * (
            (3 / nu) * (digamma(1 / nu) - digamma(3 / nu)) + 2 + (2 / nu)
        )

    def initial_values(self, y: np.ndarray) -> np.ndarray:
        mu_init = np.full_like(y, np.mean(y))
        sigma_init = np.full_like(y, np.std(y, ddof=1))
        nu_init = np.full_like(y, np.log(2))
        return np.stack((mu_init, sigma_init, nu_init), axis=1)

    def theta_to_params(self, theta: np.ndarray) -> Tuple[np.ndarray, ...]:
        theta = np.atleast_2d(theta)  # This ensures it's always 2D
        return tuple(theta[:, i] for i in range(self.n_params))

    def pmf(self, *args, **kwargs):
        raise NotImplementedError("PMF is not defined for continuous distributions")

    def logpmf(self, *args, **kwargs):
        raise NotImplementedError("logPMF is not defined for continuous distributions")
