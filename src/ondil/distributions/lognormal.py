from typing import Tuple

import numpy as np
import scipy.stats as st

from ..base import Distribution, LinkFunction, ScipyMixin
from ..links import Identity, Log
from ..robust_math import robust_exp
from ..types import ParameterShapes


class LogNormal(ScipyMixin, Distribution):
    r"""
    The Log-Normal distribution with mean and standard deviation parameterization in the log-space.

    The probability density function of the distribution is defined as:
    $$
    f(y | \\mu, \\sigma) = \\frac{1}{y\\sigma\\sqrt{2\\pi}}\\exp\\left(-\\frac{(\log y - \\mu)^2}{2\\sigma^2}\\right).
    $$
     respectively
    $$
    f(y | \\theta_0, \\theta_1) = \\frac{1}{y\\theta_1\\sqrt{2\pi}}\exp\\left(-\\frac{(\\log y - \\theta_0)^2}{2\\theta_1^2}\\right).
    $$
    where $y$ is the observed data, $\\mu = \\theta_0$ is the location parameter and $\\sigma = \\theta_1$ is the scale parameter.

    !!! note
        Note that re-parameterization used to move from scipy.stats to GAMLSS is:
        $$
            \\mu = \\exp(\\theta_0)
        $$
        and can therefore be numerically unstable for large values of $\\theta_0$.
        We have re-implemented the PDF, CDF, PPF according to avoid this issue,
        however the rvs method still uses the scipy.stats implementation which is not
        numerically stable for large values of $\\theta_0$.

    """

    corresponding_gamlss: str = "LOGNO"
    parameter_names = {0: "mu", 1: "sigma"}
    parameter_support = {
        0: (-np.inf, np.inf),
        1: (np.nextafter(0, 1), np.inf),
    }
    parameter_shape = {
        0: ParameterShapes.SCALAR,
        1: ParameterShapes.SCALAR,
    }
    distribution_support = (0, np.inf)
    scipy_dist = st.lognorm
    scipy_names = {"mu": "scale", "sigma": "s"}

    def __init__(
        self,
        loc_link: LinkFunction = Identity(),
        scale_link: LinkFunction = Log(),
    ) -> None:
        super().__init__(
            links={
                0: loc_link,
                1: scale_link,
            }
        )

    def theta_to_scipy_params(self, theta: np.ndarray) -> dict:
        mu = theta[:, 0]
        sigma = theta[:, 1]
        return {"s": sigma, "scale": robust_exp(mu), "loc": 0}

    def dl1_dp1(self, y: np.ndarray, theta: np.ndarray, param: int = 0) -> np.ndarray:
        self._validate_dln_dpn_inputs(y, theta, param)
        mu, sigma = self.theta_to_params(theta)

        if param == 0:
            return (np.log(y) - mu) / (sigma**2)

        if param == 1:
            return ((np.log(y) - mu) ** 2 - sigma**2) / (sigma**3)

    def dl2_dp2(self, y: np.ndarray, theta: np.ndarray, param: int = 0) -> np.ndarray:
        self._validate_dln_dpn_inputs(y, theta, param)
        mu, sigma = self.theta_to_params(theta)
        if param == 0:
            return -1 / (sigma**2)

        if param == 1:
            return -2 / (sigma**2)

    def dl2_dpp(
        self, y: np.ndarray, theta: np.ndarray, params: Tuple[int, int] = (0, 1)
    ) -> np.ndarray:
        self._validate_dl2_dpp_inputs(y, theta, params)
        if sorted(params) == [0, 1]:
            return np.zeros_like(y)

    def initial_values(self, y: np.ndarray) -> np.ndarray:
        log_y = np.log(y)
        initial_params = [np.mean(log_y), np.std(log_y, ddof=1)]
        return np.tile(initial_params, (y.shape[0], 1))

    def pdf(self, y: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """
        Probability density function of the Log-Normal distribution.
        """
        mu, sigma = self.theta_to_params(theta)
        return (
            1
            / (y * sigma * np.sqrt(2 * np.pi))
            * np.exp(-((np.log(y) - mu) ** 2) / (2 * sigma**2))
        )

    def cdf(self, y: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """
        Cumulative distribution function of the Log-Normal distribution.
        """
        mu, sigma = self.theta_to_params(theta)
        return st.norm.cdf((np.log(y) - mu) / sigma)

    def ppf(self, p: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """
        Percent-point function (quantile function) of the Log-Normal distribution.
        """
        mu, sigma = self.theta_to_params(theta)
        return np.exp(mu + sigma * st.norm.ppf(p))

    def logpdf(self, y: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """
        Logarithm of the probability density function of the Log-Normal distribution.
        """
        mu, sigma = self.theta_to_params(theta)
        return np.log(1 / (y * sigma * np.sqrt(2 * np.pi))) + (
            -((np.log(y) - mu) ** 2) / (2 * sigma**2)
        )

    def logcdf(self, y, theta):
        return np.log(self.cdf(y, theta))
