from typing import Tuple

import numpy as np
import scipy.stats as st

from ..base import Distribution, LinkFunction, ScipyMixin
from ..links import Log
from ..types import ParameterShapes


class Weibull(ScipyMixin, Distribution):
    r"""The Weibull distribution parameterized by scale (mu) and shape (sigma).

    The probability density function is defined as:
    $$
    f(y | \\mu, \\sigma) = \\frac{\\sigma}{\\mu} \\left(\\frac{y}{\\mu}\\right)^{\\sigma-1} \\exp\\left[-\\left(\\frac{y}{\\mu}\\right)^{\\sigma}\\right]
    $$

    where $y > 0$, $\\mu > 0$ is the scale parameter, and $\\sigma > 0$ is the shape parameter.

    !!! Note
        This distribution corresponds to the WEI() distribution in GAMLSS.

        The parameterization matches scipy.stats.weibull_min with:
        - scale = mu (GAMLSS mu)
        - c = sigma (GAMLSS sigma, the shape parameter)

    Args:
        scale_link (LinkFunction, optional): The link function for $\\mu$. Defaults to Log().
        shape_link (LinkFunction, optional): The link function for $\\sigma$. Defaults to Log().
    """

    corresponding_gamlss: str = "WEI"
    parameter_names = {0: "mu", 1: "sigma"}
    parameter_support = {
        0: (np.nextafter(0, 1), np.inf),
        1: (np.nextafter(0, 1), np.inf),
    }
    distribution_support = (np.nextafter(0, 1), np.inf)
    parameter_shape = {
        0: ParameterShapes.SCALAR,
        1: ParameterShapes.SCALAR,
    }
    # Scipy equivalent and parameter mapping ondil -> scipy
    scipy_dist = st.weibull_min
    scipy_names = {"mu": "scale", "sigma": "c"}

    def __init__(
        self,
        scale_link: LinkFunction = Log(),
        shape_link: LinkFunction = Log(),
    ) -> None:
        """Initialize the Weibull distribution.

        Args:
            scale_link (LinkFunction, optional): Link function for mu (scale). Defaults to Log().
            shape_link (LinkFunction, optional): Link function for sigma (shape). Defaults to Log().
        """
        super().__init__(
            links={
                0: scale_link,
                1: shape_link,
            }
        )

    def dl1_dp1(self, y: np.ndarray, theta: np.ndarray, param: int = 0) -> np.ndarray:
        self._validate_dln_dpn_inputs(y, theta, param)
        mu, sigma = self.theta_to_params(theta)

        if param == 0:
            # First derivative with respect to mu (scale parameter)
            # dldm = ((y/mu)^sigma - 1) * (sigma/mu)
            return ((y / mu) ** sigma - 1) * (sigma / mu)

        if param == 1:
            # First derivative with respect to sigma (shape parameter)
            # dldd = 1/sigma - log(y/mu) * ((y/mu)^sigma - 1)
            return (1 / sigma) - np.log(y / mu) * ((y / mu) ** sigma - 1)

    def dl2_dp2(self, y: np.ndarray, theta: np.ndarray, param: int = 0) -> np.ndarray:
        self._validate_dln_dpn_inputs(y, theta, param)
        mu, sigma = self.theta_to_params(theta)

        if param == 0:
            # Second derivative with respect to mu (scale parameter)
            # d2ldm2 = -sigma^2 / mu^2
            return -(sigma**2) / (mu**2)

        if param == 1:
            # Second derivative with respect to sigma (shape parameter)
            # d2ldd2 = -1.82368 / sigma^2
            # This is an approximation based on the expected Fisher information
            # The exact value would be -trigamma(1) ≈ -1.82368
            return -1.82368 / (sigma**2)

    def dl2_dpp(
        self, y: np.ndarray, theta: np.ndarray, params: Tuple[int, int] = (0, 1)
    ) -> np.ndarray:
        self._validate_dl2_dpp_inputs(y, theta, params)
        if sorted(params) == [0, 1]:
            # Cross derivative with respect to mu and sigma
            # d2ldmdd = 0.422784 / mu
            # This is an approximation based on the expected Fisher information
            # The exact value would be (digamma(1) - 1) ≈ 0.422784
            mu, _ = self.theta_to_params(theta)
            return 0.422784 / mu

    def initial_values(self, y: np.ndarray) -> np.ndarray:
        # Following GAMLSS initialization:
        # log.Y.m <- log(y)
        # var.Y.v <- var(log(y))
        # sd.Y.s <- 1.283/sqrt(var.Y.v)
        # mu <- exp(log.Y.m + 0.5772/sd.Y.s)
        # sigma <- 1.283/sqrt(var(log(y)))
        log_y = np.log(y)
        log_y_mean = np.mean(log_y)
        log_y_var = np.var(log_y, ddof=1)

        sigma_init = 1.283 / np.sqrt(log_y_var) if log_y_var > 0 else 1.0
        mu_init = np.exp(log_y_mean + 0.5772 / sigma_init)

        initial_params = [mu_init, sigma_init]
        return np.tile(initial_params, (y.shape[0], 1))
