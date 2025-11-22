from typing import Tuple

import numpy as np
import scipy.special as sp
import scipy.stats as st

from ..base import Distribution, LinkFunction, ScipyMixin
from ..links import Identity, Log
from ..types import ParameterShapes
from ..robust_math import robust_log


class SkewNormal(ScipyMixin, Distribution):
    """
    The Skew Normal distribution with location, scale, and skewness parameterization.

    The probability density function is:
    $$
    f(y | \\mu, \\sigma, \\nu) = \\frac{2}{\\sigma} \\phi\\left(\\frac{y - \\mu}{\\sigma}\\right) \\Phi\\left(\\nu \\frac{y - \\mu}{\\sigma}\\right)
    $$

    where $\\phi$ is the standard normal PDF and $\\Phi$ is the standard normal CDF.

    This distribution corresponds to the SN1() distribution in GAMLSS.
    """

    corresponding_gamlss: str = "SN1"
    parameter_names = {0: "mu", 1: "sigma", 2: "nu"}
    parameter_support = {
        0: (-np.inf, np.inf),
        1: (np.nextafter(0, 1), np.inf),
        2: (-np.inf, np.inf),
    }
    parameter_shape = {
        0: ParameterShapes.SCALAR,
        1: ParameterShapes.SCALAR,
        2: ParameterShapes.SCALAR,
    }
    distribution_support = (-np.inf, np.inf)
    scipy_dist = st.skewnorm
    scipy_names = {"mu": "loc", "sigma": "scale", "nu": "a"}

    def __init__(
        self,
        loc_link: LinkFunction = Identity(),
        scale_link: LinkFunction = Log(),
        skew_link: LinkFunction = Identity(),
    ) -> None:
        """Initialize the SkewNormal distribution.

        Args:
            loc_link (LinkFunction, optional): Location link. Defaults to Identity().
            scale_link (LinkFunction, optional): Scale link. Defaults to Log().
            skew_link (LinkFunction, optional): Skewness link. Defaults to Identity().
        """
        super().__init__(
            links={
                0: loc_link,
                1: scale_link,
                2: skew_link,
            }
        )

    def theta_to_scipy_params(self, theta: np.ndarray) -> dict:
        """Map GAMLSS parameters to scipy parameters.

        Args:
            theta (np.ndarray): parameters (mu, sigma, nu)

        Returns:
            dict: Dict of (a, loc, scale) for scipy.stats.skewnorm
        """
        mu = theta[:, 0]
        sigma = theta[:, 1]
        nu = theta[:, 2]
        return {"a": nu, "loc": mu, "scale": sigma}

    def dl1_dp1(self, y: np.ndarray, theta: np.ndarray, param: int = 0) -> np.ndarray:
        self._validate_dln_dpn_inputs(y, theta, param)
        mu, sigma, nu = self.theta_to_params(theta)

        z = (y - mu) / sigma
        w = nu * z
        s = (np.abs(w) ** 2) / 2

        # z <- (y - mu)/sigma
        # w <- nu * z
        # s <- ((abs(w))^2)/2
        # lpdf <- (1 - (1/2)) * log(2) - s - lgamma(1/2) - log(2)
        # lcdf <- log(0.5 * (1 + pgamma(s, shape = 1/2, scale = 1) * sign(w)))
        # dldm <- -(exp(lpdf - lcdf)) * nu/sigma + sign(z) * (abs(z)^(2 - 1))/sigma

        lpdf = (1 - (1 / 2)) * np.log(2) - s - sp.gammaln(1 / 2) - np.log(2)
        lcdf = robust_log(0.5 * (1 + st.gamma(a=0.5, scale=1).cdf(s) * np.sign(w)))

        if param == 0:
            # dL/dmu
            return (
                -np.exp(lpdf - lcdf) * nu / sigma
                + np.sign(z) * (np.abs(z) ** (2 - 1)) / sigma
            )

        if param == 1:
            # dL/dsigma
            return (
                -(np.exp(lpdf - lcdf)) * nu * z / sigma + ((np.abs(z) ** 2) - 1) / sigma
            )

        if param == 2:
            return (np.exp(lpdf - lcdf)) * w / nu

    def dl2_dp2(self, y: np.ndarray, theta: np.ndarray, param: int = 0) -> np.ndarray:
        self._validate_dln_dpn_inputs(y, theta, param)
        first_deriv = self.dl1_dp1(y, theta, param)
        # Pseudo second derivative
        # d2L/dparam^2
        return -first_deriv * first_deriv

    def dl2_dpp(
        self, y: np.ndarray, theta: np.ndarray, params: Tuple[int, int] = (0, 1)
    ) -> np.ndarray:
        self._validate_dl2_dpp_inputs(y, theta, params)
        first_deriv_param1 = self.dl1_dp1(y, theta, params[0])
        first_deriv_param2 = self.dl1_dp1(y, theta, params[1])
        # Pseudo second derivative
        # d2L/dparam1 dparam2
        return -first_deriv_param1 * first_deriv_param2

    def initial_values(self, y: np.ndarray) -> np.ndarray:
        """Calculate initial values for the GAMLSS fit.

        For skew normal, we start with:
        - mu = mean of y
        - sigma = std of y
        - nu = 0 (no skewness initially)
        """
        return np.vstack([
            (y + np.mean(y)) / 2,
            np.full_like(y, np.std(y, ddof=1) / 4),
            np.full_like(y, 0.1),
        ]).T
