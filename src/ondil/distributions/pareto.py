from typing import Tuple

import numpy as np
import scipy.stats as st

from ..base import Distribution, LinkFunction, ScipyMixin
from ..links import Log
from ..types import ParameterShapes


class Pareto(ScipyMixin, Distribution):
    """The Pareto Type II Distribution (Lomax) for GAMLSS.

    The distribution function is defined as in GAMLSS as:
    $$
    f(y|\\mu,\\sigma)=\\frac{\\sigma}{\\mu(1 + y/\\mu)^{\\sigma+1}}
    $$

    with the scale and shape parameters $\\mu, \\sigma > 0$ and support $y \\geq 0$.

    !!! Note
        The function is parameterized as GAMLSS' PARETO2() distribution.

        This parameterization is different to the `scipy.stats.lomax(c, loc, scale)` parameterization.

        We can use `Pareto().theta_to_scipy_params(theta)` to map the distribution parameters to scipy.

    The `scipy.stats.lomax()` distribution is defined as:
    $$
    f(x, c) = \\frac{c}{(1 + x)^{c+1}}
    $$
    with the shape parameter $c > 0$. The parameters can be mapped as follows:
    $$
    c = \\sigma
    $$
    and
    $$
    \\text{scale} = \\mu.
    $$

    Args:
        loc_link (LinkFunction, optional): The link function for $\\mu$. Defaults to Log().
        scale_link (LinkFunction, optional): The link function for $\\sigma$. Defaults to Log().
    """

    corresponding_gamlss: str = "PARETO2"

    parameter_names = {0: "mu", 1: "sigma"}
    parameter_support = {
        0: (np.nextafter(0, 1), np.inf),
        1: (np.nextafter(0, 1), np.inf),
    }
    parameter_shape = {
        0: ParameterShapes.SCALAR,
        1: ParameterShapes.SCALAR,
    }
    distribution_support = (0, np.inf)
    # Scipy equivalent and parameter mapping ondil -> scipy
    scipy_dist = st.lomax
    # Theta columns do not map 1:1 to scipy parameters for pareto
    # So we have to overload theta_to_scipy_params
    scipy_names = {}

    def __init__(
        self,
        loc_link: LinkFunction = Log(),
        scale_link: LinkFunction = Log(),
    ) -> None:
        super().__init__(links={0: loc_link, 1: scale_link})

    def theta_to_scipy_params(self, theta: np.ndarray) -> dict:
        """Map GAMLSS Parameters to scipy parameters.

        Args:
            theta (np.ndarray): parameters

        Returns:
            dict: Dict of (c, loc, scale) for scipy.stats.lomax(c, loc, scale)
        """
        mu = theta[:, 0]
        sigma = theta[:, 1]
        params = {"c": sigma, "loc": 0, "scale": mu}
        return params

    def dl1_dp1(self, y: np.ndarray, theta: np.ndarray, param: int = 0) -> np.ndarray:
        self._validate_dln_dpn_inputs(y, theta, param)
        mu, sigma = self.theta_to_params(theta)

        match param:
            case 0:
                # dldm = (sigma*y - mu) / (mu*(mu + y))
                return (sigma * y - mu) / (mu * (mu + y))
            case 1:
                # dldd = 1/sigma - log(1 + y/mu)
                return 1 / sigma - np.log(1 + y / mu)

    def dl2_dp2(self, y: np.ndarray, theta: np.ndarray, param: int = 0) -> np.ndarray:
        self._validate_dln_dpn_inputs(y, theta, param)
        mu, sigma = self.theta_to_params(theta)

        match param:
            case 0:
                # MU: Observed second derivative
                # d2ldm2 = (mu^2 - 2*mu*sigma*y - sigma*y^2) / (mu^2*(mu+y)^2)
                numerator = mu**2 - 2 * mu * sigma * y - sigma * y**2
                denominator = mu**2 * (mu + y) ** 2
                return numerator / denominator
            case 1:
                # SIGMA: Observed second derivative
                # d2ldd2 = -1/sigma^2
                return -1 / sigma**2

    def dl2_dpp(
        self, y: np.ndarray, theta: np.ndarray, params: Tuple[int, int] = (0, 1)
    ) -> np.ndarray:
        self._validate_dl2_dpp_inputs(y, theta, params)
        if sorted(params) == [0, 1]:
            # Observed cross derivative: d2ldmdd = y / (mu*(mu + y))
            mu, _ = self.theta_to_params(theta)
            return y / (mu * (mu + y))

    def initial_values(self, y: np.ndarray) -> np.ndarray:
        # Use method of moments estimates
        # For Pareto Type II: E[Y] = mu/(sigma-1) for sigma > 1
        # Var[Y] = mu^2 * sigma / ((sigma-1)^2 * (sigma-2)) for sigma > 2
        # Simple initial values: set mu = mean, sigma = 2
        initial_params = [np.mean(y), 2.0]
        return np.tile(initial_params, (y.shape[0], 1))
