from typing import Dict, Tuple

import numpy as np
import scipy.stats as st
import scipy.special as spc

from ..base import Distribution, LinkFunction, ScipyMixin
from ..links import Log
from ..types import ParameterShapes
from ..robust_math import SMALL_NUMBER


class NegativeBinomial(ScipyMixin, Distribution):
    r"""The negative binomial Distribution.

    The distribution function is defined as:

    $$
    f(y|\\mu, \\sigma) = \\frac{\\Gamma(y + \\frac{\\mu}{\\sigma})}{\\Gamma(\\frac{\\mu}{\\sigma}) y!} \\left(\\frac{1}{1 + \\sigma}\\right)^{\\frac{\\mu}{\\sigma}} \\left(\\frac{\\sigma}{1 + \\sigma}\\right)^y
    $$
    with location parameter $\\mu > 0$ and scale parameter $\\sigma > 0$.

    This is a reparameterization of the `scipy.stats.nbinom` distribution. The scipy
    parameters `n` and `p` are related to `mu` and `sigma` as follows:
    $$
        n = \\frac{1}{\\sigma}  \\quad \\text{and} \\quad p = \\frac{1}{1 + \\sigma \\mu}
    $$
    The `scipy` parameters have the following interpretations:
    - `n`: number of successes until the experiment is stopped
    - `p`: probability of success in each experiment
    and the distribution gives the number of failures before the `n`-th success.

    The parameters in the above equation can be interpreted as:
    - $\\mu$: mean of the distribution
    - $\\sigma$: dispersion parameter, controlling the variance of the distribution

    Args:
        loc_link (LinkFunction, optional): The link function for $\\mu$. Defaults to Log().
        scale_link (LinkFunction, optional): The link function for $\\sigma$. Defaults to Log().

    """

    corresponding_gamlss: str = "NBI"

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
    scipy_dist = st.nbinom
    scipy_names = {"mu": "n", "sigma": "p"}
    is_discrete = True

    def __init__(
        self,
        loc_link: LinkFunction = Log(),
        scale_link: LinkFunction = Log(),
    ) -> None:
        super().__init__(links={0: loc_link, 1: scale_link})

    def theta_to_scipy_params(self, theta) -> Dict[str, np.ndarray]:
        mu, sigma = self.theta_to_params(theta)
        n = 1 / np.clip(sigma, SMALL_NUMBER, np.inf)
        p = 1 / np.clip(1 + sigma * mu, SMALL_NUMBER, np.inf)
        return {"n": n, "p": p}

    def dl1_dp1(self, y: np.ndarray, theta: np.ndarray, param: int = 0) -> np.ndarray:
        self._validate_dln_dpn_inputs(y, theta, param)
        mu, sigma = self.theta_to_params(theta)

        match param:
            case 0:
                return (y - mu) / (mu * (1 + mu * sigma))
            case 1:
                return -((1 / sigma) ** 2) * (
                    spc.digamma(y + (1 / sigma))
                    - spc.digamma(1 / sigma)
                    - np.log(1 + mu * sigma)
                    - (y - mu) * sigma / (1 + mu * sigma)
                )

    def dl2_dp2(self, y: np.ndarray, theta: np.ndarray, param: int = 0) -> np.ndarray:
        # SH: interesting that RSS have different expressions for these second derivatives
        match param:
            case 0:
                self._validate_dln_dpn_inputs(y, theta, param)
                mu, sigma = self.theta_to_params(theta)
                return -1 / (mu * (1 + mu * sigma))
            case 1:
                return -np.clip(
                    self.dl1_dp1(y, theta, param=1) ** 2,
                    SMALL_NUMBER,
                    np.inf,
                )

    def dl2_dpp(
        self, y: np.ndarray, theta: np.ndarray, params: Tuple[int, int] = (0, 1)
    ) -> np.ndarray:
        self._validate_dl2_dpp_inputs(y, theta, params)
        # No cross derivatives for single parameter distribution
        return np.zeros_like(y)

    def initial_values(self, y: np.ndarray) -> np.ndarray:
        mixed_mean = (np.mean(y) + y).flatten() / 2

        return np.vstack([
            mixed_mean,
            np.clip((np.var(y) / mixed_mean) / np.mean(y) ** 2, 0.1, np.inf),
        ]).T
