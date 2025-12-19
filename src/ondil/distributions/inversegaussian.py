import numpy as np
import scipy.stats as st

from ..base import Distribution, LinkFunction, ScipyMixin
from ..links import Log
from ..types import ParameterShapes


class InverseGaussian(ScipyMixin, Distribution):
    r"""
    Inverse Gaussian (Wald) distribution for GAMLSS.

    This distribution is characterized by two parameters:
    - $\\mu$: the mean of the distribution.
    - $\\sigma$: the scale parameter, which is related to the variance.

    The probability density function (PDF) is given by:
    $$
        f(y; \\mu, \\sigma) = \\sqrt{\\frac{\\sigma}{2\\pi y^3}} \\exp\\left(-\\frac{(y - \\mu)^2}{2\\sigma^2 y}\\right)
    $$
    where $y > 0$, $\\mu > 0$, and $\\sigma > 0$.

    Note that the Inverse Gaussian distribution in `scipy.stats` is parameterized differently:

    * `mu` is the mean of the distribution.
    * `scale` is the scale parameter

    and the PDF is given by:
    $$
        f(y; \\mu, \\lambda) = \\sqrt{\\frac{\\lambda}{2\\pi y^3}} \\exp\\left(-\\frac{\\lambda (y - \\mu)^2}{2\\mu^2 y}\\right)
    $$
    where $y > 0$, $\\mu > 0$, and $\\lambda > 0$.

    The relationship between the parameters is:

    * `mu` in `scipy.stats` corresponds to $\\mu \\sigma^2$ in this implementation,
    * `scale` in `scipy.stats` corresponds to $1 / \\sigma^2$ in this implementation.
    * The `loc` parameter in `scipy.stats` is always 0.

    """

    corresponding_gamlss: str = "IG"
    parameter_names = {0: "mu", 1: "sigma"}
    parameter_support = {
        0: (np.nextafter(0, 1), np.inf),
        1: (np.nextafter(0, 1), np.inf),
    }
    parameter_shape = {
        0: ParameterShapes.SCALAR,
        1: ParameterShapes.SCALAR,
    }
    distribution_support = (np.nextafter(0, 1), np.inf)
    scipy_dist = st.invgauss
    scipy_names = {"mu": "mu", "sigma": "scale"}

    def __init__(
        self,
        loc_link: LinkFunction = Log(),
        scale_link: LinkFunction = Log(),
    ) -> None:
        super().__init__(
            links={
                0: loc_link,
                1: scale_link,
            }
        )

    def theta_to_scipy_params(self, theta: np.ndarray) -> dict:
        mu, sigma = self.theta_to_params(theta)
        return {"mu": mu * sigma**2, "loc": np.zeros_like(mu), "scale": 1 / sigma**2}

    def dl1_dp1(self, y: np.ndarray, theta: np.ndarray, param: int = 0) -> np.ndarray:
        mu, sigma = self.theta_to_params(theta)
        if param == 0:
            # (y-mu)/((sigma^2)*(mu^3))
            return (y - mu) / (sigma**2 * mu**3)
        if param == 1:
            # (-1/sigma) +((y-mu)^2)/(y*(sigma^3)*(mu^2)),
            return (-1 / sigma) + ((y - mu) ** 2) / (sigma**3 * mu**2 * y)
        raise ValueError("param must be 0 (mu) or 1 (sigma)")

    def dl2_dp2(self, y: np.ndarray, theta: np.ndarray, param: int = 0) -> np.ndarray:
        mu, sigma = self.theta_to_params(theta)
        if param == 0:
            # -1/((mu^3)*(sigma^2)),
            return -1 / (mu**3 * sigma**2)
        if param == 1:
            # -2/(sigma^2),
            return -2 / (sigma**2)
        raise ValueError("param must be 0 (mu) or 1 (sigma)")

    def dl2_dpp(
        self, y: np.ndarray, theta: np.ndarray, params: tuple[int, int] = (0, 1)
    ) -> np.ndarray:
        if params[0] == params[1]:
            raise ValueError("Cross derivatives must use different parameters.")
        return np.zeros_like(y)

    def initial_values(self, y: np.ndarray) -> np.ndarray:
        mu_init = np.mean(y)
        sigma_init = np.std(y) / np.sqrt(np.mean(y))
        initial_params = [mu_init, sigma_init]
        return np.tile(initial_params, (y.shape[0], 1))
