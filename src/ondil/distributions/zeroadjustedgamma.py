from typing import Tuple

import numpy as np
import scipy.special as spc
import scipy.stats as st

from ..base import Distribution, LinkFunction
from ..links import Logit, Log


class ZeroAdjustedGamma(Distribution):
    """The Zero Adjusted Gamma Distribution for GAMLSS.
    
    The zero adjusted gamma distribution is a mixture of a discrete value 0 with
    probability \\nu, and a gamma GA(\\mu; \\sigma) distribution on the positive real line (0, \\infty)
    with probability (1 - \\nu).
    
    $$
    f_Y(y \\mid \\mu, \\sigma, \\nu) = 
    \\begin{cases}
        \\nu & \\text{if } y = 0 \\
        (1 - \\nu) f_W(y \mid \mu, \sigma) & \\text{if } y > 0
    \end{cases} 
    $$
    
    where $y$ is the observed data, $\\mu > 0$ is the location parameter,
    $\\sigma > 0$ is the scale parameter,
    and $\\nu \\in [0, \\infty) $  is the inflation parameter.

    """

    corresponding_gamlss: str = "ZAGA"
    parameter_names = {0: "mu", 1: "sigma", 2: "nu"}
    parameter_support = {
        0: (np.nextafter(0, 1), np.inf),
        1: (np.nextafter(0, 1), np.inf),
        2: (np.nextafter(0, 1), np.nextafter(1, 0)),
    }
    distribution_support = (0, np.inf)

    def __init__(
        self,
        loc_link: LinkFunction = Log(),
        scale_link: LinkFunction = Log(),
        inflation_link: LinkFunction = Logit(),
    ) -> None:
        super().__init__(links={0: loc_link, 1: scale_link, 2: inflation_link})

    def dl1_dp1(self, y: np.ndarray, theta: np.ndarray, param: int = 0) -> np.ndarray:
        self._validate_dln_dpn_inputs(y, theta, param)
        mu, sigma, nu = self.theta_to_params(theta)

        if param == 0:
            result = (y - mu) / ((sigma**2) * (mu**2))

            return np.where(y == 0, 0, result)

        if param == 1:
            result = (2 / sigma**3) * (
                (y / mu)
                - np.log(y)
                + np.log(mu)
                + np.log(sigma**2)
                - 1
                + spc.digamma(1 / (sigma**2))
            )

            return np.where(y == 0, 0, result)

        if param == 2:
            return np.where(y == 0, 1 / nu, -1 / (1 - nu))

    def dl2_dp2(self, y: np.ndarray, theta: np.ndarray, param: int = 0) -> np.ndarray:
        self._validate_dln_dpn_inputs(y, theta, param)
        mu, sigma, nu = self.theta_to_params(theta)
        if param == 0:
            result = -1 / ((sigma**2) * (mu**2))

            return np.where(y == 0, 0, result)

        if param == 1:
            result = (4 / sigma**4) - (4 / sigma**6) * spc.polygamma(1, (1 / sigma**2))

            return np.where(y == 0, 0, result)

        if param == 2:
            return -1 / (nu * (1 - nu))

    def dl2_dpp(
        self, y: np.ndarray, theta: np.ndarray, params: Tuple[int, int] = (0, 1)
    ) -> np.ndarray:
        self._validate_dl2_dpp_inputs(y, theta, params)
        mu, sigma, nu = self.theta_to_params(theta)

        if sorted(params) == [0, 1]:
            return np.zeros_like(y)

        if sorted(params) == [0, 2]:
            return np.zeros_like(y)

        if sorted(params) == [1, 2]:
            return np.zeros_like(y)

    def initial_values(self, y: np.ndarray) -> np.ndarray:
        return np.tile([np.mean(y), 1, 0.5], (y.shape[0], 1))

    def cdf(self, y, theta):
        mu, sigma, nu = self.theta_to_params(theta)
        shape = 1 / sigma**2
        scale = mu * sigma**2
        cdf_gamma = st.gamma(a=shape, loc=0, scale=scale).cdf(y)
        cont_cdf = nu + ((1 - nu) * cdf_gamma)

        result = np.where(y == 0, nu, cont_cdf)

        return result

    def pdf(self, y, theta):
        mu, sigma, nu = self.theta_to_params(theta)
        shape = 1 / sigma**2
        scale = mu * sigma**2
        pdf_gamma = st.gamma(a=shape, loc=0, scale=scale).pdf(y)

        result = (
            (1 - nu)
            * (y / (mu * sigma**2)) ** (1 / sigma**2)
            * (np.exp(-y / (mu * sigma**2)))
            / (y * spc.gamma(1 / sigma**2))
        )

        return np.where(y == 0, nu, result)

    def ppf(self, q, theta):
        mu, sigma, nu = self.theta_to_params(theta)
        shape = 1 / sigma**2
        scale = mu * sigma**2

        result = st.gamma(a=shape, loc=0, scale=scale).ppf((q - nu) / (1 - nu))

        return np.where(q <= nu, 0, result)

    def pmf(self, y, theta):
        raise NotImplementedError("PMF is not implemented for mixed distributions.")

    def rvs(self, size, theta):
        sim_unif = st.uniform.rvs(size=size)
        return self.ppf(q=sim_unif, theta=theta)

    def logpdf(self, y, theta):
        mu, sigma, nu = self.theta_to_params(theta)

        result = (
            np.log(1 - nu)
            + (1 / sigma**2) * np.log(y / (mu * sigma**2))
            - y / (mu * sigma**2)
            - np.log(y)
            - spc.gammaln(1 / sigma**2)
        )

        return np.where(y == 0, np.log(nu), result)

    def logcdf(self, y, theta):
        mu, sigma, nu = self.theta_to_params(theta)
        shape = 1 / sigma**2
        scale = mu * sigma**2
        cdf_gamma = st.gamma(a=shape, loc=0, scale=scale).cdf(y)
        cont_logcdf_gamma = np.log(nu + ((1 - nu) * cdf_gamma))

        result = np.where(y == 0, np.log(nu), cont_logcdf_gamma)

        return result

    def logpmf(self, y, theta):
        return super().logpmf(y, theta)

    def calculate_conditional_initial_values(self, y, theta, param):
        return super().calculate_conditional_initial_values(y, theta, param)
