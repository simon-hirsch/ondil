from typing import Tuple

import numpy as np
import scipy.special as spc
import scipy.stats as st

from ..base import Distribution, LinkFunction, ScipyMixin
from ..links import Identity, Log, LogShiftTwo
from ..types import ParameterShapes


class SkewTMeanStd(ScipyMixin, Distribution):

    corresponding_gamlss: str = "SST"
    parameter_names = {0: "mu", 1: "sigma", 2: "nu", 3: "tau"}
    parameter_support = {
        0: (-np.inf, np.inf),
        1: (np.nextafter(0, 1), np.inf),
        2: (np.nextafter(0, 1), np.inf),
        3: (np.nextafter(0, 1), np.inf),
    }
    parameter_shape = {
        0: ParameterShapes.SCALAR,
        1: ParameterShapes.SCALAR,
        2: ParameterShapes.SCALAR,
        3: ParameterShapes.SCALAR,
    }
    distribution_support = (-np.inf, np.inf)
    scipy_dist = None
    scipy_names = None

    def __init__(
        self,
        loc_link: LinkFunction = Identity(),
        scale_link: LinkFunction = Log(),
        skew_link: LinkFunction = Log(),
        shape_link: LinkFunction = LogShiftTwo(),
    ) -> None:
        super().__init__(
            links={
                0: loc_link,
                1: scale_link,
                2: skew_link,
                3: shape_link,
            }
        )

    def _map(self, theta):
        mu, sigma, nu, tau = self.theta_to_params(theta)
        c = 2 * nu * ((1 + nu**2) * spc.beta(0.5, tau / 2) * tau**0.5) ** -1
        m = ((2 * tau**0.5) * (nu - nu**-1)) / ((tau - 1) * spc.beta(0.5, 0.5 * tau))
        s2 = (tau / (tau - 2)) * (nu**2 + nu**-2 - 1) - m**2
        mu_0 = mu - (sigma * m / np.sqrt(s2))
        sigma_0 = sigma / np.sqrt(s2)
        return mu_0, sigma_0, nu, tau, c

    def cdf(self, y: np.ndarray, theta: np.ndarray) -> np.ndarray:
        _, _, nu, tau = self.theta_to_params(theta)
        mu_0, sigma_0, nu, tau, c = self._map(theta)

        prob = np.where(
            y < mu_0,
            (2 / (1 + nu**2)) * st.t.cdf(x=(nu * (y - mu_0) / sigma_0), df=tau),
            (1 / (1 + nu**2))
            * (1 + 2 * nu**2 * (st.t.cdf(x=(y - mu_0) / (sigma_0 * nu), df=tau) - 0.5)),
        )
        return prob

    def ppf(self, q: np.ndarray, theta: np.ndarray) -> np.ndarray:
        _, _, nu, tau = self.theta_to_params(theta)
        mu_0, sigma_0, nu, tau, c = self._map(theta)

        quantile = np.where(
            q <= (1 + nu**2) ** -1,
            mu_0 + (sigma_0 / nu) * st.t.ppf((q * (1 + nu**2)) * 0.5, tau),
            mu_0
            + sigma_0 * nu * st.t.ppf((q * (1 + nu**2) - 1 + nu**2) / (2 * nu**2), tau),
        )
        return quantile

    def pdf(self, y: np.ndarray, theta: np.ndarray) -> np.ndarray:
        _, _, nu, tau = self.theta_to_params(theta)
        mu_0, sigma_0, nu, tau, c = self._map(theta)
        z = (y - mu_0) / sigma_0
        density = np.where(
            y < mu_0,
            (c / sigma_0) * (1 + ((nu**2) * (z**2)) / tau) ** (-(tau + 1) / 2),
            (c / sigma_0) * (1 + (z**2) / ((nu**2) * tau)) ** (-(tau + 1) / 2),
        )
        return density

    def rvs(self, size: Tuple[int, ...], theta: np.ndarray) -> np.ndarray:
        u = st.uniform.rvs(size=size)
        samples = self.ppf(u, theta)
        return samples

    def logcdf(self, y, theta):
        return np.log(self.cdf(y, theta))

    def logpdf(self, y: np.ndarray, theta: np.ndarray) -> np.ndarray:
        return np.log(self.pdf(y, theta))
