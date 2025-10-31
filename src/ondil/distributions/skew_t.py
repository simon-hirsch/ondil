from typing import Tuple

import numpy as np
import scipy.special as spc
import scipy.stats as st

from ..base import Distribution, LinkFunction, ScipyMixin
from ..links import Identity, Log, LogShiftTwo
from ..types import ParameterShapes


class SkewT(Distribution):
    corresponding_gamlss: str = "ST3"
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

    def cdf(self, y: np.ndarray, theta: np.ndarray) -> np.ndarray:
        mu, sigma, nu, tau = self.theta_to_params(theta)
        cdf1 = 2 * st.t(df=tau, loc=0, scale=1).cdf(nu * (y - mu) / sigma)
        cdf2 = 1 + 2 * nu * nu * (
            st.t(df=tau, loc=0, scale=1).cdf((y - mu) / (sigma * nu)) - 0.5
        )
        cdf = np.where(y < mu, cdf1, cdf2) / (1 + nu**2)
        return cdf

    def ppf(self, q: np.ndarray, theta: np.ndarray) -> np.ndarray:
        mu, sigma, nu, tau = self.theta_to_params(theta)
        q1 = mu + (sigma / nu) * st.t.ppf((q * (1 + nu**2)) * 0.5, df=tau)
        q2 = mu + sigma * nu * st.t.ppf(
            (q * (1 + nu**2) - 1 + nu**2) / (2 * nu**2), df=tau
        )
        quantile = np.where(q <= (1 + nu**2) ** -1, q1, q2)
        return quantile

    def pdf(self, y: np.ndarray, theta: np.ndarray) -> np.ndarray:
        mu, sigma, nu, tau = self.theta_to_params(theta)
        z = (y - mu) / sigma
        density1 = (2 * nu / (sigma * (1 + nu**2))) * (
            1 + ((nu**2) * (z**2)) / tau
        ) ** (-(tau + 1) / 2)
        density2 = (2 * nu / (sigma * (1 + nu**2))) * (
            1 + (z**2) / ((nu**2) * tau)
        ) ** (-(tau + 1) / 2)
        density = np.where(y < mu, density1, density2)
        return density

    def rvs(self, size: Tuple[int, ...], theta: np.ndarray) -> np.ndarray:
        u = st.uniform.rvs(size=size)
        samples = self.ppf(u, theta)
        return samples

    def logcdf(self, y, theta):
        return np.log(self.cdf(y, theta))

    def logpdf(self, y: np.ndarray, theta: np.ndarray) -> np.ndarray:
        return np.log(self.pdf(y, theta))

    @staticmethod
    def _get_terms_deriv(mu, sigma, nu, tau, y):
        s1 = sigma / nu
        s2 = sigma * nu
        dsq1 = ((y - mu) ** 2) / (s1**2)
        dsq2 = ((y - mu) ** 2) / (s2**2)
        w1 = np.where(tau < 1000000, (tau + 1) / (tau + dsq1), 1)
        w2 = np.where(tau < 1000000, (tau + 1) / (tau + dsq2), 1)
        return s1, s2, dsq1, dsq2, w1, w2

    def dl1_dp1(self, y: np.ndarray, theta: np.ndarray, param: int = 0) -> np.ndarray:
        self._validate_dln_dpn_inputs(y, theta, param)
        mu, sigma, nu, tau = self.theta_to_params(theta)
        s1, s2, dsq1, dsq2, w1, w2 = self._get_terms_deriv(mu, sigma, nu, tau, y)

        match param:
            case 0:
                deriv = np.where(
                    y < mu,
                    (w1 * (y - mu)) / s1**2,
                    (w2 * (y - mu)) / s2**2,
                )
                return deriv
            case 1:
                deriv = np.where(
                    y < mu,
                    (w1 * dsq1 - 1) / sigma,
                    (w2 * dsq2 - 1) / sigma,
                )
                return deriv
            case 2:
                deriv = np.where(
                    y < mu,
                    -(w1 * dsq1 - 1) / nu,
                    (w2 * dsq2 + 1) / nu,
                ) - 2 * nu / (1 + nu**2)
                return deriv
            case 3:
                deriv_a = -0.5 * np.log(1 + dsq1 / tau) + (w1 * dsq1 - 1) / (2 * tau)
                deriv_b = -0.5 * np.log(1 + dsq2 / tau) + (w2 * dsq2 - 1) / (2 * tau)
                deriv = np.where(y < mu, deriv_a, deriv_b)
                deriv = (
                    deriv
                    + 0.5 * spc.digamma((tau + 1) / 2)
                    - 0.5 * spc.digamma(tau / 2)
                )
                return deriv

    def dl2_dp2(self, y: np.ndarray, theta: np.ndarray, param: int = 0) -> np.ndarray:
        self._validate_dln_dpn_inputs(y, theta, param)
        deriv = -(self.dl1_dp1(y, theta, param) ** 2)
        return np.clip(deriv, -np.inf, -1e-15)

    def dl2_dpp(
        self, y: np.ndarray, theta: np.ndarray, params: Tuple[int, int] = (0, 1)
    ) -> np.ndarray:
        self._validate_dl2_dpp_inputs(y, theta, params)
        return -self.dl1_dp1(y, theta, params[0]) * self.dl1_dp1(y, theta, params[1])

    def initial_values(self, y: np.ndarray) -> np.ndarray:
        initial_params = [np.mean(y), np.std(y, ddof=1), 1, 10]
        return np.tile(initial_params, (y.shape[0], 1))

    def mean(self, theta: np.ndarray) -> np.ndarray:
        mu, sigma, nu, tau = self.theta_to_params(theta)
        c = 2 * nu * ((1 + nu**2) * spc.beta(0.5, tau / 2) * tau**0.5) ** -1
        mean = mu + sigma * c
        return mean

    def variance(self, theta: np.ndarray) -> np.ndarray:
        _, sigma, nu, tau = self.theta_to_params(theta)
        m = ((2 * tau**0.5) * (nu - nu**-1)) / ((tau - 1) * spc.beta(0.5, 0.5 * tau))
        variance = sigma**2 * ((tau / (tau - 2)) * (nu**2 + nu**-2 - 1) - m**2)
        return variance

    def calculate_conditional_initial_values(self, y, theta, param):
        return super().calculate_conditional_initial_values(y, theta, param)

    def logpmf(self, y, theta):
        raise NotImplementedError("LogPMF is not implemented for SkewT distribution.")

    def pmf(self, y, theta):
        raise NotImplementedError("PMF is not implemented for SkewT distribution.")


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
        self._st3dist = SkewT()

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

    def logpmf(self, y, theta):
        raise NotImplementedError(
            "LogPMF is not implemented for SkewTMeanStd distribution."
        )

    def pmf(self, y, theta):
        raise NotImplementedError(
            "PMF is not implemented for SkewTMeanStd distribution."
        )

    def _get_terms_deriv(self, theta):
        mu, sigma, nu, tau = self.theta_to_params(theta)
        m1 = (2 * tau ** (1 / 2) * (nu**2 - 1)) / (
            (tau - 1) * spc.beta(1 / 2, tau / 2) * nu
        )
        m2 = (tau * (nu**3 + (1 / nu**3))) / ((tau - 2) * (nu + (1 / nu)))
        s1 = np.sqrt(m2 - m1**2)
        mu1 = mu - ((sigma * m1) / s1)
        sigma1 = sigma / s1
        return m1, m2, s1, mu1, sigma1

    def initial_values(self, y: np.ndarray) -> np.ndarray:
        initial_params = [np.mean(y), np.std(y, ddof=1), 1, 10]
        return np.tile(initial_params, (y.shape[0], 1))

    def dl1_dp1(self, y, theta, param):
        self._validate_dln_dpn_inputs(y, theta, param)
        _, sigma, nu, tau = self.theta_to_params(theta)
        m1, m2, s1, mu1, sigma1 = self._get_terms_deriv(theta)

        st3theta = np.column_stack((mu1, sigma1, nu, tau))

        match param:
            case 0:
                deriv = self._st3dist.dl1_dp1(y, st3theta, param)
                return deriv
            case 1:
                return -(m1 / s1) * self._st3dist.dl1_dp1(y, st3theta, 0) + (
                    1 / s1
                ) * self._st3dist.dl1_dp1(y, st3theta, 1)
            case 2:
                dl1dmu1 = self._st3dist.dl1_dp1(y, st3theta, 0)
                dl1dd1 = self._st3dist.dl1_dp1(y, st3theta, 1)
                dl1dv = self._st3dist.dl1_dp1(y, st3theta, 2)
                dmu1dm1 = -sigma / s1
                dmu1ds1 = (sigma * m1) / (s1**2)
                dd1ds1 = -sigma / (s1**2)
                dm1dv = (
                    (2 * tau ** (1 / 2)) / ((tau - 1) * spc.beta(1 / 2, tau / 2))
                ) * ((nu**2 + 1) / (nu**2))
                dm2dv = m2 * (
                    (6 * nu**5 / (nu**6 + 1)) - (2 / nu) - (2 * nu / (nu**2 + 1))
                )
                ds1dv = (dm2dv - 2 * m1 * dm1dv) / (2 * s1)
                dldv = (
                    dl1dmu1 * dmu1dm1 * dm1dv
                    + dl1dmu1 * dmu1ds1 * ds1dv
                    + dl1dd1 * dd1ds1 * ds1dv
                    + dl1dv
                )
                return dldv
            case 3:
                dl1dmu1 = self._st3dist.dl1_dp1(y, st3theta, 0)
                dl1dd1 = self._st3dist.dl1_dp1(y, st3theta, 1)
                dl1dt = self._st3dist.dl1_dp1(y, st3theta, 3)
                dmu1dm1 = -sigma / s1
                dmu1ds1 = (sigma * m1) / (s1**2)
                dd1ds1 = -sigma / (s1**2)
                dm1dt = m1 * (
                    (1 / (2 * tau))
                    - (1 / (tau - 1))
                    - 0.5 * (spc.digamma(tau / 2))
                    + 0.5 * (spc.digamma((tau + 1) / 2))
                )
                dm2dt = -m2 * (2 / (tau * (tau - 2)))
                ds1dt = (dm2dt - 2 * m1 * dm1dt) / (2 * s1)
                dldt = (
                    dl1dmu1 * dmu1dm1 * dm1dt
                    + dl1dmu1 * dmu1ds1 * ds1dt
                    + dl1dd1 * dd1ds1 * ds1dt
                    + dl1dt
                )
                return dldt

    def dl2_dp2(self, y, theta, param):
        self._validate_dln_dpn_inputs(y, theta, param)
        deriv = -(self.dl1_dp1(y, theta, param) ** 2)
        return np.clip(deriv, -np.inf, -1e-15)

    def dl2_dpp(self, y, theta, params):
        self._validate_dl2_dpp_inputs(y, theta, params)
        return -self.dl1_dp1(y, theta, params[0]) * self.dl1_dp1(y, theta, params[1])
