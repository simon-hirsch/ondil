from typing import Tuple

import numpy as np
import scipy.stats as st

from ..base import Distribution, LinkFunction, ScipyMixin
from ..links import Identity, Log
from ..types import ParameterShapes


class JSU(ScipyMixin, Distribution):
    r"""
    Corresponds to GAMLSS JSUo() and scipy.stats.johnsonsu()

    Distribution parameters:
    0 : Location
    1 : Scale (close to standard deviation)
    2 : Skewness
    3 : Tail behaviour
    """

    corresponding_gamlss: str = "JSUo"

    parameter_names = {0: "mu", 1: "sigma", 2: "nu", 3: "tau"}
    parameter_support = {
        0: (-np.inf, np.inf),
        1: (np.nextafter(0, 1), np.inf),
        2: (-np.inf, np.inf),
        3: (np.nextafter(0, 1), np.inf),
    }
    parameter_shape = {
        0: ParameterShapes.SCALAR,
        1: ParameterShapes.SCALAR,
        2: ParameterShapes.SCALAR,
        3: ParameterShapes.SCALAR,
    }
    distribution_support = (-np.inf, np.inf)

    # Scipy equivalent and parameter mapping ondil -> scipy
    scipy_dist = st.johnsonsu
    scipy_names = {"mu": "loc", "sigma": "scale", "nu": "a", "tau": "b"}

    def __init__(
        self,
        loc_link: LinkFunction = Identity(),
        scale_link: LinkFunction = Log(),
        skew_link: LinkFunction = Identity(),
        tail_link: LinkFunction = Log(),
        use_gamlss_init_values: bool = False,
    ) -> None:
        super().__init__(
            links={
                0: loc_link,
                1: scale_link,
                2: skew_link,
                3: tail_link,
            }
        )
        self.gamlss_init_values = use_gamlss_init_values

    def dl1_dp1(self, y: np.ndarray, theta: np.ndarray, param: int = 0) -> np.ndarray:
        self._validate_dln_dpn_inputs(y, theta, param)
        mu, sigma, nu, tau = self.theta_to_params(theta)

        if param == 0:
            # MU
            z = (y - mu) / sigma
            r = nu + tau * np.arcsinh(z)
            dldm = (z / (sigma * (z * z + 1))) + (
                (r * tau) / (sigma * np.sqrt(z * z + 1))
            )
            return dldm

        if param == 1:
            # SIGMA
            z = (y - mu) / sigma
            r = nu + tau * np.arcsinh(z)
            dldd = (-1 / (sigma * (z * z + 1))) + (
                (r * tau * z) / (sigma * np.sqrt(z * z + 1))
            )
            return dldd

        if param == 2:
            # nu
            z = (y - mu) / sigma
            r = nu + tau * np.arcsinh(z)
            dldv = -r
            return dldv

        if param == 3:
            # tau
            z = (y - mu) / sigma
            r = nu + tau * np.arcsinh(z)
            dldt = (1 + r * nu - r * r) / tau
            return dldt

    def dl2_dp2(self, y: np.ndarray, theta: np.ndarray, param: int = 0) -> np.ndarray:
        self._validate_dln_dpn_inputs(y, theta, param)
        mu, sigma, nu, tau = self.theta_to_params(theta)
        if param == 0:
            # MU
            z = (y - mu) / sigma
            r = nu + tau * np.arcsinh(z)
            dldm = (z / (sigma * (z * z + 1))) + (
                (r * tau) / (sigma * np.sqrt(z * z + 1))
            )
            d2ldm2 = -dldm * dldm
            return d2ldm2

        if param == 1:
            # SIGMA
            z = (y - mu) / sigma
            r = nu + tau * np.arcsinh(z)
            dldd = (-1 / (sigma * (z * z + 1))) + (
                (r * tau * z) / (sigma * np.sqrt(z * z + 1))
            )
            d2ldd2 = -(dldd * dldd)
            return d2ldd2

        if param == 2:
            # TAIL
            z = (y - mu) / sigma
            r = nu + tau * np.arcsinh(z)
            d2ldv2 = -(r * r)
            return d2ldv2

        if param == 3:
            z = (y - mu) / sigma
            r = nu + tau * np.arcsinh(z)
            dldt = (1 + r * nu - r * r) / tau
            d2ldt2 = -dldt * dldt
            return d2ldt2

    def dl2_dpp(
        self, y: np.ndarray, theta: np.ndarray, params: Tuple[int, int] = (0, 1)
    ) -> np.ndarray:
        self._validate_dl2_dpp_inputs(y, theta, params)
        mu, sigma, nu, tau = self.theta_to_params(theta)
        if sorted(params) == [0, 1]:
            z = (y - mu) / sigma
            r = nu + tau * np.arcsinh(z)
            dldm = (z / (sigma * (z * z + 1))) + (
                (r * tau) / (sigma * np.sqrt(z * z + 1))
            )
            dldd = (-1 / (sigma * (z * z + 1))) + (
                (r * tau * z) / (sigma * np.sqrt(z * z + 1))
            )
            d2ldmdd = -(dldm * dldd)
            return d2ldmdd

        if sorted(params) == [0, 2]:
            z = (y - mu) / sigma
            r = nu + tau * np.arcsinh(z)
            dldm = (z / (sigma * (z * z + 1))) + (
                (r * tau) / (sigma * np.sqrt(z * z + 1))
            )
            dldv = -r
            d2ldmdv = -(dldm * dldv)
            return d2ldmdv

        if sorted(params) == [0, 3]:
            z = (y - mu) / sigma
            r = nu + tau * np.arcsinh(z)
            dldm = (z / (sigma * (z * z + 1))) + (
                (r * tau) / (sigma * np.sqrt(z * z + 1))
            )
            dldt = (1 + r * nu - r * r) / tau
            d2ldmdt = -(dldm * dldt)
            return d2ldmdt

        if sorted(params) == [1, 2]:
            z = (y - mu) / sigma
            r = nu + tau * np.arcsinh(z)
            dldd = (-1 / (sigma * (z * z + 1))) + (
                (r * tau * z) / (sigma * np.sqrt(z * z + 1))
            )
            dldv = -r
            d2ldddv = -(dldd * dldv)
            return d2ldddv

        if sorted(params) == [1, 3]:
            z = (y - mu) / sigma
            r = nu + tau * np.arcsinh(z)
            dldd = (-1 / (sigma * (z * z + 1))) + (
                (r * tau * z) / (sigma * np.sqrt(z * z + 1))
            )
            dldt = (1 + r * nu - r * r) / tau
            d2ldddt = -(dldd * dldt)
            return d2ldddt

        if sorted(params) == [2, 3]:
            z = (y - mu) / sigma
            r = nu + tau * np.arcsinh(z)
            dldv = -r
            dldt = (1 + r * nu - r * r) / tau
            d2ldvdt = -(dldv * dldt)
            return d2ldvdt

    def initial_values(self, y: np.ndarray) -> np.ndarray:
        out = np.empty((y.shape[0], self.n_params))
        if self.gamlss_init_values:
            out[:, 0] = (np.repeat(np.mean(y), y.shape[0]) + y) / 2
            out[:, 1] = 0.1
            out[:, 2] = 0.0
            out[:, 3] = 0.5
        else:
            params = st.johnsonsu.fit(y)
            out[:, 0] = params[2]
            out[:, 1] = params[3]
            out[:, 2] = params[0]
            out[:, 3] = params[1]
        return out
