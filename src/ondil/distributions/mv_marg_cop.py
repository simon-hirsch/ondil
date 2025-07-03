# Author Simon Hirsch
# MIT Licence
from typing import Dict

import numpy as np
import scipy.special as sp
import scipy.stats as st

from ..base import Distribution, LinkFunction, MultivariateDistributionMixin
from ..types import ParameterShapes


class MarginalCopula(MultivariateDistributionMixin, Distribution):

    corresponding_gamlss: str = None
    parameter_names = {0: "marginal_I", 1: "marginal_II", 2: "copula"}
    parameter_support = {0: (-np.inf, np.inf), 1: (-np.inf, np.inf), 2: (-1, 1)}
    distribution_support = (-np.inf, np.inf)
    parameter_shape = {
        0: ParameterShapes.VECTOR,
        1: ParameterShapes.VECTOR,
        2: ParameterShapes.SCALAR,
    }

    def __init__(
        self,
        marginal_I,
        marginal_II,
        copula,
    ):
        super().__init__(
            distributions={
                0: marginal_I,
                1: marginal_II,
                2: copula,
            }
        )
        self.is_multivariate = True
        self._adr_lower_diag = {0: False, 1: False, 2: False}
        self._regularization_allowed = {0: False, 1: False, 2: False}
        self._regularization = "adr"  # or adr
        self._scoring = "fisher"

    @staticmethod
    def fitted_elements(self, dim: int):
        return {0: self.marginal_I.parameter_support , 1: self.marginal_II.parameter_support, 2: self.copula.parameter_support} # depends on the marginal distribution here
    
    @property
    def param_structure(self):
        return self._param_structure

    @staticmethod
    def index_flat_to_cube(k: int, d: int, param: int):
        #if (param == 0) | (param == 2):
            return k
        #if param == 1:
            # tril_indicies is row-wise
            # "inverted" triu_indicies is column-wise
            #i, j = np.triu_indices(d)
            #return j[k], i[k]

    @staticmethod
    def set_theta_element(theta: Dict, value: np.ndarray, param: int, k: int) -> Dict:
        """Sets an element of theta for parameter param and place k.

        !!! Note
            This will mutate `theta`!

        Args:
            theta (Dict): Current fitted $\theta$
            value (np.ndarray): Value to set
            param (int): Distribution parameter
            k (int): Flat element index $k$

        Returns:
            Dict: Theta where element (param, k) is set to value.
        """
       # if (param == 0) | (param == 2):
        theta[param][:, k] = value
       # if param == 1:
       #     d = theta[0].shape[1]
       #     i, j = np.triu_indices(d)
       #     theta[param][:, j[k], i[k]] = value
        return theta

    def theta_to_params(self, theta):
        marg_1 = theta[0]
        marg_2 = theta[1]
        copula = theta[2].squeeze()
        return marg_1, marg_2, copula

    def dl1_dp1(self, y: np.ndarray, theta: Dict, param: int = 0):
        raise NotImplementedError("Not implemented")

    def dl2_dp2(self, y: np.ndarray, theta: Dict, param: int = 0):
        raise NotImplementedError("Not implemented")

    def element_score(self, y: np.ndarray, theta: Dict, param: int = 0, k: int = 0):
        return self.element_dl1_dp1(y=y, theta=theta, param=param, k=k)

    def element_hessian(self, y: np.ndarray, theta: Dict, param: int = 0, k: int = 0):
        return self.element_dl2_dp2(y=y, theta=theta, param=param, k=k)

    def link_function(self, y, param=0):
        return self.links[param].link(y)

    def link_inverse(self, y, param=0):
        return self.links[param].inverse(y)

    def link_function_derivative(self, y, param=0):
        return self.links[param].link_derivative(y)

    def link_function_second_derivative(self, y, param=0):
        return self.links[param].link_second_derivative(y)

    def link_inverse_derivative(self, y, param=0):
        return self.links[param].inverse_derivative(y)

    def element_dl1_dp1(
        self, y: np.ndarray, theta: Dict, param: int = 0, k: int = 0, clip: bool = False
    ):
        marg_1, marg_2, copula = self.theta_to_params(theta)
        if param == 0:
            deriv = self.distribution[param].dl1_dp1(y, marg_1[:,k], param=k)
        if param == 1:
            deriv = self.distribution[param].dl1_dp1(y, marg_2[:,k], param=k)
        if param == 2:
            deriv = self.distribution[param].dl1_dp1(y, copula, param=0)
        return deriv

    def element_dl2_dp2(
        self, y: np.ndarray, theta: Dict, param: int = 0, k: int = 0, clip: bool = True
    ):
        marg_1, marg_2, copula = self.theta_to_params(theta)
        if param == 0:
            deriv = self.distribution[param].dl2_dp2(y, marg_1[:,k], param=k)
        if param == 1:
            deriv = self.distribution[param].dl2_dp2(y, marg_2[:,k], param=k)
        if param == 2:
            deriv = self.distribution[param].dl2_dp2(y, copula, param=0)
        if clip:
            deriv = np.clip(deriv, -np.inf, -1e-5)

        return deriv

    def dl2_dpp(self, y: np.ndarray, theta: Dict, param: int = 0):
        raise NotImplementedError("Not implemented.")

    def element_link_function(
        self, y: np.ndarray, param: int = 0, k: int = 0, d: int = 0
    ) -> np.ndarray:
        return self.distribution[param].links[k](y)
  
    def element_link_function_derivative(
        self, y: np.ndarray, param: int = 0, k: int = 0, d: int = 0
    ) -> np.ndarray:
        return self.distribution[param].links[k].link_derivative(y)

    def element_link_function_second_derivative(
        self, y: np.ndarray, param: int = 0, k: int = 0, d: int = 0
    ) -> np.ndarray:
        return self.distribution[param].links[k].link_second_derivative(y)

    def element_link_inverse(
        self, y: np.ndarray, param: int = 0, k: int = 0, d: int = 0
    ) -> np.ndarray:
        return self.distribution[param].links[k].inverse(y)

    def element_link_inverse_derivative(
        self, y: np.ndarray, param: int = 0, k: int = 0, d: int = 0
    ) -> np.ndarray:
        return self.distribution[param].links[k].inverse_derivative(y)

    def initial_values(self, y, param):
            return self.distribution[param].intial_values(y)
   
    def set_initial_guess(self, theta, param):
        if param < 2:
            return theta
        if param == 2:
            #theta[2] = np.full_like(theta[2], 5)
            return theta

    def cube_to_flat(self, x: np.ndarray, param: int):
        #if (param == 0) | (param == 2):
            return x
        #if param == 1:
          #  d = x.shape[1]
            #i = np.triu_indices(d)
           # out = x[:, i[1], i[0]]
            #return out

    def flat_to_cube(self, x: np.ndarray, param: int):
       # if (param == 0) | (param == 2):
            return x
       # if param == 1:
            n, k = x.shape
            # The following conversion holds for upper diagonal matrices
            # We INCLUDE the diagonal!!
            # (D + 1) * D // 2 = k
            # (D + 1) * D = 2 * k
            # D^2 + D = 2 * k
            # ... Wolfram Alpha ...
            # D = 0.5 * (sqrt(8k + 1) - 1)
            #d = int(1 / 2 * (np.sqrt(8 * k + 1) - 1))
            #i = np.triu_indices(d)
            #out = np.zeros((n, d, d))
           # out[:, i[1], i[0]] = x
            #return out

    def log_likelihood(self, y: np.ndarray, theta: Dict[int, np.ndarray]):
        marg_1, marg_2, copula = self.theta_to_params(theta)
        return batched_log_likelihood(y, marg_1, marg_2, copula)

    def cdf(self, y, theta):
        raise NotImplementedError("Not implemented")

    def pdf(self, y, theta):
        raise NotImplementedError("Not implemented")

    def ppf(self, q, theta):
        raise NotImplementedError("Not implemented")

    def rvs(self, size, theta):
        raise NotImplementedError("Not implemented")

    def logcdf(self, y, theta):
        raise NotImplementedError("Not implemented")

    def logpdf(self, y, theta):
        raise NotImplementedError("Not implemented")

    def logpmf(self, y, theta):
        raise NotImplementedError("Not implemented")

    def pmf(self, y, theta):
        raise NotImplementedError("Not implemented")

    def calculate_conditional_initial_values(
        self, y: np.ndarray, theta: Dict[int, np.ndarray]
    ) -> Dict[int, np.ndarray]:
        raise NotImplementedError("Not implemented")


def batched_log_likelihood(self, y, marg_1, marg_2, copula):
    return (
          self.distribution[0].logpdf(y, marg_1)
        + self.distribution[1].logpdf(y, marg_2)
        + self.distribution[2].log_likelihood(y, copula)  
    )



