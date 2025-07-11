# Author Simon Hirsch
# MIT Licence
from typing import Dict

import numpy as np
from pytest import param
import scipy.special as sp
import scipy.stats as st

from ..base import Distribution, LinkFunction , MarginalCopulaMixin
from ..types import ParameterShapes

class MarginalCopula(MarginalCopulaMixin):

    corresponding_gamlss: str = None
    parameter_names = {0: "marginal_0", 1: "marginal_1", 2: "dependence"}
    parameter_support = {0: (-np.inf, np.inf), 1: (-np.inf, np.inf), 2: (-1, 1)}
    distribution_support = (-np.inf, np.inf)
    parameter_shape = {
        0: ParameterShapes.VECTOR,
        1: ParameterShapes.VECTOR,
        2: ParameterShapes.SCALAR,
    }

    def __init__(
        self,
        marginal_0,
        marginal_1,
        dependence,
    ):
        self.marginal_0 = marginal_0
        self.marginal_1 = marginal_1
        self.dependence = dependence
        super().__init__(
            distributions={
                0: self.marginal_0,
                1: self.marginal_1,
                2: self.dependence,
            }
        )
        self.is_multivariate = True
        self._adr_lower_diag = {0: False, 1: False, 2: False}
        self._regularization_allowed = {0: False, 1: False, 2: False}
        self._regularization = ""  # or adr
        self._scoring = "fisher"
    
    def fitted_elements(self, dim):
        return {0: len(self.distributions[0].parameter_support), 1: len(self.distributions[1].parameter_support), 2: len(self.distributions[2].parameter_support)}

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
        if param < 2:
            theta[param][:, k] = value
        else:
            theta[param] = value
        return theta

    def theta_to_params(self, theta):
        marg_1 = theta[0]
        marg_2 = theta[1]
        copula = theta[2]
        return marg_1, marg_2, copula

    def dl1_dp1(self, y: np.ndarray, theta: Dict, param: int = 0):
        raise NotImplementedError("Not implemented")

    def dl2_dp2(self, y: np.ndarray, theta: Dict, param: int = 0):
        raise NotImplementedError("Not implemented")

    def element_score(self, y: np.ndarray, theta: Dict, param: int = 0, k: int = 0):
        return self.element_dl1_dp1(y=y, theta=theta, param=param, k=k)

    def element_hessian(self, y: np.ndarray, theta: Dict, param: int = 0, k: int = 0):
        return self.element_dl2_dp2(y=y, theta=theta, param=param, k=k)

    def link_function(self, y, param=0 , k=0):
        marg_1, marg_2, copula = self.theta_to_params(y)
        if param == 0:
            return self.distributions[param].links[k].link(marg_1[:,k])
        if param == 1:
            return self.distributions[param].links[k].link(marg_2[:,k])
        if param == 2:
            return self.distributions[param].links[k].link((1-1e-5)*copula)
        
    def link_inverse(self, y, param=0 , k=0):
        marg_1, marg_2, copula = self.theta_to_params(y)
        if param == 0:
            return self.distributions[param].links[k].inverse(marg_1[:,k])
        if param == 1:
            return self.distributions[param].links[k].inverse(marg_2[:,k])
        if param == 2:
            return self.distributions[param].links[k].inverse(copula)
        
    def update_link_inverse(self, y, param=0 , k=0):
        if param == 0:
            return self.distributions[param].links[k].inverse(y)
        if param == 1:
            return self.distributions[param].links[k].inverse(y)
        if param == 2:
            return self.distributions[param].links[k].inverse(y)


    def link_function_derivative(self, y, param=0 , k=0):
        marg_1, marg_2, copula = self.theta_to_params(y)
        if param == 0:
            return self.distributions[param].links[k].link_derivative(marg_1[:,k])
        if param == 1:
            return self.distributions[param].links[k].link_derivative(marg_2[:,k])
        if param == 2:
            return self.distributions[param].links[k].link_derivative(copula)

    def link_function_second_derivative(self, y, param=0 , k=0):
        marg_1, marg_2, copula = self.theta_to_params(y)
        if param == 0:
            return self.distributions[param].links[k].link_second_derivative(marg_1[:,k])
        if param == 1:
            return self.distributions[param].links[k].link_second_derivative(marg_2[:,k])
        if param == 2:
            return self.distributions[param].links[k].link_second_derivative(copula)
        
    def link_inverse_derivative(self, y, param=0 , k=0):
        marg_1, marg_2, copula = self.theta_to_params(y)
        if param == 0:
            return self.distributions[param].links[k].inverse_derivative(marg_1[:,k])
        if param == 1:
            return self.distributions[param].links[k].inverse_derivative(marg_2[:,k])
        if param == 2:
            return self.distributions[param].links[k].inverse_derivative(copula)

    def element_dl1_dp1(
        self, y: np.ndarray, theta: Dict, param: int = 0, k: int = 0, clip: bool = False
    ):
        marg_1, marg_2, copula = self.theta_to_params(theta)
        if param == 0:
            deriv = self.distributions[param].dl1_dp1(y[:,0], marg_1, param=k)
        if param == 1:
            deriv = self.distributions[param].dl1_dp1(y[:,1], marg_2, param=k)
        if param == 2:
            y_transformed_0= self.distributions[0].cdf(y[:,0], marg_1)
            y_transformed_1= self.distributions[1].cdf(y[:,1], marg_2)
            y_transformed = np.column_stack([y_transformed_0, y_transformed_1])
            deriv = self.distributions[param].dl1_dp1(y_transformed, copula, param=0)

        return deriv

    def element_dl2_dp2(
        self, y: np.ndarray, theta: Dict, param: int = 0, k: int = 0, clip: bool = True
    ):
        marg_1, marg_2, copula = self.theta_to_params(theta)
        if param == 0:
            deriv = self.distributions[param].dl2_dp2(y[:,0], marg_1, param=k)
        if param == 1:
            deriv = self.distributions[param].dl2_dp2(y[:,1], marg_2, param=k)
        if param == 2:
            y_transformed_0= self.distributions[0].cdf(y[:,0],marg_1)
            y_transformed_1= self.distributions[1].cdf(y[:,1],marg_2)
            y_transformed = np.column_stack([y_transformed_0, y_transformed_1])
            deriv = self.distributions[param].dl2_dp2(y_transformed, copula, param=0)
        if clip:
            deriv = np.clip(deriv, -np.inf, -1e-5)

        return deriv

    def dl2_dpp(self, y: np.ndarray, theta: Dict, param: int = 0):
        raise NotImplementedError("Not implemented.")

    def element_link_function(
        self, y: np.ndarray, param: int = 0, k: int = 0, d: int = 0
    ) -> np.ndarray:
        return self.distributions[param].links[k](y)
  
    def element_link_function_derivative(
        self, y: np.ndarray, param: int = 0, k: int = 0, d: int = 0
    ) -> np.ndarray:
        return self.distributions[param].links[k].link_derivative(y)

    def element_link_function_second_derivative(
        self, y: np.ndarray, param: int = 0, k: int = 0, d: int = 0
    ) -> np.ndarray:
        return self.distributions[param].links[k].link_second_derivative(y)

    def element_link_inverse(
        self, y: np.ndarray, param: int = 0, k: int = 0, d: int = 0
    ) -> np.ndarray:
        return self.distributions[param].links[k].inverse(y)

    def element_link_inverse_derivative(
        self, y: np.ndarray, param: int = 0, k: int = 0, d: int = 0
    ) -> np.ndarray:
        return self.distributions[param].links[k].inverse_derivative(y)

    def initial_values(self, y, param):
            return self.distributions[param].initial_values(y)
   
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

    def logpdf(self, y: np.ndarray, theta: Dict[int, np.ndarray]):
        marg_1, marg_2, copula = self.theta_to_params(theta)
        return batched_log_likelihood(self, y, marg_1, marg_2, copula)



    def theta_to_scipy(self, theta: Dict[int, np.ndarray]):
        out = {
            "marginal_1": theta[0],
            "marginal_2": theta[1],
            "dependence": theta[2],
        }
        return out


    def cdf(self, y, theta):
        raise NotImplementedError("Not implemented")

    def pdf_test(self, y, theta, param=0):
        marg_1, marg_2, copula = self.theta_to_params(theta)
        y_transformed_0 = self.distributions[0].cdf(y[:, 0], marg_1)
        y_transformed_1 = self.distributions[1].cdf(y[:, 1], marg_2)
        y_transformed = np.column_stack([y_transformed_0, y_transformed_1])
        return self.distributions[param].pdf(y_transformed, copula)

    def pdf(self, q, theta):
        raise NotImplementedError("Not implemented")


    def ppf(self, q, theta):
        raise NotImplementedError("Not implemented")

    def rvs(self, size, theta):
        raise NotImplementedError("Not implemented")

    def logcdf(self, y, theta):
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
    y_transformed_0= self.distributions[0].cdf(y[:,0], marg_1)
    y_transformed_1= self.distributions[1].cdf(y[:,1], marg_2)
    y_transformed = np.column_stack([y_transformed_0, y_transformed_1])
    return (
        self.distributions[0].logpdf(y[:, 0], marg_1)
        + self.distributions[1].logpdf(y[:, 1], marg_2)
        + self.distributions[2].logpdf(y_transformed, copula)
    )


