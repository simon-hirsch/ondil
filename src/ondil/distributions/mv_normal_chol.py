# Author Simon Hirsch
# MIT Licence
from typing import Dict

import numba as nb
import numpy as np
import scipy.stats as st

from ..base import Distribution, LinkFunction, MultivariateDistributionMixin
from ..links import Identity, Log, MatrixDiagTriu
from ..types import ParameterShapes


class MultivariateNormalInverseCholesky(MultivariateDistributionMixin, Distribution):
    # The cholesky decomposition of
    # COV = L @ L.T
    # PRC = (L^-1).T @ (L^-1)
    # The cross derivatives are defined on (L^-1).T
    # The transpose is important!!

    """
    The multivariate normal (Gaussian) distribution parameterized by the inverse Cholesky factor of the precision (inverse scale) matrix.

    The PDF of the multivariate normal distribution is given by:
    $$
    p(y \\mid \\mu, L) =
    |L| \\cdot (2\\pi)^{-k/2}
    \\exp\\left(-\\frac{1}{2} (y - \\mu)^T (L L^T) (y - \\mu)\\right)
    $$

    where \\( k \\) is the dimensionality of the data, \\( \\mu \\) is the location parameter, and \\( L \\) is the inverse Cholesky factor of the precision matrix (so the precision is \\( L L^T \\)).
    """

    corresponding_gamlss: str = None
    parameter_names = {0: "mu", 1: "sigma"}
    parameter_support = {0: (-np.inf, np.inf), 1: (-np.inf, np.inf)}
    distribution_support = (-np.inf, np.inf)
    scipy_dist = st.multivariate_normal
    scipy_names = {"mu": "mean", "sigma": "cov"}
    parameter_shape = {
        0: ParameterShapes.VECTOR,
        1: ParameterShapes.UPPER_TRIANGULAR_MATRIX,
    }

    def __init__(
        self,
        loc_link: LinkFunction = Identity(),
        scale_link: LinkFunction = MatrixDiagTriu(Log(), Identity()),
    ):
        super().__init__(
            links={
                0: loc_link,
                1: scale_link,
            }
        )
        self._regularization_allowed = {0: False, 1: True}

    @staticmethod
    def fitted_elements(dim: int):
        return {0: dim, 1: int(dim * (dim + 1) // 2)}

    @staticmethod
    def index_flat_to_cube(k: int, d: int, param: int):
        if param == 0:
            return k
        if param == 1:
            i, j = np.triu_indices(d)
            return i[k], j[k]

    def get_adr_regularization_distance(self, dim: int, param: int) -> float:
        if param == 0:
            return None
        if param == 1:
            i, j = np.triu_indices(dim, k=0)
            return np.abs(i - j)

    def get_regularization_size(self, dim: int) -> int:
        return dim

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
        if param == 0:
            theta[param][:, k] = value
        if param == 1:
            d = theta[0].shape[1]
            i, j = np.triu_indices(d)
            theta[param][:, i[k], j[k]] = value
        return theta

    def theta_to_params(self, theta):
        fitted_loc = theta[0]
        fitted_inv_tr_chol = theta[1]
        return fitted_loc, fitted_inv_tr_chol

    def set_initial_guess(
        self,
        y: np.ndarray,
        theta: Dict[int, np.ndarray],
        param: int,
    ) -> Dict[int, np.ndarray]:
        if param == 0:
            return theta
        if param == 1:
            residual = y - theta[0]
            cov = np.cov(residual, rowvar=False)
            chol = np.linalg.inv(np.linalg.cholesky(cov)).T
            theta[1] = np.tile(chol, (y.shape[0], 1, 1))
            return theta

    def dl1_dp1(self, y: np.ndarray, theta: Dict, param: int = 0):
        """Return the first derivatives wrt to the parameter.

        !!! Note
            We expect the fitted L^-1)^T to be handed in matrix/cube form, i.e of shape n x d x d.
            But we return the derivatives in flat format.

        Args:
            y (np.ndarray): Y values of shape n x d
            theta (Dict): Dict with {0 : fitted mu, 1 : fitted (L^-1)^T}
            param (int, optional): Which parameter derivatives to return. Defaults to 0.

        Returns:
            derivative (np.ndarray): The 1st derivatives.
        """
        fitted_loc, fitted_inv_tr_chol = self.theta_to_params(theta)

        if param == 0:
            deriv = _derivative_1st_mu(
                y=y, fitted_loc=fitted_loc, fitted_inv_tr_chol=fitted_inv_tr_chol
            )
        if param == 1:
            deriv = _derivative_1st_inv_chol(
                y=y, fitted_loc=fitted_loc, fitted_inv_tr_chol=fitted_inv_tr_chol
            )

        return deriv

    def dl2_dp2(
        self, y: np.ndarray, theta: Dict, param: int = 0, clip=False
    ) -> np.ndarray:
        """Return the second derivatives wrt to the parameter.

        !!! Note
            We expect the fitted L^-1)^T to be handed in matrix/cube form, i.e of shape n x d x d.
            But we return the derivatives in flat format.

        Args:
            y (np.ndarray): Y values of shape n x d
            theta (Dict): Dict with {0 : fitted mu, 1 : fitted (L^-1)^T}
            param (int, optional): Which parameter derivatives to return. Defaults to 0.

        Returns:
            derivative (np.ndarray): The 2nd derivatives.
        """
        fitted_loc, fitted_inv_tr_chol = self.theta_to_params(theta)

        if param == 0:
            deriv = _derivative_2nd_mu(
                y=y, fitted_loc=fitted_loc, fitted_inv_tr_chol=fitted_inv_tr_chol
            )
        if param == 1:
            deriv = _derivative_2nd_inv_chol(
                y=y, fitted_loc=fitted_loc, fitted_inv_tr_chol=fitted_inv_tr_chol
            )

        return deriv

    def element_score(self, y: np.ndarray, theta: Dict, param: int = 0, k: int = 0):
        return self.element_dl1_dp1(y=y, theta=theta, param=param, k=k)

    def element_hessian(self, y: np.ndarray, theta: Dict, param: int = 0, k: int = 0):
        return self.element_dl2_dp2(y=y, theta=theta, param=param, k=k)

    def element_dl1_dp1(
        self, y: np.ndarray, theta: Dict, param: int = 0, k: int = 0, clip=False
    ):
        fitted_loc, fitted_inv_tr_chol = self.theta_to_params(theta)
        d = y.shape[1]
        if param == 0:
            deriv = _derivative_1st_mu(
                y=y, fitted_loc=fitted_loc, fitted_inv_tr_chol=fitted_inv_tr_chol
            )
            deriv = deriv[:, k]
        if param == 1:
            ii, jj = np.triu_indices(d)
            if ii[k] == jj[k]:
                deriv = _derivative_1st_inv_chol_diag(
                    y, fitted_loc, fitted_inv_tr_chol, ii[k]
                )
            else:
                deriv = _derivative_1st_inv_chol_triu(
                    y, fitted_loc, fitted_inv_tr_chol, ii[k], jj[k]
                )
        return deriv

    def element_dl2_dp2(
        self, y: np.ndarray, theta: Dict, param: int = 0, k: int = 0, clip=False
    ):
        fitted_loc, fitted_inv_tr_chol = self.theta_to_params(theta)
        d = y.shape[1]
        if param == 0:
            deriv = _derivative_2nd_mu(
                y=y, fitted_loc=fitted_loc, fitted_inv_tr_chol=fitted_inv_tr_chol
            )
            deriv = deriv[:, k]
        if param == 1:
            ii, jj = np.triu_indices(d)
            if ii[k] == jj[k]:
                deriv = _derivative_2nd_inv_chol_diag(
                    y, fitted_loc, fitted_inv_tr_chol, ii[k]
                )
            else:
                deriv = _derivative_2nd_inv_chol_triu(
                    y, fitted_loc, fitted_inv_tr_chol, ii[k], jj[k]
                )
        return deriv

    def dl2_dpp(self, y: np.ndarray, theta: Dict, param: int = 0):
        raise NotImplementedError("Not implemented.")

    def element_link_function(
        self, y: np.ndarray, param: int = 0, k: int = 0, d: int = 0
    ) -> np.ndarray:
        if param == 0:
            return self.links[param].link(y)
        if param == 1:
            i, j = np.triu_indices(d)
            return self.links[param].element_link(y, i=i[k], j=j[k])

    def element_link_function_derivative(
        self, y: np.ndarray, param: int = 0, k: int = 0, d: int = 0
    ) -> np.ndarray:
        if param == 0:
            return self.links[param].link_derivative(y)
        if param == 1:
            i, j = np.triu_indices(d)
            return self.links[param].element_derivative(y, i=i[k], j=j[k])

    def element_link_function_second_derivative(
        self, y: np.ndarray, param: int = 0, k: int = 0, d: int = 0
    ) -> np.ndarray:
        if param == 0:
            return self.links[param].link_second_derivative(y)
        if param == 1:
            i, j = np.triu_indices(d)
            return self.links[param].element_link_second_derivative(y, i=i[k], j=j[k])

    def element_link_inverse(
        self, y: np.ndarray, param: int = 0, k: int = 0, d: int = 0
    ) -> np.ndarray:
        if param == 0:
            return self.links[param].inverse(y)
        if param == 1:
            i, j = np.triu_indices(d)
            return self.links[param].element_inverse(y, i=i[k], j=j[k])

    def element_link_inverse_derivative(
        self, y: np.ndarray, param: int = 0, k: int = 0, d: int = 0
    ) -> np.ndarray:
        if param == 0:
            return self.links[param].inverse_derivative(y)
        if param == 1:
            i, j = np.triu_indices(d)
            return self.links[param].element_inverse_derivative(y, i=i[k], j=j[k])

    def initial_values(self, y, param=0):
        M = y.shape[0]
        if param == 0:
            return np.tile(np.mean(y, axis=0), (M, 1))
        if param == 1:
            chol = np.linalg.inv(np.linalg.cholesky(np.cov(y, rowvar=False))).T
            return np.tile(chol, (M, 1, 1))

    def cube_to_flat(self, x: np.ndarray, param: int):
        if param == 0:
            return x
        if param == 1:
            d = x.shape[1]
            i = np.triu_indices(d)
            out = x[:, i[0], i[1]]
            return out

    def flat_to_cube(self, x: np.ndarray, param: int):
        if param == 0:
            return x
        if param == 1:
            n, k = x.shape
            # The following conversion holds for upper diagonal matrices
            # We INCLUDE the diagonal!!
            # (D + 1) * D // 2 = k
            # (D + 1) * D = 2 * k
            # D^2 + D = 2 * k
            # ... Wolfram Alpha ...
            # D = 0.5 * (sqrt(8k + 1) - 1)
            d = int(1 / 2 * (np.sqrt(8 * k + 1) - 1))
            i = np.triu_indices(d)
            out = np.zeros((n, d, d))
            out[:, i[0], i[1]] = x
            return out

    def param_conditional_likelihood(
        self, y: np.ndarray, theta: Dict, eta: np.ndarray, param: int
    ) -> np.ndarray:
        """Calulate the log-likelihood for (flat) eta for parameter (param)
        and theta for all other parameters.

        Args:
            y (np.ndarray): True values
            theta (Dict): Fitted theta.
            eta (np.ndarray): Fitted eta.
            param (int): Param for which we take eta.

        Returns:
            np.ndarray: Log-likelihood.
        """
        fitted = self.flat_to_cube(eta, param=param)
        fitted = self.link_inverse(fitted, param=param)
        # fitted_theta = {**theta, param: fitted_eta}
        return self.log_likelihood(y, theta={**theta, param: fitted})

    def theta_to_scipy(self, theta: Dict[int, np.ndarray]):
        out = {
            "mean": theta[0],
            "cov": np.linalg.inv(theta[1] @ theta[1].swapaxes(-1, -2)),
        }
        return out

    def cdf(self, y, theta):
        raise NotImplementedError("Not implemented")

    def pdf(self, y, theta):
        return np.exp(self.logpdf(y, theta))

    def ppf(self, q, theta):
        raise NotImplementedError("Not implemented")

    def rvs(self, size, theta):
        raise NotImplementedError("Not implemented")

    def logcdf(self, y, theta):
        raise NotImplementedError("Not implemented")

    def logpdf(self, y, theta):
        return _log_likelihood(y, theta[0], theta[1])

    def logpmf(self, y, theta):
        raise NotImplementedError("Not implemented")

    def pmf(self, y, theta):
        raise NotImplementedError("Not implemented")

    def calculate_conditional_initial_values(
        self, y: np.ndarray, theta: Dict[int, np.ndarray]
    ) -> Dict[int, np.ndarray]:
        raise NotImplementedError("Not implemented")


##########################################################
### numba JIT-compiled functions for the derivatives #####
##########################################################


@nb.jit(nopython=True)
def _log_likelihood(y, mu, mod_chol_transpose):
    M = y.shape[0]
    k = y.shape[1]
    y_centered = y - mu
    ll = np.empty(M)
    for m in range(M):
        ll[m] = float(
            np.log(np.linalg.det(mod_chol_transpose[m]))
            - 0.5
            * (
                y_centered[m]
                @ mod_chol_transpose[m]
                @ mod_chol_transpose[m].T
                @ y_centered[m]
            )
        )
    ll -= k / 2 * np.log(np.pi * 2)
    return ll


def _derivative_1st_mu(
    y: np.ndarray, fitted_loc: np.ndarray, fitted_inv_tr_chol: np.ndarray
) -> np.ndarray:
    y_centered = y - fitted_loc
    precision = fitted_inv_tr_chol @ fitted_inv_tr_chol.transpose(0, 2, 1)
    return np.sum((y_centered[:, :, None] * precision), -1)


@nb.jit()
def _derivative_1st_inv_chol_diag(y, fitted_loc, fitted_inv_tr_chol, i):
    y_centered = y - fitted_loc
    sum_m = np.zeros(y.shape[0])
    for m in range(i + 1):
        sum_m += y_centered[:, m] * fitted_inv_tr_chol[:, m, i]
    deriv = 1 / fitted_inv_tr_chol[:, i, i] - y_centered[:, i] * sum_m
    return deriv


@nb.jit()
def _derivative_1st_inv_chol_triu(y, fitted_loc, fitted_inv_tr_chol, i, j):
    y_centered = y - fitted_loc
    sum_m = np.zeros(y.shape[0])
    for m in range(j + 1):
        sum_m += y_centered[:, m] * fitted_inv_tr_chol[:, m, j]
    deriv = -(y_centered[:, i] * sum_m)
    return deriv


@nb.jit()
def _derivative_1st_inv_chol(y, fitted_loc, fitted_inv_tr_chol):
    n = y.shape[0]
    d = y.shape[1]
    dim = ((d + 1) * d) // 2
    k = 0
    deriv = np.zeros((n, dim))
    for i in range(d):
        for j in range(i, d):
            if i == j:
                deriv[:, k] = _derivative_1st_inv_chol_diag(
                    y, fitted_loc, fitted_inv_tr_chol, i
                )
            else:
                deriv[:, k] = _derivative_1st_inv_chol_triu(
                    y, fitted_loc, fitted_inv_tr_chol, i, j
                )
            k += 1
    return deriv


@nb.jit()
def _derivative_2nd_mu(y, fitted_loc, fitted_inv_tr_chol):
    d = y.shape[1]
    deriv = np.zeros_like(y)
    for i in range(d):
        deriv[:, i] = -np.sum(fitted_inv_tr_chol[:, i, i:] ** 2, -1)
    return deriv


@nb.jit()
def _derivative_2nd_inv_chol_diag(y, fitted_loc, fitted_inv_tr_chol, i):
    y_centered = y - fitted_loc
    deriv = -1 / fitted_inv_tr_chol[:, i, i] ** 2 - y_centered[:, i] ** 2
    return deriv


# @nb.jit()
# def _derivative_2nd_inv_chol_diag(y, fitted_loc, fitted_inv_tr_chol, i):
#     y_centered = y - fitted_loc
#     sum_m = np.zeros(y.shape[0])
#     for m in range(i):
#         sum_m += y_centered[:, m] * fitted_inv_tr_chol[:, m, i]
#     deriv = (
#         -2 * fitted_inv_tr_chol[:, i, i] ** 2 * y_centered[:, i] ** 2
#         - fitted_inv_tr_chol[:, i, i] * y_centered[:, i] * sum_m
#     )
#     return deriv


@nb.jit()
def _derivative_2nd_inv_chol_triu(y, fitted_loc, fitted_inv_tr_chol, i, j):
    y_centered = y - fitted_loc
    return -(y_centered[:, i] ** 2)


@nb.jit()
def _derivative_2nd_inv_chol(y, fitted_loc, fitted_inv_tr_chol):
    n = y.shape[0]
    d = y.shape[1]
    dim = ((d + 1) * d) // 2
    k = 0
    deriv = np.zeros((n, dim))
    for i in range(d):
        for j in range(i, d):
            if i == j:
                deriv[:, k] = _derivative_2nd_inv_chol_diag(
                    y, fitted_loc, fitted_inv_tr_chol, i
                )
            else:
                deriv[:, k] = _derivative_2nd_inv_chol_triu(
                    y, fitted_loc, fitted_inv_tr_chol, i, j
                )
            k += 1
    return deriv
