# Author Simon Hirsch
# MIT Licence
from typing import Dict

import numpy as np
import scipy.special as sp
import scipy.stats as st

from ..base import Distribution, LinkFunction, MultivariateDistributionMixin
from ..links import Identity, Log, LogShiftTwo, MatrixDiagTril
from ..types import ParameterShapes


class MultivariateStudentTInverseCholesky(MultivariateDistributionMixin, Distribution):
    # Covariance Matrix SIGMA
    # Precision Matrix OMEGA
    # Cholesky Factor inv(chol(SIGMA))
    # Lower diagonal matrix
    """
    The multivariate \\( t \\)-distribution parameterized by the inverse Cholesky factor of the precision (inverse scale) matrix.

    The PDF of the multivariate \\( t \\)-distribution is given by:
    $$
    p(y \mid \\mu, L, \\nu) =
    \\frac{\\Gamma\\left(\\frac{\\nu + k}{2}\\right)}
         {\\Gamma\\left(\\frac{\\nu}{2}\\right) \\left(\\pi \\nu\\right)^{k/2}}
    \\cdot |L|
    \\left(1 + \\frac{1}{\\nu} (y - \\mu)^T (L L^T) (y - \\mu)\\right)^{-\\frac{\\nu + k}{2}}
    $$

    where \\( k \\) is the dimensionality of the data, \\( \\mu \\) is the location parameter, \\( L \\) is the inverse Cholesky factor of the precision matrix (so the precision is \\( L L^T \\)), and \\( \\nu \\) is the degrees of freedom.
    """

    corresponding_gamlss: str = None
    parameter_names = {0: "mu", 1: "sigma", 2: "nu"}
    parameter_support = {0: (-np.inf, np.inf), 1: (-np.inf, np.inf), 2: (0, np.inf)}
    distribution_support = (-np.inf, np.inf)
    scipy_dist = st.multivariate_t
    scipy_names = {"mu": "loc", "sigma": "shape", "nu": "dof"}
    parameter_shape = {
        0: ParameterShapes.VECTOR,
        1: ParameterShapes.LOWER_TRIANGULAR_MATRIX,
        2: ParameterShapes.SCALAR,
    }

    def __init__(
        self,
        loc_link: LinkFunction = Identity(),
        scale_link: LinkFunction = MatrixDiagTril(Log(), Identity()),
        tail_link: LinkFunction = LogShiftTwo(),
        dof_guesstimate: float = 10.0,
    ):
        super().__init__(
            links={
                0: loc_link,
                1: scale_link,
                2: tail_link,
            }
        )
        self.is_multivariate = True
        self.dof_guesstimate = dof_guesstimate
        self.dof_independence = 1e6
        self._regularization_allowed = {0: False, 1: True, 2: False}

    @staticmethod
    def fitted_elements(dim: int):
        return {0: dim, 1: int(dim * (dim + 1) // 2), 2: 1}

    @property
    def param_structure(self):
        return self._param_structure

    def get_adr_regularization_distance(self, dim: int, param: int) -> float:
        if param in (0, 2):
            return None
        if param == 1:
            j, i = np.triu_indices(dim, k=0)
            return np.abs(i - j)

    def get_regularization_size(self, dim: int) -> int:
        return dim

    @staticmethod
    def index_flat_to_cube(k: int, d: int, param: int):
        if (param == 0) | (param == 2):
            return k
        if param == 1:
            # tril_indicies is row-wise
            # "inverted" triu_indicies is column-wise
            i, j = np.triu_indices(d)
            return j[k], i[k]

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
        if (param == 0) | (param == 2):
            theta[param][:, k] = value
        if param == 1:
            d = theta[0].shape[1]
            i, j = np.triu_indices(d)
            theta[param][:, j[k], i[k]] = value
        return theta

    def theta_to_params(self, theta):
        loc = theta[0]
        chol = theta[1]
        dof = theta[2].squeeze()
        return loc, chol, dof

    def theta_to_scipy(self, theta: Dict[int, np.ndarray]):
        out = {
            "loc": theta[0],
            "shape": np.linalg.inv(theta[1].swapaxes(-1, -2) @ theta[1]),
            "df": theta[2],
        }
        return out

    def dl1_dp1(self, y: np.ndarray, theta: Dict, param: int = 0):
        raise NotImplementedError("Not implemented")

    def dl2_dp2(self, y: np.ndarray, theta: Dict, param: int = 0):
        raise NotImplementedError("Not implemented")

    def element_score(self, y: np.ndarray, theta: Dict, param: int = 0, k: int = 0):
        return self.element_dl1_dp1(y=y, theta=theta, param=param, k=k)

    def element_hessian(self, y: np.ndarray, theta: Dict, param: int = 0, k: int = 0):
        return self.element_dl2_dp2(y=y, theta=theta, param=param, k=k)

    def element_dl1_dp1(
        self, y: np.ndarray, theta: Dict, param: int = 0, k: int = 0, clip: bool = False
    ):
        mu, chol, dof = self.theta_to_params(theta)
        if param == 0:
            deriv = first_partial_mu(y, mu, chol, dof, i=k)
        if param == 1:
            ## Weird notation in the Muchinsky Paper
            ii, jj = np.triu_indices(y.shape[1])
            deriv = first_partial_cov(y, mu, chol, dof, i=ii[k], j=jj[k])
        if param == 2:
            deriv = first_partial_dof(y, mu, chol, dof)
        return deriv

    def element_dl2_dp2(
        self, y: np.ndarray, theta: Dict, param: int = 0, k: int = 0, clip: bool = True
    ):
        mu, chol, dof = self.theta_to_params(theta)
        if param == 0:
            deriv = second_partial_mu(y, mu, chol, dof, i=k)
        if param == 1:
            ii, jj = np.triu_indices(y.shape[1])
            ## Weird notation in the Muchinsky Paper
            deriv = second_partial_cov(y, mu, chol, dof, i=ii[k], j=jj[k])
        if param == 2:
            deriv = second_partial_dof(y, mu, chol, dof)

        if clip:
            deriv = np.clip(deriv, -np.inf, -1e-5)

        return deriv

    def dl2_dpp(self, y: np.ndarray, theta: Dict, param: int = 0):
        raise NotImplementedError("Not implemented.")

    def element_link_function(
        self, y: np.ndarray, param: int = 0, k: int = 0, d: int = 0
    ) -> np.ndarray:
        if (param == 0) | (param == 2):
            return self.links[param].link(y)
        if param == 1:
            i, j = np.triu_indices(d)
            return self.links[param].element_link(y, i=j[k], j=i[k])

    def element_link_function_derivative(
        self, y: np.ndarray, param: int = 0, k: int = 0, d: int = 0
    ) -> np.ndarray:
        if (param == 0) | (param == 2):
            return self.links[param].link_derivative(y)
        if param == 1:
            i, j = np.triu_indices(d)
            return self.links[param].element_link_derivative(y, i=j[k], j=i[k])

    def element_link_function_second_derivative(
        self, y: np.ndarray, param: int = 0, k: int = 0, d: int = 0
    ) -> np.ndarray:
        if (param == 0) | (param == 2):
            return self.links[param].link_second_derivative(y)
        if param == 1:
            i, j = np.triu_indices(d)
            return self.links[param].element_link_second_derivative(y, i=j[k], j=i[k])

    def element_link_inverse(
        self, y: np.ndarray, param: int = 0, k: int = 0, d: int = 0
    ) -> np.ndarray:
        if (param == 0) | (param == 2):
            return self.links[param].inverse(y)
        if param == 1:
            i, j = np.triu_indices(d)
            return self.links[param].element_inverse(y, i=j[k], j=i[k])

    def element_link_inverse_derivative(
        self, y: np.ndarray, param: int = 0, k: int = 0, d: int = 0
    ) -> np.ndarray:
        if (param == 0) | (param == 2):
            return self.links[param].inverse_derivative(y)
        if param == 1:
            i, j = np.triu_indices(d)
            return self.links[param].element_inverse_derivative(y, i=j[k], j=i[k])

    def initial_values(self, y, param=0):
        M = y.shape[0]
        if param == 0:
            return np.tile(np.mean(y, axis=0), (M, 1))
        if param == 1:
            var = np.var(y, axis=0)
            shape = np.diag(var * (self.dof_guesstimate - 2) / self.dof_guesstimate)
            chol = np.linalg.inv(np.linalg.cholesky(shape))
            return np.tile(chol, (M, 1, 1))
        if param == 2:
            return np.full((y.shape[0], 1), self.dof_independence)

    def set_initial_guess(
        self,
        y: np.ndarray,
        theta: Dict[int, np.ndarray],
        param: int,
    ) -> Dict[int, np.ndarray]:
        if param == 1:
            M = y.shape[0]
            residuals = y - theta[0]
            var = np.var(residuals, axis=0)
            shape = np.diag(var * (self.dof_guesstimate - 2) / self.dof_guesstimate)
            chol = np.linalg.inv(np.linalg.cholesky(shape))
            theta[1] = np.tile(chol, (M, 1, 1))
        if param == 2:
            theta[2] = np.full_like(theta[2], self.dof_guesstimate)

        return theta

    def cube_to_flat(self, x: np.ndarray, param: int):
        if (param == 0) | (param == 2):
            return x
        if param == 1:
            d = x.shape[1]
            i = np.triu_indices(d)
            out = x[:, i[1], i[0]]
            return out

    def flat_to_cube(self, x: np.ndarray, param: int):
        if (param == 0) | (param == 2):
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
            out[:, i[1], i[0]] = x
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
        loc, chol, dof = self.theta_to_params(theta)
        return batched_log_likelihood(y, loc, chol, dof)

    def logpmf(self, y, theta):
        raise NotImplementedError("Not implemented")

    def pmf(self, y, theta):
        raise NotImplementedError("Not implemented")

    def calculate_conditional_initial_values(
        self, y: np.ndarray, theta: Dict[int, np.ndarray]
    ) -> Dict[int, np.ndarray]:
        raise NotImplementedError("Not implemented")


def batched_log_likelihood(y, mu, chol, dof):
    D = y.shape[1]
    A = np.squeeze(sp.gammaln((dof + D) / 2))
    B = np.squeeze(sp.gammaln(dof / 2))
    C = 1 / 2 * D * np.squeeze(np.log(np.pi * dof))
    Z = np.sum(np.einsum("ndk, nk -> nd", chol, (y - mu)) ** 2, axis=1)
    return (
        A
        - B
        - C
        + np.log(np.linalg.det(chol))
        - np.squeeze((dof + D) / 2) * np.log((1 + np.squeeze((1 / dof)) * Z))
    )


def first_partial_mu(y, mu, chol, dof, i):
    k = y.shape[1]
    Z = np.sum(np.einsum("ndk, nk -> nd", chol, (y - mu)) ** 2, axis=1)
    precision = chol.transpose(0, 2, 1) @ chol
    part1 = (k + dof) / (2 * (Z + dof))
    part2 = 2 * np.sum(precision[:, i, :] * (y - mu), axis=1)
    return part1 * part2


def second_partial_mu(y, mu, chol, dof, i):
    k = y.shape[1]
    Z = np.sum(np.einsum("ndk, nk -> nd", chol, (y - mu)) ** 2, axis=1)
    precision = chol.transpose(0, 2, 1) @ chol

    deriv_1 = 2 * np.sum(precision[:, i, :] * (y - mu), axis=1)
    deriv_2 = 2 * precision[:, i, i]

    part1 = k + dof
    part2 = (Z + dof) * deriv_2 - deriv_1**2
    part3 = 2 * (Z + dof) ** 2

    return -(part1 * part2) / part3


def first_partial_cov(y, mu, chol, dof, i, j):
    k = y.shape[1]
    m = j + 1
    Z = np.sum(np.einsum("ndk, nk -> nd", chol, (y - mu)) ** 2, axis=1)
    # Calculate parts
    part2 = (k + dof) / (2 * (Z + dof))
    part3 = 2 * (y - mu)[:, i] * np.sum((y - mu)[:, :m] * chol[:, j, :m], axis=1)
    if i == j:
        part1 = 1 / chol[:, i, i]
    else:
        part1 = 0

    return part1 - part2 * part3


def second_partial_cov(y, mu, chol, dof, i, j):
    k = y.shape[1]
    m = j + 1
    Z = np.sum(np.einsum("ndk, nk -> nd", chol, (y - mu)) ** 2, axis=1)

    deriv_lambda_1 = (
        2 * (y - mu)[:, i] * np.sum((y - mu)[:, :m] * chol[:, j, :m], axis=1)
    )
    deriv_lambda_2 = 2 * (y - mu)[:, i] ** 2

    part2 = k + dof
    part3 = (Z + dof) * deriv_lambda_2 - deriv_lambda_1**2
    part4 = 2 * (Z + dof) ** 2

    if i == j:
        part1 = -1 / chol[:, i, i] ** 2
    else:
        part1 = 0

    return part1 - (part2 * part3) / part4


def first_partial_dof(y, mu, chol, dof):
    k = y.shape[1]
    Z = np.sum(np.einsum("ndk, nk -> nd", chol, (y - mu)) ** 2, axis=1)
    part1 = -(-dof * sp.digamma((k + dof) / 2) + k + dof * sp.digamma(dof / 2)) / (
        2 * dof
    )
    part2 = (Z * (k + dof)) / (2 * dof * (dof + Z)) - 1 / 2 * np.log((dof + Z) / dof)
    return part1 + part2


def second_partial_dof(y, mu, chol, dof):
    k = y.shape[1]
    Z = np.sum(np.einsum("ndk, nk -> nd", chol, (y - mu)) ** 2, axis=1)
    part1 = (
        1
        / 4
        * (
            (2 * k) / (dof**2)
            + sp.polygamma(1, (dof + k) / 2)
            - sp.polygamma(1, dof / 2)
        )
    )
    part2 = (Z * (dof * Z - k * (2 * dof + Z))) / (2 * dof**2 * (dof + Z) ** 2)
    return part1 + part2
