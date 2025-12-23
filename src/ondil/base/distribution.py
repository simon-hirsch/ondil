from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple
import warnings

import numpy as np
import scipy.optimize as opt
import scipy.stats as st

from ..warnings import OutOfSupportWarning
from .link import LinkFunction


class Distribution(ABC):
    def __init__(self, links: dict[int, LinkFunction]) -> None:
        self.links = links
        self._validate_links()

    def __call__(self, *args, **kwds):
        raise NotImplementedError("Not implemented but necessary for sklearn.")

    @property
    def corresponding_gamlss(self) -> str | None:
        """The name of the corresponding implementation in 'gamlss.dist' R package."""
        return None

    @property
    def is_discrete(self) -> bool:
        """Whether the distribution is discrete or continuous."""
        return False

    @property
    @abstractmethod
    def parameter_names(self) -> dict:
        """Parameter name for each column of theta."""
        pass

    @property
    @abstractmethod
    def parameter_shape(self) -> dict:
        """Parameter name for each column of theta."""
        pass

    @property
    def n_params(self) -> int:
        """Each subclass must define 'n_params'."""
        return len(self.parameter_names)

    def theta_to_params(self, theta: np.ndarray) -> Tuple[np.ndarray, ...]:
        """Take the fitted values and return tuple of vectors for distribution parameters."""
        return tuple(theta[:, i] for i in range(self.n_params))

    @property
    @abstractmethod
    def distribution_support(self) -> Tuple[float, float]:
        """The support of the distribution."""
        pass

    @property
    @abstractmethod
    def parameter_support(self) -> dict:
        """The support of each parameter of the distribution."""
        pass

    @abstractmethod
    def dl1_dp1(self, y: np.ndarray, theta: np.ndarray, param: int) -> np.ndarray:
        """Take the first derivative of the likelihood function with respect to the param."""

    @abstractmethod
    def dl2_dp2(self, y: np.ndarray, theta: np.ndarray, param: int) -> np.ndarray:
        """Take the second derivative of the likelihood function with respect to the param."""

    @abstractmethod
    def dl2_dpp(
        self, y: np.ndarray, theta: np.ndarray, params: Tuple[int, int]
    ) -> np.ndarray:
        """Take the first derivative of the likelihood function with respect to both parameters."""

    def _validate_links(self):
        for param, link in self.links.items():
            if link.link_support[0] < self.parameter_support[param][0]:
                warnings.warn(
                    message=f"Lower bound of link function is smaller than the parameter support for parameter {param} ",
                    category=OutOfSupportWarning,
                )
            if link.link_support[1] > self.parameter_support[param][1]:
                warnings.warn(
                    message=f"Upper bound of link function is larger than the parameter support for parameter {param} ",
                    category=OutOfSupportWarning,
                )

            # TODO: Add valid shapes to all link functions.
            if self.parameter_shape[param] not in self.links[param].valid_shapes:
                raise ValueError(
                    f"Link function does not match parameter structure for parameter {param}. \n"
                    f"Parameter structure is {self.parameter_shape[param]}. \n"
                    f"Link function supports {self.links[param].valid_shapes}"
                )

    def _validate_dln_dpn_inputs(
        self, y: np.ndarray, theta: np.ndarray, param: int
    ) -> None:
        if param >= self.n_params:
            raise ValueError(
                f"{self.__class__.__name__} has only {self.n_params} distribution parameters.\nYou have passed {param}. Please remember we start counting at 0."
            )

    def _validate_dl2_dpp_inputs(
        self, y: np.ndarray, theta: np.ndarray, params: Tuple[int, int]
    ) -> None:
        if max(params) >= self.n_params:
            raise ValueError(
                f"{self.__class__.__name__} has only {self.n_params} distribution parameters.\nYou have passed {params}. Please remember we start counting at 0."
            )
        if params[0] == params[1]:
            raise ValueError("Cross derivatives must use different parameters.")

    def link_function(
        self,
        y: np.ndarray,
        param: int = 0,
    ) -> np.ndarray:
        """Apply the link function for param on y."""
        return self.links[param].link(y)

    def link_inverse(
        self,
        y: np.ndarray,
        param: int = 0,
    ) -> np.ndarray:
        """Apply the inverse of the link function for param on y."""
        return self.links[param].inverse(y)

    def link_function_derivative(
        self,
        y: np.ndarray,
        param: int = 0,
    ) -> np.ndarray:
        """Apply the derivative of the link function for param on y."""
        return self.links[param].link_derivative(y)

    def link_inverse_derivative(
        self,
        y: np.ndarray,
        param: int = 0,
    ) -> np.ndarray:
        """Apply the derivative of the inverse link function for param on y."""
        return self.links[param].inverse_derivative(y)

    def link_function_second_derivative(
        self,
        y: np.ndarray,
        param: int = 0,
    ) -> np.ndarray:
        """Apply the second derivative of the link function for param on y."""
        return self.links[param].link_second_derivative(y)

    @abstractmethod
    def initial_values(
        self, y: np.ndarray, param: int = 0, axis: Optional[int | None] = None
    ) -> np.ndarray:
        """Calculate the initial values for the GAMLSS fit."""

    def quantile(self, q: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """
        Compute the quantile function for the given data.

        This is a alias for the `ppf` method.

        Args:
            q (np.ndarray): The quantiles to compute.
            theta (np.ndarray): The parameters of the distribution.

        Returns:
            np.ndarray: The quantiles corresponding to the given probabilities.
        """
        return self.ppf(q, theta)

    def _raise_not_implemented(self, method_for_discrete: bool = True) -> None:
        if self.is_discrete and (not method_for_discrete):
            raise NotImplementedError(
                "This method is not implemented for discrete distributions."
            )
        if (not self.is_discrete) and method_for_discrete:
            raise NotImplementedError(
                "This method is not implemented for continuous distributions."
            )

    @abstractmethod
    def calculate_conditional_initial_values(
        self,
        y: np.ndarray,
        theta: np.ndarray,
        param: int,
    ) -> np.ndarray:
        """Calculate the conditional initial values for the GAMLSS fit."""

    @abstractmethod
    def cdf(self, y: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """
        Compute the cumulative distribution function (CDF) for the given data.

        Args:
            y (np.ndarray): The data points at which to evaluate the CDF.
            theta (np.ndarray): The parameters of the distribution.

        Returns:
            np.ndarray: The CDF evaluated at the given data points.
        """

    @abstractmethod
    def pdf(self, y: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """
        Compute the probability density function (PDF) for the given data points.

        Args:
            y (np.ndarray): An array of data points at which to evaluate the PDF.
            theta (np.ndarray): An array of parameters for the distribution.

        Returns:
            np.ndarray: An array of PDF values corresponding to the data points in `y`.
        """
        self._raise_not_implemented(method_for_discrete=False)

    @abstractmethod
    def pmf(self, y: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """
        Compute the probability mass function (PMF) for the given data points.

        Args:
            y (np.ndarray): An array of data points at which to evaluate the PDF.
            theta (np.ndarray): An array of parameters for the distribution.

        Returns:
            np.ndarray: An array of PMF values corresponding to the data points in `y`.
        """
        self._raise_not_implemented(method_for_discrete=True)

    @abstractmethod
    def ppf(self, q: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """
        Percent Point Function (Inverse of CDF).

        Args:
            q (np.ndarray): Quantiles.
            theta (np.ndarray): Distribution parameters.

        Returns:
            np.ndarray: The quantile corresponding to the given probabilities.
        """

    @abstractmethod
    def rvs(self, size: int, theta: np.ndarray) -> np.ndarray:
        """
        Generate random variates of given size and parameters.

        Args:
            size (int): The number of random variates to generate.
            theta (np.ndarray): The parameters for the distribution.

        Returns:
            np.ndarray: A 2D array of random variates with shape (theta.shape[0], size).
        """

    @abstractmethod
    def logpmf(self, y: np.ndarray, theta: np.ndarray) -> np.ndarray:
        self._raise_not_implemented(method_for_discrete=True)

    @abstractmethod
    def logpdf(self, y: np.ndarray, theta: np.ndarray) -> np.ndarray:
        self._raise_not_implemented(method_for_discrete=False)

    @abstractmethod
    def logcdf(self, y: np.ndarray, theta: np.ndarray) -> np.ndarray:
        self._raise_not_implemented(method_for_discrete=False)

    def loglikelihood(self, y: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """Compute the log-likelihood for the given data points.

        Args:
            y (np.ndarray): An array of data points at which to evaluate the log-likelihood.
            theta (np.ndarray): An array of parameters for the distribution.
        Returns:
            np.ndarray: An array of log-likelihood values corresponding to the data points in `y`.
        """
        if self.is_discrete:
            return self.logpmf(y, theta)
        else:
            return self.logpdf(y, theta)


class MultivariateDistributionMixin(ABC):
    # TODO: The element link functions should be defined here as well.

    @staticmethod
    @abstractmethod
    def fitted_elements(dim: int) -> Dict[int, int]:
        return NotImplementedError(
            "This method should return a list of fitted elements for the multivariate distribution."
        )

    @abstractmethod
    def element_dl1_dp1(
        self, y: np.ndarray, theta: Dict, param: int = 0, k: int = 0, clip=False
    ):
        raise NotImplementedError("Not implemented in ABC")

    @abstractmethod
    def element_dl2_dp2(
        self, y: np.ndarray, theta: Dict, param: int = 0, k: int = 0, clip=False
    ):
        raise NotImplementedError("Not implemented in ABC")


class ScipyMixin(ABC):
    @property
    @abstractmethod
    def parameter_names(self) -> dict:
        """Parameter name for each column of theta."""
        pass

    @property
    @abstractmethod
    def scipy_dist(self) -> st.rv_continuous:
        """The names of the parameters in the scipy.stats distribution and the corresponding column in theta."""
        pass

    @property
    @abstractmethod
    def scipy_names(self) -> Tuple[str]:
        """The names of the parameters in the scipy.stats distribution and the corresponding column in theta."""
        pass

    def theta_to_scipy_params(self, theta: np.ndarray) -> Dict[str, np.ndarray]:
        """Maps $\\theta$ to the `scipy` parameters.

        Args:
            theta (np.ndarray): $\\theta$ as estimated by `OnlineDistributionalRegression()` estimator

        Raises:
            ValueError: If we don't define the `scipy_names` attribute.

        Returns:
            dict: Dictionary that can be unrolled into scipy distribution class as in `st.some_dist(**return_value)`
        """
        if not self.scipy_names:
            raise ValueError(
                f"{self.__class__.__name__} has no scipy_names defined. To use theta_to_scipy_params Please define them in the subclass. Or override this method in the subclass if there is no 1:1 mapping between theta columns and scipy params."
            )

        params = {}
        for idx, name in self.parameter_names.items():
            params[self.scipy_names[name]] = theta[:, idx]
        return params

    def cdf(self, y: np.ndarray, theta: np.ndarray) -> np.ndarray:
        super().cdf(y, theta)
        return self.scipy_dist(**self.theta_to_scipy_params(theta)).cdf(y)

    def pdf(self, y: np.ndarray, theta: np.ndarray) -> np.ndarray:
        super().pdf(y, theta)
        return self.scipy_dist(**self.theta_to_scipy_params(theta)).pdf(y)

    def pmf(self, y: np.ndarray, theta: np.ndarray) -> np.ndarray:
        super().pmf(y, theta)
        return self.scipy_dist(**self.theta_to_scipy_params(theta)).pmf(y)

    def ppf(self, q: np.ndarray, theta: np.ndarray) -> np.ndarray:
        super().ppf(q, theta)
        return self.scipy_dist(**self.theta_to_scipy_params(theta)).ppf(q)

    def logpmf(self, y: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """Compute the log of the probability mass function (PMF) for the given data points.

        Args:
            y (np.ndarray): An array of data points at which to evaluate the log PMF.
            theta (np.ndarray): An array of parameters for the distribution.

        Returns:
            np.ndarray: An array of log PMF values corresponding to the data points in `y`.
        """
        super().cdf(y, theta)
        return self.scipy_dist(**self.theta_to_scipy_params(theta)).logpmf(y)

    def logpdf(self, y: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """Compute the log of the probability density function (PDF) for the given data points.

        Args:
            y (np.ndarray): An array of data points at which to evaluate the log PDF.
            theta (np.ndarray): An array of parameters for the distribution.

        Returns:
            np.ndarray: An array of log PDF values corresponding to the data points in `y`.
        """
        super().logpdf(y, theta)
        return self.scipy_dist(**self.theta_to_scipy_params(theta)).logpdf(y)

    def logcdf(self, y: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """Compute the log of the cumulative distribution function (CDF) for the given data points.

        Args:
            y (np.ndarray): An array of data points at which to evaluate the log CDF.
            theta (np.ndarray): An array of parameters for the distribution.

        Returns:
            np.ndarray: An array of log CDF values corresponding to the data points in `y`.
        """
        super().logcdf(y, theta)
        return self.scipy_dist(**self.theta_to_scipy_params(theta)).logcdf(y)

    def rvs(self, size: int, theta: np.ndarray) -> np.ndarray:
        return (
            self
            .scipy_dist(**self.theta_to_scipy_params(theta))
            .rvs((size, theta.shape[0]))
            .T
        )

    def mean(self, theta: np.ndarray) -> np.ndarray:
        """Compute the mean of the distribution for the given parameters.

        Args:
            theta (np.ndarray): An array of parameters for the distribution.

        Returns:
            np.ndarray: An array of means corresponding to the parameters in `theta`.
        """
        return self.scipy_dist(**self.theta_to_scipy_params(theta)).mean()

    def _scipy_mle_objective(
        self,
        start: np.ndarray,
        y: np.ndarray,
        fixed: dict,
        param: int,
    ):
        unknown = np.arange(param, self.n_params)
        decision = {}
        for i, value in zip(unknown, start):
            decision[self.scipy_names[self.parameter_names[i]]] = value
        return -np.sum(self.scipy_dist(**decision, **fixed).logpdf(y))

    def calculate_conditional_initial_values(
        self,
        y: np.ndarray,
        theta: np.ndarray,
        param: int,
    ):
        start = theta[0, param:]
        fixed = self.theta_to_scipy_params(theta)
        unknown = np.arange(param, self.n_params)
        for i in unknown:
            del fixed[self.scipy_names[self.parameter_names[i]]]

        bounds = [self.links[b].link_support for b in unknown]
        result = opt.minimize(
            self._scipy_mle_objective,
            x0=start,
            args=(y, fixed, param),
            method="L-BFGS-B",
            bounds=bounds,
            # options={"disp": True}
        )
        new = theta.copy()
        if not result.success:
            print("Optimization failed: " + result.message)
            print("Will use unconditional start values")
        else:
            new[:, unknown] = np.expand_dims(result.x, 0)
        return new
