from typing import Literal, Union
import numpy as np


class InformationCriterion:
    """Calculate the information criteria.

    +---------+--------------------------------------+--------------------------+
    | `ic`    | Information Criterion                | Formula                  |
    +=========+======================================+==========================+
    | `"aic"` | Akaike's Information Criterion       | $- 2l + 2p$              |
    | `"aicc"`| Corr. Akaike's Information Criterion | $- 2l + 2pn/(n-p-1)$     |
    | `"bic"` | Bayesian Information Criterion       | $- 2l + p\log(n)$        |
    | `"hqc"` | Hannan-Quinn Information Criterion   | $- 2l + 2p\log(\log(n))$ |
    | `"max"` | Select the largest model             |                          |
    +---------+--------------------------------------+--------------------------+

    Methods:
    -------
    from_rss(rss)
        Compute the chosen criterion from residual sum of squares.
    from_ll(log_likelihood)
        Compute the chosen criterion directly from a log-likelihood value.
    """

    def __init__(
        self,
        n_observations: Union[int, np.ndarray],
        n_parameters: Union[int, np.ndarray],
        criterion: Literal["aic", "bic", "hqc", "aicc", "max"] = "aic",
    ):
        """
        Args:
            n_observations (int or array-like): Number of observations used in the model.
            n_parameters (int or array-like): Number of estimated parameters in the model.
            criterion ({"aic","bic","hqc","aicc", "max"}, default="aic"): The information criterion to compute.


        Raises:
            ValueError: If the criterion is not recognized.
        """
        # validate criterion early
        if criterion not in ("aic", "aicc", "bic", "hqc", "max"):
            raise ValueError(f"Criterion '{criterion}' not recognized.")
        self.n_observations = n_observations
        self.n_parameters = n_parameters
        self.criterion = criterion

    def from_rss(self, rss: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Compute the specified information criterion from the residual sum of squares (RSS).

        The Gaussian log-likelihood is estimated as:
            ll = -n/2 * log(rss / n) - n/2 * (1 + log(2π))

        Args:
            rss (float or array-like): Residual sum of squares of the fitted model.

        Returns:
            ic (float or array-like): The information criterion value (AIC, AICC, BIC, HQC, or Max).
        """
        # Gaussian log‐likelihood: ll = -n/2*log(rss/n) + constant
        # https://en.wikipedia.org/wiki/Akaike_information_criterion#Comparison_with_least_squares
        constant_term = -self.n_observations / 2 * (1 + np.log(2 * np.pi))
        log_likelihood = (
            -self.n_observations / 2 * np.log(rss / self.n_observations) + constant_term
        )
        return self.from_ll(log_likelihood)

    def from_ll(
        self, log_likelihood: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """
        Compute the specified information criterion directly from log-likelihood.

        Args:
            log_likelihood (float or array-like): The log-likelihood of the model.

        Returns:
            ic (float or array-like): The information criterion value (AIC, AICC, BIC, HQC, or Max).
        """
        c = self.criterion
        n, p, ll = self.n_observations, self.n_parameters, log_likelihood
        if c == "aic":
            return - 2 * ll + p * 2 
        elif c == "aicc":
            if n - p - 1 == 0:
                raise ValueError("Invalid inputs: n - p - 1 must not be zero for AICC calculation.")
            return - 2 * ll + p * 2 * n / ( n - p - 1)
        elif c == "bic":
            return - 2 * ll + p * np.log(n) 
        elif c == "hqc":
            return - 2 * ll + p * 2 * np.log(np.log(n))
        elif c == "max":
            return -ll
