import inspect

import ondil.distributions
from ondil.base.distribution import MultivariateDistributionMixin


def get_distributions_with_gamlss():
    """Get all distribution classes that have a corresponding_gamlss attribute."""
    distributions = []
    for name, obj in inspect.getmembers(ondil.distributions):
        if hasattr(obj, "corresponding_gamlss") and (
            obj.corresponding_gamlss is not None
        ):
            instance = obj()
            if instance.corresponding_gamlss is not None:
                distributions.append(instance)
    return distributions


def get_univariate_distributions():
    """Get all univariate distribution classes."""
    distributions = []
    for name, obj in inspect.getmembers(ondil.distributions):
        if inspect.isclass(obj) and not issubclass(obj, MultivariateDistributionMixin):
            distributions.append(obj())
    return distributions
