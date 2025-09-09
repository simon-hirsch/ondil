import inspect

import ondil.distributions


def get_distributions_with_gamlss():
    """Get all distribution classes that have a corresponding_gamlss attribute."""
    distributions = []
    for name, obj in inspect.getmembers(ondil.distributions):
        if hasattr(obj, "corresponding_gamlss"):
            instance = obj()
            if instance.corresponding_gamlss is not None:
                distributions.append(instance)
    return distributions
