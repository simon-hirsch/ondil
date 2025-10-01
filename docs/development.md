# Development

First of all, we're happy that you want to take the time to contribute to this package and extending it. Thanks!

This page is trying to give some guidance on how to write additional link functions or distributions and some of the ideas underlying the [`LinkFunction`][ondil.base.LinkFunction], the [`Distribution`][ondil.base.Distribution] and the [`ScipyMixin`][ondil.base.ScipyMixin] classes.

## Writing Link Functions

Link functions map the predictors of the distribution parameter to the distribution parameter's support. Hence, we need to have the link function itself and its inverse. To calculate the score vector and the weights, we additionally need the first and second derivative of the link, as well as the derivative of the inverse.

The [`LinkFunction`][ondil.base.LinkFunction] abstract base class enforces, that `SomeNewLink()` implements:

- The link: `SomeNewLink().link()`.
- The inverse: `SomeNewLink().inverse()`.
- The first derivative of the link: `SomeNewLink().link_derivative()`.
- The second derivative of the link: `SomeNewLink().link_second_derivative()`.
- The derivative of the inverse: `SomeNewLink().inverse_derivative()`.

Additionally, each link defines the `SomeNewLink().link_support` as `tuple` of `float` values. This is used to ensure at the initialization of a distribution that the link functions support is inside the support of the distribution parameter. We allow the link to shorten the possible outcome space of the parameter, but we don't allow to return impossible values. As an example, you can constrain the degrees of freedom $\nu$ of the $t$-distribution by taking a [`LogShiftTwo()`][ondil.links.LogShiftTwo], which ensures that $\nu > 2$ (and therefore the variance exists), but you cannot choose the [`Identity()`][ondil.links.Identity], since the degrees of freedom must be positive.

There are two ways to implement the support of the link function:

- As a class attribute, i.e. before the `__init__()`. This is usually done if parameters passed to the link function *do not* affect the support. We write something like

    ```python
    class SomeNewLink(LinkFunction):
        link_support = (0, np.inf)

        def __init__(self)
            pass # or do something

        def all_the_necessary_methods():
            # implement links and derivative
    ```

- As property to the class. This is usually done if the link functions parameter affect its support. Then we do this:

    ```python
    class SomeNewLink(LinkFunction):

        def __init__(self, lower_support_bound):
            self.lower_support_bound = lower_support_bound

        @property
        def link_support(self):
            return (self.lower_support_bound, np.inf)

        def all_the_necessary_methods():
            # implement links and derivative
    ```

To ensure that we model the support correctly - think about, e.g. $\log(x)$, we can use `np.nextafter(a, b)`, which gives the next possible float after `a` in the direction of `b`. Hence, `np.nextafter(0, 1)` will return a very small positive float, which describes the support of $\log(x)$ well.

## Writing Distributions

Distributions are an integral part of the package. All distributions inherit from [`Distribution`][ondil.base.Distribution], an abstract base class. This class enforces that each distribution has certain methods and members and also provides some default methods.

The [`Distribution`][ondil.base.Distribution] class is a very general class. However, some distributions have the corresponding implementation in `scipy.stats`. To use this, the [`ScipyMixin`][ondil.base.ScipyMixin] class provides the already existing implementations of `cdf`, `pdf` etc. This makes implementing a new distribution very easy.

### Scipy Distributions

Let's consider the `Normal` distribution as an example. The `Normal` distribution is implemented in `scipy.stats` as `scipy.stats.norm`. Hence, we can use the [`ScipyMixin`][ondil.base.ScipyMixin] together with [`Distribution`][ondil.base.Distribution] to implement the `Normal` distribution. The `Normal` distribution is implemented as follows:

```python
class Normal(Distribution, ScipyMixin):
    """Corresponds to GAMLSS NO() and scipy.stats.norm()"""

 corresponding_gamlss: str = "NO"

 parameter_names = {0: "mu", 1: "sigma"}
 parameter_support = {0: (-np.inf, np.inf), 1: (np.nextafter(0, 1), np.inf)}
 distribution_support = (-np.inf, np.inf)

    # Scipy equivalent and parameter mapping ondil -> scipy
 scipy_dist = st.norm
 scipy_names = {"mu": "loc", "sigma": "scale"}
```

First, we assign a name and declare the inheritance. Then, we define some class properties like `corresponding_gamlss`, `parameter_names`, `parameter_support` etc. These are enforced through `@property` decorators in [`Distribution`][ondil.base.Distribution] and [`ScipyMixin`][ondil.base.ScipyMixin].

Note that `parameter_names` relates the variable names used throughout the methods to the columns of the parameter array `theta`. In turn, `scipy_names` relates these variable names to the argument names of the `scipy.stats` distribution. This is necessary to map the parameters correctly.

If possible, we follow the naming conventions of the `gamlss` package:

```
0 : Location
1 : Scale (close to standard deviation)
2 : Skewness
3 : Tail behaviour
```

Then, we add an initialization method:

```python
def __init__(
 self,
 loc_link: LinkFunction = Identity(),
 scale_link: LinkFunction = Log(),
) -> None:
    super().__init__(
        links={
            0: loc_link,
            1: scale_link,
 }
 )
```

This must provide the links for the parameters to the initializer of the base class.

Furthermore, we need to define a method `initial_values`, which provides the initial values for the distribution parameters:

```python
def initial_values(
 self, y: np.ndarray, param: int = 0, axis: Optional[int | None] = None
) -> np.ndarray:
    if param == 0:
        return np.repeat(np.mean(y, axis=axis), y.shape[0])
    if param == 1:
        return np.repeat(np.std(y, axis=axis), y.shape[0])
```

Lastly, we need to implement the log-likelihood function's first, second and cross-derivatives: `dl1_dp1`, `dl2_dp2`, and `dl2_dpp` respectively.

Thats it! We implemented the `Normal` distribution.

#### Special Cases

Some distributions may be special. For example, our implementation of the Gamma distribution [`Gamma`][ondil.distributions.Gamma] uses a different parameterization than `scipy.stats.gamma`. In consequence, we cannot use `theta_to_scipy_params()` from [`ScipyMixin`][ondil.base.ScipyMixin] to map the parameters to `scipy`. In this case, we must implement the `theta_to_scipy_params()` method ourselves. This is done in the `Gamma` class:

```python
def theta_to_scipy_params(self, theta: np.ndarray) -> dict:
    """Map GAMLSS Parameters to scipy parameters.

 Args:
 theta (np.ndarray): parameters

 Returns:
 dict: Dict of (a, loc, scale) for scipy.stats.gamma(a, loc, scale)
 """
 mu = theta[:, 0]
 sigma = theta[:, 1]
 beta = 1 / (sigma**2 * mu)
 params = {"a": 1 / sigma**2, "loc": 0, "scale": 1 / beta}
    return params
```

By providing a method called `theta_to_scipy_params` in the distribution class, we overwrite the default implementation from [`ScipyMixin`][ondil.base.ScipyMixin] and can map the parameters correctly.

## Summary

Implementing your own distribution is fairly straightforward, especially if there is already a `scipy.stats` implementation. A good starting point is to look at the [`Normal`][ondil.distributions.Normal] function.

If things need to be adjusted, then [`Gamma`][ondil.distributions.Gamma] demonstrates how to do this.

If you want to implement a distribution that is not available in `scipy.stats`, then you need to implement some methods, including `cdf`, `pdf`, and `rvs` yourself. In that case, your new class will inherit only from [`Distribution`][ondil.base.Distribution]. You can inspect [`ScipyMixin`][ondil.base.ScipyMixin] to see which other methods need to be implemented.
