# Design proposal for additive models in `ondil`

- Author: Simon [@simon-hirsch](https://www.github.com/simon-hirsch)
- Date: 2025-09-18

This document outlines a design proposal for implementing online additive models in `ondil`. The goal is create a flexible and extensible framework that allows users to define, train, and update additive models efficiently. Currently, `ondil` supports only non-linear terms that can be created by a combination of linear basis functions.

The design consists of two main components:

- A user-facing API for defining and training additive models.
- An internal representation for additive models and the individual components, comparable with a layer structure in neural networks and the respective frameworks (e.g., PyTorch, TensorFlow).
- Some considerations for _online_ training of additive models.

## User-facing API

The user-facing API consists of _additive terms_ (think layers) that can be combined to form an additive model. On a very high level, the API could look like this:

```python
from ondil.distributions import StudentT
from ondil.estimators import AdditiveDistributionalRegression
from ondil.terms import LinearTerm, SplineTerm, AutoregresiveTerm

model = AdditiveDistributionalRegression(
    distribution=StudentT(),
    equation={
        0 : [
            LinearTerm(features=["feature1", "feature2"]),
            SplineTerm(features=["feature3"], n_splines=10, degree=3),
            AutoregresiveTerm(lags=3),
        ], 
        1 : [LinearTerm(features=["feature1"])]
    }, 
    ... # other parameters
)

# And we do the usual stuff
model.fit(X, y)
model.predict(X_new)
model.update(X_new, y_new)
```

That means the `equation` parameter is a dictionary where keys are the indices of the distribution parameters (e.g., `0` for the location parameter, `1` for the scale parameter in a two-parameter distribution) and values are lists of additive terms that contribute to that parameter. Currently, our equation syntax looks like this:

```python
equation = {
    0 : np.array([0, 2, 4, 5]), 
    1 : np.array([1, 3])
}
```

i.e. we only support linear terms and the indices of the features that contribute to each distribution parameter.

### Additive Terms

Additive terms can be of different types (e.g., linear, spline-based, autoregressive, etc.) and can have their own hyperparameters. What is important is that each term implements a common interface, which includes methods for:

- Initializing the term with its hyperparameters.
- Computing the contribution of the term to the linear predictor given input features.
- Updating the term's parameters based on new data (for online learning).

## Internal Representation and API

### Fitting Procedure

The high-level overview of the iterative fitting procedure for additive models is as follows:

Assume that all terms are initialized and that we have a dataset with features `X` and target variable `y`. We save all terms as dictionary entries in the `AdditiveDistributionalRegression` class, e.g., `self.terms = {0: [term1, term2], 1: [term3]}`.

```python
# in the inner loop of the IRLS training procedure

p = ... # current distribution parameter index
it_outer = ... # current outer iteration
it_inner = ... # current inner iteration

theta = ... # current parameter estimates for parameter p
eta = ... # compute current linear predictor for parameter p
working_vector = ...  # compute working vector 
working_weights = ...  # compute working weights
fitted_values = 0

for t, term in enumerate(self.terms[p]):    
    working_vector_term = working_vector - fitted_values
    self.terms[p][t] = term._fit(
        X=X, 
        y=working_vector, 
        w=working_weights, 
    )
    fitted_values = fitted_values + self.predict(X=X)
    eta = eta + self.predict(X=X)  # update eta 

```

We need to take some care with the calculation of the working vector and the weights. I have not yet fully worked this out, but I understand the GAMLSS code as if the weights are constant across all terms for given inner and outer iterations, while the working vector is updated after each term fit by only taking the residuals.

### Term Interface

The terms need to define a number of methods to be compatible with the above fitting procedure. At a minimum, we need:

- `fit(self, X, y, w)`: Fit the term to the working vector `y` with weights `w` and features `X`. In this function, the term needs to update its internal parameters and be able to create the necessary basis functions, etc.
- `predict(self, X)`: Compute the contribution of the term to the linear predictor given features `X`.
- `update(self, X, y, w)`: Update the term's parameters based on new data for online learning. A caveeat here is that we need to to be able to update _return_ an updated version of the term _without_ altering the existing term in place. This is because we need to be able to compute the working vector for each term fit based on the previous version of the term in the update of the GAMLSS fitting procedure (we always go from Gram $G^{[T]}$ to $G^{[T+1, i, j]}$ for the inner and outer iterations $i,j$  in the algorithm) - otherwise we would run the risk of using already updating the gramian multiple times in one update step.

Taking the last point into consideration, the `fit` and the `update` methods could return an updated version of the term instead of updating it in place. This helps avoiding side effects. On the flip side, this could lead to increased memory usage if not handled carefully (and to generally a lot of rather slow memory allocations).

Also, in each online update we are going to have "old terms" and "new terms", i.e., the terms before and after the update. We need to be careful to always use the "old terms" when computing the working vector for each term fit in the update procedure. After all the fitting, we need to make sure that we set the "new terms" to be the "old terms" for the next update.

### Additional Considerations

#### Intercept Term

Do we want to have a term that only represents an intercept? This could be useful for models that do not have any other terms for a given distribution parameter.

#### Regularization and Model Selection

What is the interaction between terms and `EstimationMethods`? Do we need to have different fitting procedures for different estimation methods, or can we abstract this away in the term interface. I think we could use the estimation methods as parameters to _some_ of the term types, e.g. having:

```python
LinearTerm(features=[1, 2, 4], method=LassoPathMethod())
```

But then we need to take into account model selection also inside the term (which is fine, but we might need to have a few additional terms then (i.e. terms that do not support model selection and terms that do). This would leave us to have:

- `LinearTerm` (no model selection)
- `RegularizedICLinearTerm` (with model selection using IC)

To be able to support our existing `EstimationMethods`. For global model selection, this gets more complicated because you could select from the product of all terms, which is combinatorially explosive.

For the future, the following would be nice:

- `SplineTerm` (no model selection resp. automatic smoothing parameter selection)
- `AutoRegressiveTerm` (no model selection)
- `RegularizedICAutoRegressiveTerm` (with model selection using IC)
- `ScikitLearnTerm` (wrapper around sklearn models, e.g., random forests, gradient boosting machines, etc.)
