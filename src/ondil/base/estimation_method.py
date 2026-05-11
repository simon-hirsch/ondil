from abc import ABC, abstractmethod


class EstimationMethod(ABC):
    def __init__(
        self,
        _path_based_method,
        _accepts_bounds,
        _accepts_selection,
    ):
        self._path_based_method = _path_based_method
        self._accepts_bounds = _accepts_bounds
        self._accepts_selection = _accepts_selection

    @abstractmethod
    def init_x_gram(self, X, weights, forget):
        pass

    @abstractmethod
    def init_y_gram(self, X, y, weights, forget):
        pass

    @abstractmethod
    def update_x_gram(self, gram, X, weights, forget):
        pass

    @abstractmethod
    def update_y_gram(self, gram, X, y, weights, forget):
        pass

    @abstractmethod
    def fit_beta(self, x_gram, y_gram, is_regularized):
        if self._path_based_method:
            raise NotImplementedError("Method does not support non-path-based fitting.")

    @abstractmethod
    def update_beta(self, x_gram, y_gram, beta, is_regularized):
        if self._path_based_method:
            raise NotImplementedError("Method does not support non-path-based fitting.")

    @abstractmethod
    def fit_beta_path(self, x_gram, y_gram, is_regularized):
        if not self._path_based_method:
            raise NotImplementedError("Method does not support path-based fitting.")

    @abstractmethod
    def update_beta_path(self, x_gram, y_gram, beta_path, is_regularized):
        if not self._path_based_method:
            raise NotImplementedError("Method does not support path-based fitting.")

    def compute_edf(self, x_gram, beta_path, is_regularized):
        """Effective degrees of freedom.

        For path-based methods this should return an ``np.ndarray`` of shape
        ``(lambda_n,)`` — one EDF value per point on the regularization
        path. For single-lambda methods this should return a scalar.

        Default implementation raises ``NotImplementedError``; subclasses
        override when an EDF formula consistent with their penalty is
        available.

        The standard formula implemented by subclasses is
        ``edf(lambda) = trace[ G (G + lambda * J)^{-1} ]`` where ``G`` is
        (the active-set restriction of) the weighted Gram matrix and ``J``
        is the penalty Hessian.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement compute_edf."
        )
