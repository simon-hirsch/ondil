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
    def fit_beta(self, x_gram, y_gram, **kwargs):
        if self._path_based_method:
            raise NotImplementedError("Method does not support non-path-based fitting.")

    @abstractmethod
    def update_beta(self, x_gram, y_gram, beta, **kwargs):
        if self._path_based_method:
            raise NotImplementedError("Method does not support non-path-based fitting.")

    @abstractmethod
    def fit_beta_path(self, x_gram, y_gram, **kwargs):
        if not self._path_based_method:
            raise NotImplementedError("Method does not support path-based fitting.")

    @abstractmethod
    def update_beta_path(self, x_gram, y_gram, beta_path, **kwargs):
        if not self._path_based_method:
            raise NotImplementedError("Method does not support path-based fitting.")
