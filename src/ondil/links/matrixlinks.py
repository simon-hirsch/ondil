from typing import Tuple

import numpy as np

from ..base import LinkFunction
from ..types import ParameterShapes


class MatrixDiag(LinkFunction):
    """
    Wraps a link functions to be applied only on the diagonal of a square matrix.
    """

    valid_shapes = [ParameterShapes.DIAGONAL_MATRIX]

    def __init__(self, diag_link: LinkFunction, other_val=0):
        self.diag_link = diag_link
        self.other_val = other_val

    @property
    def link_support(self) -> Tuple[float, float]:
        return (self.diag_link.link_support[0], self.diag_link.link_support[1])

    def _make_indices(self, x: np.ndarray) -> Tuple:
        d = x.shape[1]
        i = np.diag_indices(d)
        return i

    def link(self, x: np.ndarray) -> np.ndarray:
        i = self._make_indices(x)
        out = np.full_like(x, self.other_val)
        out[:, i[0], i[1]] = self.diag_link.link(x[:, i[0], i[1]])
        return out

    def inverse(self, x: np.ndarray) -> np.ndarray:
        i = self._make_indices(x)
        out = np.full_like(x, self.other_val)
        out[:, i[0], i[1]] = self.diag_link.inverse(x[:, i[0], i[1]])
        return out

    def link_derivative(self, x: np.ndarray) -> np.ndarray:
        i = self._make_indices(x)
        out = np.full_like(x, self.other_val)
        out[:, i[0], i[1]] = self.diag_link.link_derivative(x[:, i[0], i[1]])
        return out

    def link_second_derivative(self, x: np.ndarray) -> np.ndarray:
        i = self._make_indices(x)
        out = np.full_like(x, self.other_val)
        out[:, i[0], i[1]] = self.diag_link.link_second_derivative(x[:, i[0], i[1]])
        return out

    def inverse_derivative(self, x: np.ndarray) -> np.ndarray:
        i = self._make_indices(x)
        out = np.full_like(x, self.other_val)
        out[:, i[0], i[1]] = self.diag_link.inverse_derivative(x[:, i[0], i[1]])
        return out

    def element_link(self, x: np.ndarray, i: int, j: int) -> np.ndarray:
        if i == j:
            return self.diag_link.link(x)
        else:
            raise ValueError("Element does not exist in the diagonal.")

    def element_link_derivative(self, x: np.ndarray, i: int, j: int) -> np.ndarray:
        if i == j:
            return self.diag_link.link_derivative(x)
        else:
            raise ValueError("Element does not exist in the diagonal.")

    def element_link_second_derivative(
        self, x: np.ndarray, i: int, j: int
    ) -> np.ndarray:
        if i == j:
            return self.diag_link.link_second_derivative(x)
        else:
            raise ValueError("Element does not exist in the diagonal.")

    def element_inverse_derivative(self, x: np.ndarray, i: int, j: int) -> np.ndarray:
        if i == j:
            return self.diag_link.inverse_derivative(x)
        else:
            raise ValueError("Element does not exist in the diagonal.")

    def element_inverse(self, x: np.ndarray, i: int, j: int) -> np.ndarray:
        if i == j:
            return self.diag_link.inverse(x)
        else:
            raise ValueError("Element does not exist in the diagonal.")


class MatrixDiagTriu(LinkFunction):
    """
    Wraps two link functions to be applied on the diagonal and the upper diagonal of a square matrix.
    """

    valid_shapes = [ParameterShapes.UPPER_TRIANGULAR_MATRIX]

    def __init__(self, diag_link: LinkFunction, triu_link: LinkFunction):
        self.diag_link = diag_link
        self.triu_link = triu_link

    @property
    def link_support(self) -> Tuple[float, float]:
        return (
            min(self.triu_link.link_support[0], self.diag_link.link_support[0]),
            max(self.triu_link.link_support[1], self.diag_link.link_support[1]),
        )

    def _make_indices(self, x: np.ndarray) -> Tuple:
        d = x.shape[1]
        i = np.diag_indices(d)
        j = np.triu_indices(d, k=1)
        return i, j

    def link(self, x: np.ndarray) -> np.ndarray:
        i, j = self._make_indices(x)
        out = np.zeros_like(x)
        out[:, i[0], i[1]] = self.diag_link.link(x[:, i[0], i[1]])
        out[:, j[0], j[1]] = self.triu_link.link(x[:, j[0], j[1]])
        return out

    def inverse(self, x: np.ndarray) -> np.ndarray:
        i, j = self._make_indices(x)
        out = np.zeros_like(x)
        out[:, i[0], i[1]] = self.diag_link.inverse(x[:, i[0], i[1]])
        out[:, j[0], j[1]] = self.triu_link.inverse(x[:, j[0], j[1]])
        return out

    def link_derivative(self, x: np.ndarray) -> np.ndarray:
        i, j = self._make_indices(x)
        out = np.zeros_like(x)
        out[:, i[0], i[1]] = self.diag_link.link_derivative(x[:, i[0], i[1]])
        out[:, j[0], j[1]] = self.triu_link.link_derivative(x[:, j[0], j[1]])
        return out

    def link_second_derivative(self, x: np.ndarray) -> np.ndarray:
        i, j = self._make_indices(x)
        out = np.zeros_like(x)
        out[:, i[0], i[1]] = self.diag_link.link_second_derivative(x[:, i[0], i[1]])
        out[:, j[0], j[1]] = self.triu_link.link_second_derivative(x[:, j[0], j[1]])
        return out

    def inverse_derivative(self, x: np.ndarray) -> np.ndarray:
        i, j = self._make_indices(x)
        out = np.zeros_like(x)
        out[:, i[0], i[1]] = self.diag_link.inverse_derivative(x[:, i[0], i[1]])
        out[:, j[0], j[1]] = self.triu_link.inverse_derivative(x[:, j[0], j[1]])
        return out

    def element_link(self, x: np.ndarray, i: int, j: int) -> np.ndarray:
        if i == j:
            return self.diag_link.link(x)
        else:
            return self.triu_link.link(x)

    def element_link_derivative(self, x: np.ndarray, i: int, j: int) -> np.ndarray:
        if i == j:
            return self.diag_link.link_derivative(x)
        else:
            return self.triu_link.link_derivative(x)

    def element_link_second_derivative(
        self, x: np.ndarray, i: int, j: int
    ) -> np.ndarray:
        if i == j:
            return self.diag_link.link_second_derivative(x)
        else:
            return self.triu_link.link_second_derivative(x)

    def element_inverse_derivative(self, x: np.ndarray, i: int, j: int) -> np.ndarray:
        if i == j:
            return self.diag_link.element_inverse_derivative(x)
        else:
            return self.triu_link.element_inverse_derivative(x)

    def element_inverse(self, x: np.ndarray, i: int, j: int) -> np.ndarray:
        if i == j:
            return self.diag_link.inverse(x)
        else:
            return self.triu_link.inverse(x)


class MatrixDiagTril(LinkFunction):
    valid_shapes = [ParameterShapes.LOWER_TRIANGULAR_MATRIX]

    def __init__(self, diag_link: LinkFunction, tril_link: LinkFunction):
        self.diag_link = diag_link
        self.tril_link = tril_link

    @property
    def link_support(self) -> Tuple[float, float]:
        return (
            min(self.tril_link.link_support[0], self.diag_link.link_support[0]),
            max(self.tril_link.link_support[1], self.diag_link.link_support[1]),
        )

    def _make_indices(self, x: np.ndarray) -> Tuple:
        d = x.shape[1]
        i = np.diag_indices(d)
        j = np.tril_indices(d, k=-1)
        return i, j

    def link(self, x: np.ndarray) -> np.ndarray:
        i, j = self._make_indices(x)
        out = np.zeros_like(x)
        out[:, i[0], i[1]] = self.diag_link.link(x[:, i[0], i[1]])
        out[:, j[0], j[1]] = self.tril_link.link(x[:, j[0], j[1]])
        return out

    def inverse(self, x: np.ndarray) -> np.ndarray:
        i, j = self._make_indices(x)
        out = np.zeros_like(x)
        out[:, i[0], i[1]] = self.diag_link.inverse(x[:, i[0], i[1]])
        out[:, j[0], j[1]] = self.tril_link.inverse(x[:, j[0], j[1]])
        return out

    def link_derivative(self, x: np.ndarray) -> np.ndarray:
        i, j = self._make_indices(x)
        out = np.zeros_like(x)
        out[:, i[0], i[1]] = self.diag_link.link_derivative(x[:, i[0], i[1]])
        out[:, j[0], j[1]] = self.tril_link.link_derivative(x[:, j[0], j[1]])
        return out

    def inverse_derivative(self, x):
        i, j = self._make_indices(x)
        out = np.zeros_like(x)
        out[:, i[0], i[1]] = self.diag_link.inverse_derivative(x[:, i[0], i[1]])
        out[:, j[0], j[1]] = self.tril_link.inverse_derivative(x[:, j[0], j[1]])
        return out

    def link_second_derivative(self, x: np.ndarray) -> np.ndarray:
        i, j = self._make_indices(x)
        out = np.zeros_like(x)
        out[:, i[0], i[1]] = self.diag_link.link_second_derivative(x[:, i[0], i[1]])
        out[:, j[0], j[1]] = self.tril_link.link_second_derivative(x[:, j[0], j[1]])
        return out

    def element_link(self, x: np.ndarray, i: int, j: int) -> np.ndarray:
        if i == j:
            return self.diag_link.link(x)
        else:
            return self.tril_link.link(x)

    def element_link_derivative(self, x: np.ndarray, i: int, j: int) -> np.ndarray:
        if i == j:
            return self.diag_link.link_derivative(x)
        else:
            return self.tril_link.link_derivative(x)

    def element_link_second_derivative(
        self, x: np.ndarray, i: int, j: int
    ) -> np.ndarray:
        if i == j:
            return self.diag_link.link_second_derivative(x)
        else:
            return self.tril_link.link_second_derivative(x)

    def element_inverse_derivative(self, x: np.ndarray, i: int, j: int) -> np.ndarray:
        if i == j:
            return self.diag_link.inverse_derivative(x)
        else:
            return self.tril_link.inverse_derivative(x)

    def element_inverse(self, x: np.ndarray, i: int, j: int) -> np.ndarray:
        if i == j:
            return self.diag_link.inverse(x)
        else:
            return self.tril_link.inverse(x)
