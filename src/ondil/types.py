from enum import Enum


class ParameterShapes(Enum):
    SCALAR = "A scalar parameter N x 1"
    VECTOR = "A vector parameter N x D"
    MATRIX = "A matrix parameter N x D x M"
    SQUARE_MATRIX = "A matrix parameter N x D x D"
    DIAGONAL_MATRIX = "A diagonal matrix N x D x D."
    LOWER_TRIANGULAR_MATRIX = "A lower triangular matrix N x D x D."
    UPPER_TRIANGULAR_MATRIX = "A upper triangular matrix N x D x D."
