import numpy as np


def sparsy(array: np.ndarray) -> np.ndarray:
    matrix = array.dot(array.T)
    np.fill_diagonal(matrix, 0)
    return matrix.sum(axis=1)
