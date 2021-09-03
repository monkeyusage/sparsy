from __future__ import annotations

import numpy as np
from numpy import ma
from numba import njit
from scipy.sparse import bsr_matrix, diags


def gen_data(n_rows: int, n_classes: int, n_firms: int) -> np.ndarray:
    return np.array(
        [
            np.random.randint(low=1, high=n_firms, size=n_rows, dtype=np.uint64),
            np.random.randint(low=1, high=n_classes, size=n_rows, dtype=np.uint16),
            np.random.randint(low=1950, high=2020, size=n_rows, dtype=np.uint16),
        ]
    ).T

@njit
def zero_diag(array: bsr_matrix) -> bsr_matrix:
    assert array.shape[0] == array.shape[1], "array must be square matrix"
    for i in range(array.shape[0]):
        array[i, i] = 0
    return array


def tclass_corr(tech: int, var: bsr_matrix) -> bsr_matrix:
    base_var = var.copy()
    for i in range(tech):
        for j in range(tech):
            if base_var[i, i] == 0 or base_var[j, j] == 0:
                continue
            var[i, j] = var[i, j] / (np.sqrt(base_var[i, i]) * np.sqrt(base_var[j, j]))
    return var


def firm_corr(
    num: int, tech: int, values: bsr_matrix, base_std: bsr_matrix
) -> bsr_matrix:
    norm_values = bsr_matrix(np.zeros(values.shape))
    for j in range(tech):
        for i in range(num):
            if base_std[i, i] == 0:
                continue
            norm_values[i, j] = values[i, j] / np.sqrt(base_std[i, i])
    return norm_values


def dot_zero(array_a: bsr_matrix, array_b: bsr_matrix) -> bsr_matrix:
    dot_product = np.dot(array_a, array_b)
    rounded = np.round(dot_product, decimals=2)
    multiplied = np.multiply(rounded, 100)
    zero_out = zero_diag(multiplied)
    summed = zero_out.sum(axis=1)
    logged = ma.log(summed).filled(0)
    return logged

def squeeze(matrix:np.matrix) -> np.ndarray:
    return np.array(matrix).T.squeeze()

def zero_squeeze(a:bsr_matrix, b:bsr_matrix) -> np.ndarray:
    return squeeze(dot_zero(a,b))

def compute(matrix: bsr_matrix, tech: int) -> tuple[np.ndarray, ...]:

    # values= (matrix / total[:, None]) * 100
    values = diags(1/matrix.sum(axis=1).A.ravel())
    values = (values @ matrix) * 100

    num = values.shape[0]

    # compute matrix of correlations between classes
    var = np.dot(values.T, values)
    var = tclass_corr(tech, var)

    # correlation between firms overs classes (n x n || 694x694)
    base_std : bsr_matrix = np.dot(values, values.T)
    norm_values = firm_corr(num, tech, values, base_std)

    # generate standard measures
    std = zero_squeeze(norm_values, norm_values.T)
    cov_std = zero_squeeze(values, values.T)

    # generate MAHALANOBIS measure ==> gives n x n matrix
    mal = zero_squeeze(np.dot(norm_values, var), norm_values.T)
    cov_mal = zero_squeeze(np.dot(values, var), values.T)

    return std, cov_std, mal, cov_mal
