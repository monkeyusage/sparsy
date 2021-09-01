from __future__ import annotations
import numpy as np
from numba import njit

def gen_data() -> np.ndarray:
    return np.array([
        np.random.randint(low=1, high=500_000, size=15_800_000),
        np.random.randint(low=1, high=1000, size=15_800_000),
        np.random.randint(low=1970, high=2020, size=15_800_000)
    ]).T

@njit
def zero_diag(array: np.ndarray) -> np.ndarray:
    assert array.shape[0] == array.shape[1], "array must be square matrix"
    for i in range(array.shape[0]):
        array[i,i] = 0
    return array

@njit
def tclass_corr(tech: int, var: np.ndarray) -> np.ndarray:
    base_var = var.copy()
    for i in range(tech):
        for j in range(tech):
            if base_var[i,i] == 0 or base_var[j,j] == 0: continue
            var[i,j] = var[i,j] / (np.sqrt(base_var[i,i]) * np.sqrt(base_var[j,j]))
    return var

@njit
def firm_corr(num:int, tech:int, values: np.ndarray, base_std: np.ndarray) -> np.ndarray:
    norm_values = values.copy()    
    for j in range(tech):
        for i in range(num):
            if base_std[i,i] == 0: continue
            norm_values[i,j] = values[i,j] / np.sqrt(base_std[i,i])
    return norm_values

@njit
def dot_zero(array_a: np.ndarray, array_b: np.ndarray) -> np.ndarray:
    dot_product = np.dot(array_a, array_b)
    rounded = np.empty_like(dot_product)
    np.round(dot_product, 2, rounded)
    multiplied = np.multiply(rounded, 100)
    zero_out = zero_diag(multiplied)
    summed = np.sum(zero_out, axis=1)
    logged = np.log(summed)
    return logged

@njit
def compute(matrix: np.ndarray, tech: int) -> tuple[np.ndarray, ...]:
    total = matrix.sum(axis=1)

    values = matrix
    # values= (matrix / total[:, None]) * 100
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if total[i] == 0:
                continue
            values[i,j] = (matrix[i,j] / total[i]) * 100

    num = values.shape[0]

    # compute matrix of correlations between classes
    var = np.dot(values.T, values)
    var = tclass_corr(tech, var)

    # correlation between firms overs classes (n x n || 694x694)
    base_std = np.dot(values, values.T)
    norm_values = firm_corr(num, tech, values, base_std)
    
    # generate standard measures
    std = dot_zero(norm_values, norm_values.T)
    cov_std = dot_zero(values, values.T)

    # generate MAHALANOBIS measure ==> gives n x n matrix
    mal = dot_zero(np.dot(norm_values, var),norm_values.T)
    cov_mal = dot_zero(np.dot(values, var), values.T)

    return std, cov_std, mal, cov_mal