from __future__ import annotations

import logging

import numpy as np
from numba import njit, prange
# from sparsy.numeric import dot_zero

from sparsy.utils import extract_type, get_memory_usage

@njit
def tclass_corr(values: np.ndarray) -> np.ndarray:
    var = values.T.dot(values)
    base_var = var.copy()
    for i in range(var.shape[0]):
        for j in range(var.shape[0]):
            if var[i, i] == 0 or var[j, j] == 0:
                continue
            var[i, j] = var[i, j] / (np.sqrt(base_var[i, i]) * np.sqrt(base_var[j, j]))
    return var

@njit
def dot_zero3(matrix: np.ndarray) -> np.ndarray: # best for now
    K = matrix.shape[0]
    J = matrix.shape[1]
    I = matrix.shape[0]
    out = np.zeros(K, dtype=np.float32)
    for k in prange(K):
        total = 0
        for i in range(I):
            if i == k:
                continue
            for j in range(J):
                total += matrix[k, j] * matrix[i, j]
        out[k] = total
    return out

def dot_zero_old(matrix:np.ndarray) -> np.ndarray:
    out = matrix.dot(matrix.T) * 100
    np.fill_diagonal(out, 0)
    out = out.sum(axis=1)
    return out

def mahalanobis(biggie:np.ndarray, small:np.ndarray) -> np.ndarray:
    out = biggie.dot(small.dot(biggie.T))
    out = np.round(out, decimals=2) * 100
    out.setdiag(0)
    out = out.sum(axis=1)
    return out

def compute(matrix: np.ndarray) -> tuple[np.ndarray, ...]:
    values : np.ndarray = ((matrix / matrix.sum(axis=1)[:, None]) * 100).astype("float32")
    # compute matrix of correlations between classes (m x m)
    var = tclass_corr(values)

    # correlation between firms overs classes (n x n)
    logging.info("most cpu intensive tasks now")
    
    import pdb;pdb.set_trace()
    # np.dot(arr, arr.T).diagonal() == (arr * arr).sum(axis=1)
    norm_values = values / np.sqrt((values * values).sum(axis=1))[:, None]

    # warm up the jit
    dot_zero(var)
    # generate standard measures
    std = dot_zero(norm_values)
    cov_std = dot_zero(values)
    # generate MAHALANOBIS measure
    mal = mahalanobis(norm_values, var)
    cov_mal = mahalanobis(values, var)

    return std, cov_std, mal, cov_mal
