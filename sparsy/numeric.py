from __future__ import annotations

import logging
from time import perf_counter

import numpy as np
from numpy import ndarray
from numba import njit

from sparsy.utils import extract_type, get_memory_usage


def gen_data(n_rows: int, n_classes: int, n_firms: int) -> np.ndarray:
    """
    This functions creates a synthetic dataset for stress testing
    """
    return np.array(
        [
            np.random.randint(low=1, high=n_firms, size=n_rows),
            np.random.randint(low=1, high=n_classes, size=n_rows),
            np.random.randint(low=1950, high=2020, size=n_rows),
        ],
        dtype=np.int32,
    ).T


def tclass_corr(values: ndarray) -> ndarray:
    logging.info("computing tclass correlation matrix")
    # insertions in sparse matrix should be on type "lil_matrix" -> + efficient
    var = np.dot(values.T, values)
    base_var = var.copy()
    for i in range(var.shape[0]):
        for j in range(var.shape[0]):
            if var[i, i] == 0 or var[j, j] == 0:
                continue
            var[i, j] = var[i, j] / (np.sqrt(base_var[i, i]) * np.sqrt(base_var[j, j]))
    return var

@njit
def dot_zero(matrix: ndarray) -> ndarray:
    out = matrix.dot(matrix.T)
    out= np.round(out, decimals=2) * 100
    out.setdiag(0)
    out = out.sum(axis=1)
    return out

@njit
def mahalanobis(biggie:ndarray, small:ndarray) -> ndarray:
    out = biggie.dot(small.dot(biggie.T))
    out = np.round(out, decimals=2) * 100
    out.setdiag(0)
    out = out.sum(axis=1)
    return out


def compute(matrix: ndarray) -> tuple[np.ndarray, ...]:
    logging.info("entering core computing intensive function")
    logging.info(
        "matrix shape is %s, taking about %s bytes",
        matrix.shape,
        matrix.shape[0] * extract_type(matrix.dtype),
    )
    values : ndarray = (matrix / matrix.sum(axis=1)) * 100
    # compute matrix of correlations between classes (m x m)

    var = tclass_corr(values)

    # correlation between firms overs classes (n x n)
    logging.info("most cpu intensive tasks now")
    
    norm_values = values / np.sqrt(np.dot(values, values.T).diagonal())[:, None]

    logging.info(
        "norm_values shape is %s, taking about %s bytes",
        norm_values.shape,
        norm_values.shape[0] * extract_type(norm_values.dtype),
    )

    # generate standard measures
    std = dot_zero(norm_values)
    cov_std = dot_zero(values)

    # generate MAHALANOBIS measure ==> gives n x n matrix

    # this one is wrong
    mal = mahalanobis(norm_values, var)
    cov_mal = mahalanobis(values, var)

    logging.info("Memory used in MB: %s", get_memory_usage())
    return std, cov_std, mal, cov_mal
