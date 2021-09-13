from __future__ import annotations

from typing import cast

import numpy as np
from numpy import linalg
from numpy import ma, ndarray
from scipy.sparse.csr import csr_matrix
from scipy.sparse.lil import lil_matrix


def gen_data(n_rows: int, n_classes: int, n_firms: int) -> np.ndarray:
    """
    This functions creates a synthetic dataset for stress testing
    """
    return np.array(
        [
            np.random.randint(low=1, high=n_firms, size=n_rows),
            np.random.randint(low=1, high=n_classes, size=n_rows),
            np.random.randint(low=1950, high=2020, size=n_rows),
        ]
    ).T


def tclass_corr(var: lil_matrix) -> lil_matrix:
    # insertions in sparse matrix should be on type "lil_matrix" -> + efficient
    base_var = var.copy()
    for i in range(var.shape[0]):
        for j in range(var.shape[0]):
            if var[i, i] == 0 or var[j, j] == 0:
                continue
            var[i, j] = var[i, j] / (np.sqrt(base_var[i, i]) * np.sqrt(base_var[j, j]))
    return var


def dot_zero(array_a: csr_matrix, array_b: csr_matrix) -> ndarray:
    # arithmetic operations should be on type "csr_matrix" -> + efficient
    dot_product: csr_matrix = np.dot(array_a, array_b)
    rounded: csr_matrix = np.round(dot_product, decimals=2)
    del dot_product
    multiplied: csr_matrix = cast(csr_matrix, rounded * 100)
    del rounded
    multiplied.setdiag(0)
    summed: ndarray = multiplied.sum(axis=1)
    del multiplied
    #logged = ma.log(summed)
    #return np.array(logged).T.squeeze()
    return np.array(summed).T.squeeze()


def compute(matrix: csr_matrix) -> tuple[np.ndarray, ...]:
    values = csr_matrix((matrix / matrix.sum(axis=1)) * 100)
    # compute matrix of correlations between classes (m x m)

    _var: csr_matrix = np.dot(values.T, values)
    var = csr_matrix(tclass_corr(lil_matrix(_var)))

    # correlation between firms overs classes (n x n)
    norm_values = csr_matrix(
        values / np.sqrt(np.dot(values, values.T).diagonal())[:, None]
    )

    # generate standard measures
    std = dot_zero(norm_values, norm_values.T)
    cov_std = dot_zero(values, values.T)

    # generate MAHALANOBIS measure ==> gives n x n matrix
    mal = dot_zero(norm_values, np.dot(var, norm_values.T))
    cov_mal = dot_zero(values, np.dot(var, values.T))

    return std, cov_std, mal, cov_mal
