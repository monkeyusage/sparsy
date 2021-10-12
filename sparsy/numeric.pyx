import numpy as np

cimport cython
cimport numpy as np

from cython.parallel import parallel, prange

from libc.stdlib cimport free, malloc
from os import cpu_count

cdef int __CPUS = cpu_count() - 1 

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def dot_zero(float[:, :] matrix) -> list[float]:
    """
    Uses MemoryView from numpy array as input
    Rough equivalent of numpy:
    >>> def dot_zero(matrix: np.ndarray) -> np.ndarray:
            out = matrix.dot(matrix.T)
            np.fill_diagonal(out, 0)
            return out.sum(axis=1)
    However in this version we do not expand memory from N x M (e.g 68K x 7) to N x N (68K x 68K)
    We reduce the output while computing results thus keeping memory to the minimum N

    matrix shape : N x M
    out shape : N x 1
    """
    cdef Py_ssize_t  K = matrix.shape[0]
    cdef Py_ssize_t  J = matrix.shape[1]
    cdef Py_ssize_t  I = matrix.shape[0]
    cdef float* out = <float *> malloc(K * sizeof(float))
    if not out:
        raise MemoryError()
    cdef float total
    cdef Py_ssize_t i
    cdef Py_ssize_t j
    cdef Py_ssize_t k
    try:
        with nogil, parallel(num_threads=__CPUS):
            for k in prange(K):
                total = 0
                for i in range(I):
                    if i == k:
                        continue
                    for j in range(J):
                        total = total + (matrix[k, j] * matrix[i, j])
                out[k] = total
        return [item for item in out[:K]]
    finally:
        free(out)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def mahalanobis(float[:, :] biggie, float[:, :] small) -> list[float]:
    """
    Uses MemoryView from numpy array as input
    Rough equivalent of numpy:
    >>> def dot_zero(biggie: np.ndarray, small: np.ndarray) -> np.ndarray:
            out = biggie.dot(small)
            np.fill_diagonal(out, 0)
            return out.sum(axis=1)
    However in this version we do not expand memory from N x M (e.g 68K x 7) to N x N (68K x 68K)
    We reduce the output while computing results thus keeping memory to the minimum N
    
    biggie shape : N x M
    small shape : M x N
    with small != biggie.T

    we require user to perform f(small, biggie) = small.dot(biggie.T) before calling this function since:
        - f is fast AND memory efficient
        - memory views do not implement numpy functions
    """
    cdef Py_ssize_t  K = biggie.shape[0]
    cdef Py_ssize_t  J = biggie.shape[1]
    cdef Py_ssize_t  I = small.shape[1]
    cdef float* out = <float *> malloc(K * sizeof(float))
    if not out:
        raise MemoryError()
    cdef float total
    cdef Py_ssize_t i
    cdef Py_ssize_t j
    cdef Py_ssize_t k
    try:
        with nogil, parallel(num_threads=__CPUS):
            for k in prange(K):
                total = 0
                for i in range(I):
                    if i == k:
                        continue
                    for j in range(J):
                        total = total + (biggie[k, j] * small[j, i])
                out[k] = total
        return [item for item in out[:K]]
    finally:
        free(out)
