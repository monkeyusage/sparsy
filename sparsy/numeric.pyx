import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import prange, parallel
from libc.stdlib cimport malloc, free

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def dot_zero_cy3(float[:, :] matrix) -> list[float]:
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
        with nogil:
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