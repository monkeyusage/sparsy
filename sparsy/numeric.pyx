import numpy as np
cimport numpy as np

DTYPE = np.float
ctypedef np.float DTYPE_t

def dot_zero(np.ndarray matrix):
    cdef int size = matrix.shape[0]
    cdef np.ndarray out = np.zeros(size, dtype=DTYPE)
    cdef float total = 0.0
    for idx in range(size):
        total = 0.0
        for idy in range(size):
            if idx == idy:
                total  += 0
            else:
                total += np.dot(matrix[idx], matrix[idy])
        out[idx] = total
    return out