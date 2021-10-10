import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import prange, parallel

DTYPE = np.float
ctypedef const np.float16 DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
def dot_zero_cy1(np.ndarray[DTYPE_t, ndim=2] matrix) -> np.ndarray[DTYPE_t, 1]:
    cdef int size = matrix.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=1] out = np.zeros(size, dtype=DTYPE)
    cdef float total = 0.0
    # with nogil:
    for idx in range(size):
        total = 0.0
        for idy in range(size):
            if idx == idy:
                total  += 0
            else:
                total += np.dot(matrix[idx], matrix[idy])
        out[idx] = total
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
def dot_zero_cy2(np.ndarray[DTYPE_t, ndim=2] matrix) -> np.ndarray[DTYPE_t, 1]:
    cdef int K = matrix.shape[0]
    cdef int J = matrix.shape[1]
    cdef np.ndarray flat = matrix.flatten()
    cdef np.ndarray out = np.zeros(K)
    cdef np.ndarray x
    cdef int i
    cdef float total
    # with nogil:
    for k in prange(K):
        x = matrix[k, :]
        i = 0
        total = 0.0
        while i < k*J:
            total += x[i%J] * flat[i] # modulo potentiellement merdique
            i+=1
        i += J
        while i < K*J:
            total += x[i%J] * flat[i]
            i+=1
        out[k] = total
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
def dot_zero_cy3(np.ndarray matrix) -> np.ndarray[DTYPE_t, 1]:
    cdef int K = matrix.shape[0]
    cdef int J = matrix.shape[1]
    cdef int I = matrix.shape[0]
    cdef np.ndarray out = np.zeros(K)
    # with nogil:
    for k in prange(K):
        total = 0
        for i in range(I):
            if i == k:
                continue
            for j in range(J):
                total += matrix[k, j] * matrix[i, j]
        out[k] = total
    return out