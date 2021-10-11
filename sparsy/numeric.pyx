import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import prange, parallel

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef float[:] dot_zero_cy1(float[:, :] matrix) nogil:
    cdef int size = matrix.shape[0]
    cdef int idx = 0
    cdef int idy = 0
    cdef float[:] out = np.zeros(size, dtype=np.float32)
    cdef float total = 0.0
    for idx in prange(size):
        total = 0.0
        idy = 0
        while idy < size:
            if idx == idy:
                total  += 0
            else:
                total += np.dot(matrix[idx], matrix[idy])
            idy += 1
        out[idx] = total
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef float[:] dot_zero_cy2(float[:, :] matrix) nogil:
    cdef Py_ssize_t  K = matrix.shape[0]
    cdef Py_ssize_t  J = matrix.shape[1]
    cdef float[:] flat = matrix.flatten()
    cdef float[:] out = np.zeros(K)
    cdef float[:] x
    cdef int i
    cdef float total
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
@cython.nonecheck(False)
cpdef float[:] dot_zero_cy3(float[:, :] matrix) nogil:
    cdef Py_ssize_t  K = matrix.shape[0]
    cdef Py_ssize_t  J = matrix.shape[1]
    cdef Py_ssize_t  I = matrix.shape[0]
    cdef float[:] out = np.zeros(K)
    for k in prange(K):
        total = 0
        for i in range(I):
            if i == k:
                continue
            for j in range(J):
                total += matrix[k, j] * matrix[i, j]
        out[k] = total
    return out