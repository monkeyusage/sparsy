import numpy as np
import numba


@numba.njit
def sparsy(array: np.ndarray) -> np.ndarray:
    N, M = array.shape
    out = np.zeros(array.shape[0])
    for i in range(N):
        total = 0
        for ii in range(N):
            if i == ii: continue
            for j in range(M):
                total += array[i, j] * array[ii, j]
        out[i] = total
    return out
