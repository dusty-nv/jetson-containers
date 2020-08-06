print('testing numba...')

import math
import numba
from numba import vectorize, cuda
import numpy as np

print('numba version: ' + str(numba.__version__))
print('testing cuda ufunc...')

@vectorize(['float32(float32, float32, float32)',
            'float64(float64, float64, float64)'],
           target='cuda')
def cu_discriminant(a, b, c):
    return math.sqrt(b ** 2 - 4 * a * c)

N = 10000
dtype = np.float32

# prepare the input
A = np.array(np.random.sample(N), dtype=dtype)
B = np.array(np.random.sample(N) + 10, dtype=dtype)
C = np.array(np.random.sample(N), dtype=dtype)

D = cu_discriminant(A, B, C)

print('cuda ufunc result:')
print(D)  # print result

print('numba OK\n')