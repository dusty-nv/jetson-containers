print('testing numba...')

import math
import numba
from numba import vectorize, guvectorize, cuda
import numpy as np

print('numba version: ' + str(numba.__version__))

# test scalar vectorization
print('testing cuda vectorized ufunc...')

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

print('cuda vectorized ufunc result:')
print(D)  # print result

# test array vectorization
print('testing cuda guvectorized ufunc...')

@guvectorize(['uint8[:], uint8[:], uint8[:]',
              'float32[:], float32[:], float32[:]'], 
              '(n),(n)->(n)',
             target='cuda')
def cu_add_arrays(x, y, res):
    for i in range(x.shape[0]):  # number of channels (3)
        res[i] = x[i] + y[i]
    
A = np.full((2,4,3), 1, dtype)
B = np.full(A.shape, 2, dtype)
C = cu_add_arrays(A, B)

print('cuda guvectorized ufunc result:')
print(C)  # results should be '3'

print('numba OK\n')