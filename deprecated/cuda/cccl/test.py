#!/usr/bin/env python3
print('Testing CCCL...')
import cupy as cp
import numpy as np

import cuda.cccl.parallel.experimental as parallel


def sum_reduction_example():
    """Sum all values in an array using reduction."""

    def add_op(a, b):
        return a + b

    dtype = np.int32
    h_init = np.array([0], dtype=dtype)
    d_input = cp.array([1, 2, 3, 4, 5], dtype=dtype)
    d_output = cp.empty(1, dtype=dtype)

    # Instantiate reduction
    reduce_into = parallel.reduce_into(d_output, d_output, add_op, h_init)

    # Determine temporary device storage requirements
    temp_storage_size = reduce_into(None, d_input, d_output, len(d_input), h_init)

    # Allocate temporary storage
    d_temp_storage = cp.empty(temp_storage_size, dtype=np.uint8)

    # Run reduction
    reduce_into(d_temp_storage, d_input, d_output, len(d_input), h_init)

    expected_output = 15  # 1+2+3+4+5
    assert (d_output == expected_output).all()
    print(f"Sum: {d_output[0]}")
    return d_output[0]


if __name__ == "__main__":
    print("Running basic reduction examples...")
    sum_reduction_example()
    print("All examples completed successfully!")
print('CCCL OK\n')
