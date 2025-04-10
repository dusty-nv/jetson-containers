#!/usr/bin/env python3
print('testing dask_cudf...')
import dask_cudf
import cudf
print('dask_cudf version:', dask_cudf.__version__)

# https://docs.rapids.ai/api/cudf/stable/user_guide/10min/
df = cudf.DataFrame(
    {
        "a": list(range(10)),
        "b": list(reversed(range(10))),
        "c": list(range(10)),
    }
)

ddf = dask_cudf.from_cudf(df, npartitions=2)

print(ddf)
print(ddf.head())
print('\ndask_cudf OK\n')

