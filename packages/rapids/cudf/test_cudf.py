#!/usr/bin/env python3
print('testing cudf...')
import cudf
print('cudf version:', cudf.__version__)

# https://docs.rapids.ai/api/cudf/stable/user_guide/10min/
s = cudf.Series([1, 2, 3, None, 4])
print(s)

df = cudf.DataFrame(
    {
        "a": list(range(10)),
        "b": list(reversed(range(10))),
        "c": list(range(10)),
    }
)

print(df)
print(df.sort_values(by="b"))

#x = df.to_numpy()   # not in cudf 21.10
x = df.to_pandas()
x = df.to_arrow()

df.to_parquet("/tmp/test_parquet")
df.to_orc("/tmp/test_orc")

print('\ncudf OK\n')