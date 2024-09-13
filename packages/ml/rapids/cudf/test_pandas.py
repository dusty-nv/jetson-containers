#!/usr/bin/env python3
import cudf.pandas
cudf.pandas.install()

if not cudf.pandas.LOADED:
    raise ImportError("These tests must be run with cudf.pandas loaded")
    
import pandas as pd
import requests
from io import StringIO

url = "https://github.com/plotly/datasets/raw/master/tips.csv"
content = requests.get(url).content.decode("utf-8")

tips_df = pd.read_csv(StringIO(content))
tips_df["tip_percentage"] = tips_df["tip"] / tips_df["total_bill"] * 100

# display average tip by dining party size
print(tips_df.groupby("size").tip_percentage.mean())

# https://github.com/rapidsai/cudf/blob/branch-23.12/python/cudf/cudf_pandas_tests/test_cudf_pandas_cudf_interop.py
def test_cudf_pandas_loaded_to_cudf(hybrid_df):
    #hybrid_df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    cudf_df = cudf.from_pandas(hybrid_df)
    pd.testing.assert_frame_equal(hybrid_df, cudf_df.to_pandas())
    
test_cudf_pandas_loaded_to_cudf(tips_df)

print("\ncuDF <-> Pandas interoperability OK")