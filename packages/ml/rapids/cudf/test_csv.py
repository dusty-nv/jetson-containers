#!/usr/bin/env python3
import cudf
import requests
from io import StringIO
from jetson_containers import handle_text_request_or_fail

url = "https://github.com/plotly/datasets/raw/master/tips.csv"
csv_text = handle_text_request_or_fail(url)

tips_df = cudf.read_csv(StringIO(csv_text))
tips_df["tip_percentage"] = tips_df["tip"] / tips_df["total_bill"] * 100

# display average tip by dining party size
print(tips_df.groupby("size").tip_percentage.mean())