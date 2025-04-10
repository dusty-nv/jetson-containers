#!/usr/bin/env python3
print('testing cuml...')

import cuml
print('cuml version:', cuml.__version__)

# test DBSCAN
import cudf
from cuml.cluster import DBSCAN

gdf_float = cudf.DataFrame()

gdf_float['0'] = [1.0, 2.0, 5.0]
gdf_float['1'] = [4.0, 2.0, 1.0]
gdf_float['2'] = [4.0, 2.0, 1.0]

dbscan_float = DBSCAN(eps=1.0, min_samples=1)
dbscan_float.fit(gdf_float)

print(dbscan_float.labels_)

print('\ncuml OK\n')