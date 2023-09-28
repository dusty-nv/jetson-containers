#!/usr/bin/env python3
import os
import time
import argparse

import numpy as np

from .nanodb import NanoDB, DistanceMetrics

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--scan", action='append', nargs='*', help="a directory or file to extract embeddings from")
parser.add_argument("--max-scan", type=int, default=None, help="the maximum number of items to scan (None for unlimited)")
parser.add_argument('--reserve', type=int, default=1024, help="the memory to reserve for the database in MB")
parser.add_argument('--dtype', type=str, default='float32', choices=['float32', 'float16'], help='datatype of the vectors')
parser.add_argument('--metric', type=str, default='l2', choices=DistanceMetrics, help="the distance metric to use during search")
parser.add_argument('--max-search-queries', type=int, default=1, help="the maximum number of searches performed in one batch")

parser.add_argument('--k', type=int, default=4, help='the number of search results to return per query')
parser.add_argument('--seed', type=int, default=1234, help='change the random seed used')

parser.add_argument('--crop', action='store_true')
parser.add_argument('--validate', action='store_true')
parser.add_argument('--test', action='store_true')

args = parser.parse_args()

if args.scan:
    args.scan = [x[0] for x in args.scan]
    
print(args)

np.random.seed(args.seed)

db = NanoDB(dtype=args.dtype, reserve=args.reserve, metric=args.metric, max_search_queries=args.max_search_queries)

for path in args.scan:
    print(f"-- scanning {path}")
    indexes = db.scan(path, max_items=args.max_scan, crop=args.crop)

if args.validate:
    print(f"-- validating index with k={args.k}")
    db.index.validate(k=args.k)
  
if args.test:
    print(f"-- testing index with k={args.k}")
    db.test(k=args.k)
