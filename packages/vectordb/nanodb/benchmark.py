#!/usr/bin/env python3
import os
import time
import pprint
import argparse
import numpy as np

from cuda.cudart import cudaGetDeviceProperties, cudaDeviceSynchronize

from .vector_index import cudaVectorIndex, DistanceMetrics


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-d', '--dim', type=int, default=5120, help='the dimensionality of the embedding vectors') 
parser.add_argument('-n', '--num-vectors', type=int, default=512, help='the number of vectors to add to the index')
parser.add_argument('-k', type=int, default=4, help='the number of search results to return per query')
parser.add_argument('--dtype', type=str, default='float32', help='datatype of the vectors')
parser.add_argument('--metric', type=str, default='l2', choices=DistanceMetrics, help='the distance metric to use during search')

parser.add_argument('--seed', type=int, default=1234, help='change the random seed used')
parser.add_argument('--num-queries', type=int, default=1, help='the number of searches to run')

args = parser.parse_args()
print(args)

np.random.seed(args.seed)
dtype = np.dtype(args.dtype)

_, device_props = cudaGetDeviceProperties(0)
print(f"-- cuda device:  {device_props.name}")

index = cudaVectorIndex(args.dim, args.num_vectors, args.num_queries, dtype=dtype, metric=args.metric)

print('-- generating random test vectors')
vectors = np.random.random((args.num_vectors, args.dim)).astype(args.dtype)
#vectors[:, 0] += np.arange(args.num_vectors) / 1000.
queries = np.random.random((args.num_queries, args.dim)).astype(args.dtype)
#queries[:, 0] += np.arange(args.num_queries) / 1000.

print('vectors', vectors.shape, vectors.dtype)
print('queries', queries.shape, queries.dtype)

for n in range(args.num_vectors):
    index.add(vectors[n])

print(f"-- added {index.shape} vectors")
print(f"-- validating index")

index.validate()

for n in range(args.num_vectors):
    indexes, distances = index.search(vectors[n], k=args.k)
    if indexes[0] != n:
        print(f"incorrect index[{n}]\n", indexes, "\n", distances)
        assert(indexes[0] == n)
    
print(f"-- searching {queries.shape} vectors (metric={args.metric}, k={args.k})")
time_begin=time.perf_counter()
#for i in range(3):
indexes, distances = index.search(queries, k=args.k)
time_end=time.perf_counter()

print(indexes)
print(distances)
print(f"time:  {(time_end-time_begin)*1000} ms")

"""
for m in range(args.num_queries):
    search = index.search(xq[m], metric=args.metric)
    #print(search.shape)
    #print(search)
    
for m in range(args.num_vectors):
    search = index.search(xb[m], metric=args.metric)
    assert(search[0] == m)
    #print(search)
    #print(search.shape)
    #
"""