#!/usr/bin/env python3
import os
import math
import time
import socket
import datetime
import resource
import argparse

import ctypes as C
import numpy as np
    
from cuda.cudart import (
    cudaGetDeviceProperties, 
    cudaDeviceSynchronize,
    cudaGetLastError,
)

from faiss_lite import (
    cudaKNN, 
    cudaL2Norm, 
    cudaAllocMapped, 
    DistanceMetrics, 
    assert_cuda
)
    
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-d', '--dim', type=int, default=1024, help='the dimensionality of the embedding vectors') 
parser.add_argument('-n', '--num-vectors', type=int, default=64, help='the number of vectors to add to the index')
parser.add_argument('-k', type=int, default=4, help='the number of nearest-neighbors to find')
parser.add_argument('--dtype', type=str, default='float32', choices=['float32', 'float16'], help='datatype of the vectors')
parser.add_argument('--metric', type=str, default='l2', choices=DistanceMetrics, help='the distance metric to use during search')
parser.add_argument('--num-queries', type=int, default=1) 

parser.add_argument('--seed', type=int, default=1234, help='change the random seed used')
parser.add_argument('--runs', type=int, default=15)
parser.add_argument('--warmup', type=int, default=3)
parser.add_argument('--save', type=str, default='', help='CSV file to save benchmarking results to')
parser.add_argument('--skip-validation', action='store_true')


args = parser.parse_args()
print(args)

np.random.seed(args.seed)

if args.dtype == 'float32':
    dtype = np.float32
elif args.dtype.lower() == 'float16':
    dtype = np.float16
else:
    raise ValueError(f"unsupported dtype:  {args.dtype}")
 
dsize = np.dtype(dtype).itemsize     
print("-- using datatype:", dtype, dsize)

_, device_props = cudaGetDeviceProperties(0)
print(f"-- cuda device:  {device_props.name}")

xb = np.random.random((args.num_vectors, args.dim)).astype(dtype)
#xb[:, 0] += np.arange(args.num_vectors) / 1000.
xq = np.random.random((args.num_queries, 1, args.dim)).astype(dtype)
#xq[:, 0] += np.arange(args.num_queries) / 1000.

vector_size = (args.dim * xb.itemsize) / (1024*1024)  # size of one vector in MB

vectors = cudaAllocMapped((args.num_vectors, args.dim), dtype)
queries = cudaAllocMapped((args.num_queries, args.dim), dtype)

distances = cudaAllocMapped((args.num_queries, args.k), np.float32)
indexes = cudaAllocMapped((args.num_queries, args.k), np.int64)

for n in range(args.num_vectors):
    vectors.array[n] = xb[n]
 
for n in range(args.num_queries):
    queries.array[n] = xq[n]
   
vector_norms = None

avg_l2_time = 0
avg_l2_rate = 0

if args.metric == 'l2':
    vector_norms = cudaAllocMapped(args.num_vectors, np.float32)

    if not args.skip_validation:
        assert(cudaL2Norm(
            C.cast(vectors.ptr, C.c_void_p),
            dsize, args.num_vectors, args.dim,
            C.cast(vector_norms.ptr, C.POINTER(C.c_float)),
            True, None
        ))
        assert_cuda(cudaDeviceSynchronize())
    
    #print('vector_norms', vector_norms, vector_norms.shape, vector_norms.dtype)

    # check the time for single vectors
    for n in range(args.num_vectors):
        time_begin = time.perf_counter()
        cudaL2Norm(
            C.cast(vectors.ptr+n*args.dim*dsize, C.c_void_p),
            dsize, 1, args.dim,
            C.cast(vector_norms.ptr+n*4, C.POINTER(C.c_float)) if vector_norms else None,
            True, None
        )
        assert_cuda(cudaDeviceSynchronize())
        time_elapsed = (time.perf_counter() - time_begin) * 1000
        if n % int(args.num_vectors/10) == 0:
            print(f"l2_norm time for (1,{args.dim}) vector:  {time_elapsed:.3f} ms")
        if n > 0:
            avg_l2_time += time_elapsed / (args.num_vectors-1)
        
# sanity check
if not args.skip_validation:
    print("-- running validation")

    for n in range(args.num_vectors):
        assert(cudaKNN(
            C.cast(vectors.ptr, C.c_void_p),
            C.cast(vectors.ptr+n*args.dim*dsize, C.c_void_p),
            dsize,
            args.num_vectors,
            1,
            args.dim,
            args.k,
            DistanceMetrics[args.metric],
            C.cast(vector_norms.ptr, C.POINTER(C.c_float)) if vector_norms else None,
            C.cast(distances.ptr, C.POINTER(C.c_float)),
            C.cast(indexes.ptr, C.POINTER(C.c_longlong)),
            None
        ))
        assert_cuda(cudaDeviceSynchronize())
        assert(indexes.array[0][0]==n)

avg_time = 0

for r in range(args.runs + args.warmup):
    time_begin = time.perf_counter()
    assert(cudaKNN(
        C.cast(vectors.ptr, C.c_void_p),
        C.cast(queries.ptr, C.c_void_p),
        dsize,
        args.num_vectors,
        args.num_queries,
        args.dim,
        args.k,
        DistanceMetrics[args.metric],
        C.cast(vector_norms.ptr, C.POINTER(C.c_float)) if vector_norms else None,
        C.cast(distances.ptr, C.POINTER(C.c_float)),
        C.cast(indexes.ptr, C.POINTER(C.c_longlong)),
        None
    ))
    time_enqueue = (time.perf_counter() - time_begin) * 1000
    assert_cuda(cudaDeviceSynchronize())
    time_elapsed = (time.perf_counter() - time_begin) * 1000
    
    print(f"cudaKNN  enqueue:  {time_enqueue:.3f} ms   process:  {time_elapsed:.3f} ms")
    
    if r >= args.warmup:
        avg_time += time_elapsed / args.runs

avg_rate = args.num_queries * 1000 / avg_time

print("")
print(f"N={args.num_vectors} M={args.num_queries} D={args.dim} K={args.k} metric={args.metric} dtype={args.dtype}\n")
print(f"average search time:   {avg_time:.3f} ms, {avg_rate:.1f} vectors/sec, {avg_rate*vector_size:.1f} MB/s")

if args.metric == 'l2':
    avg_l2_rate = 1000 / avg_l2_time
    print(f"average l2 norm time:  {avg_l2_time:.3f} ms, {avg_l2_rate:.1f} vectors/sec, {avg_l2_rate*vector_size:.1f} MB/s")
    
memory_usage = (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss + resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss) / 1024  
print(f"peak memory usage:     {memory_usage:.1f} MB")

if args.save:
    if not os.path.isfile(args.save):  # csv header
        with open(args.save, 'w') as file:
            file.write(f"timestamp, hostname, api, metric, dtype, vector_dim, num_vectors, ")
            file.write(f"index_batch, index_time, index_rate, index_bandwidth, ")
            file.write(f"search_batch, search_time, search_rate, search_bandwidth, memory\n")
    with open(args.save, 'a') as file:
        file.write(f"{datetime.datetime.now().strftime('%Y%m%d %H:%M:%S')}, {socket.gethostname()}, faiss_lite, ")
        file.write(f"{args.metric}, {args.dtype}, {args.dim}, {args.num_vectors}, ")
        file.write(f"1, {avg_l2_time}, {avg_l2_rate}, {avg_l2_rate*vector_size}, ")
        file.write(f"{args.num_queries}, {avg_time}, {avg_rate}, {avg_rate*vector_size}, {memory_usage}\n")
        