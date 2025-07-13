#!/usr/bin/env python3
import argparse
import faiss
import numpy as np
import time

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-k', type=int, default=4)
parser.add_argument('-d', '--dim', type=int, default=64)  # 2621440
parser.add_argument('--num-vectors', type=int, default=100000)  # 512
parser.add_argument('--num-queries', type=int, default=1)
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--cpu', action='store_true')

args = parser.parse_args()
print(args)

np.random.seed(args.seed)

print(f"building random numpy arrays ({args.num_vectors}, {args.dim})")

xb = np.random.random((args.num_vectors, args.dim)).astype('float32')
xb[:, 0] += np.arange(args.num_vectors) / 1000.
xq = np.random.random((args.num_queries, args.dim)).astype('float32')
xq[:, 0] += np.arange(args.num_queries) / 1000.

print(f"numpy array size:  {(xb.size * xb.itemsize) / (1024*1024):.3f} MB")
print(f"creating index")

index = faiss.IndexFlatL2(args.dim)   # build the index

if not args.cpu:
    res = faiss.StandardGpuResources()  # use a single GPU
    index = faiss.index_cpu_to_gpu(res, 0, index)

# https://github.com/facebookresearch/faiss/wiki/FAQ#why-does-the-ram-usage-not-go-down-when-i-delete-an-index
print(f"mem usage:  {faiss.get_mem_usage_kb() / 1024:.3f} MB")
print(index.is_trained)

time_begin = time.perf_counter()
index.add(xb[:-1])                  # add vectors to the index
print(f"time to add {xb.shape} vectors:  {time.perf_counter()-time_begin:.3} sec")
print(index.ntotal)

time_begin = time.perf_counter()
index.add(xb[-1:])                  # add vectors to the index
print(f"time to add 1 vector:  {time.perf_counter()-time_begin:.3} sec")
print(index.ntotal)

def search(queries, k=args.k):
    time_begin = time.perf_counter()
    D, I = index.search(queries, k) # sanity check
    print(I)
    print(D)
    print(f"time to search {len(queries)}:  {time.perf_counter()-time_begin:.3} sec")


"""
Sanity check on the first 5 vectors:

[[  0 393 363  78]
 [  1 555 277 364]
 [  2 304 101  13]
 [  3 173  18 182]
 [  4 288 370 531]]

[[ 0.          7.17517328  7.2076292   7.25116253]
 [ 0.          6.32356453  6.6845808   6.79994535]
 [ 0.          5.79640865  6.39173603  7.28151226]
 [ 0.          7.27790546  7.52798653  7.66284657]
 [ 0.          6.76380348  7.29512026  7.36881447]]
"""
search(xb[:5])
search(xq)
