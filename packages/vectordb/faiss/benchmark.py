#!/usr/bin/env python3
import os
import time
import socket
import datetime
import argparse

import faiss 
import numpy as np

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-k', type=int, default=4, help='the number of nearest neighbors to search for')
parser.add_argument('-d', '--dim', type=int, default=5120, help='the dimensionality of the embedding vectors')  # 2621440

parser.add_argument('--index', type=str, default='Flat', help='the type of index to use')  # https://github.com/facebookresearch/faiss/wiki/Faiss-indexes
parser.add_argument('--index-size', type=int, default=4096, help='the number of vectors to add to the index')
parser.add_argument('--index-batch', type=int, default=1, help='the number of vectors to add to index at a time')

parser.add_argument('--search-queries', type=int, default=4096, help='the number of search queries to run')
parser.add_argument('--search-batch', type=int, default=1, help='the number of search queries to run at a time')

parser.add_argument('--dtype', type=str, default='float32', help='datatype of the vectors')
parser.add_argument('--seed', type=int, default=1234, help='change the random seed used')
parser.add_argument('--cpu', action='store_true', help='disable GPU acceleration')
parser.add_argument('--save', type=str, default='', help='CSV file to save benchmarking results to')


args = parser.parse_args()
print(args)

np.random.seed(args.seed)

print(f"building random numpy arrays ({args.index_size}, {args.dim})")

xb = np.random.random((args.index_size, args.dim)).astype(args.dtype)
xb[:, 0] += np.arange(args.index_size) / 1000.
xq = np.random.random((args.search_queries, args.dim)).astype(args.dtype)
xq[:, 0] += np.arange(args.search_queries) / 1000.

print(xb.shape, xb.dtype)

vector_size = (args.dim * xb.itemsize) / (1024*1024)  # size of one vector in MB

print(f"vector size:         {vector_size*1024*1024:.0f} bytes")
print(f"numpy array size:    {(xb.size * xb.itemsize) / (1024*1024):.3f} MB")
print(f"creating index type: {args.index}")

index = faiss.index_factory(args.dim, args.index) #faiss.IndexFlatL2(args.dim)

if not args.cpu:
    res = faiss.StandardGpuResources()  # use a single GPU
    index = faiss.index_cpu_to_gpu(res, 0, index)
   
if not index.is_trained:
    print(f"training index {args.index}")
    index.train(xb)
    
# profile indexing
avg_index_time = 0
avg_index_rate = 0

avg_factor = 1.0 / (args.index_size / args.index_batch)

for i in range(0, args.index_size, args.index_batch):
    time_begin = time.perf_counter()
    index.add(xb[i:i+args.index_batch])
    index_time = time.perf_counter() - time_begin
    index_rate = args.index_batch / index_time
    avg_index_time += index_time * avg_factor
    avg_index_rate += index_rate * avg_factor
    if i % 32 == 0:
        print(f"added ({args.index_batch}, {args.dim}) vectors:  {index_time*1000:.2f} ms,  {index_rate:.1f} vectors/sec,  {index_rate*vector_size:.1f} MB/s")
 
def print_index_stats():      
    print(f"{args.index} index size:       ({index.ntotal}, {args.dim})")
    print(f"{args.index} index time:       {avg_index_time*1000:.2f} ms")
    print(f"{args.index} index rate:       {avg_index_rate:.1f} vectors/sec") 
    print(f"{args.index} index bandwidth:  {avg_index_rate*vector_size:.1f} MB/s") 
    print(f"{args.index} index trained:    {index.is_trained}")

# profile search
avg_search_time = 0
avg_search_rate = 0

avg_factor = 1.0 / (args.search_queries / args.search_batch)

for i in range(0, args.search_queries, args.search_batch): 
    time_begin = time.perf_counter()
    D, I = index.search(xq[i:i+args.search_batch], args.k)
    search_time = time.perf_counter() - time_begin
    search_rate = args.search_batch / search_time
    avg_search_time += search_time * avg_factor
    avg_search_rate += search_rate * avg_factor
    #if i % 32 == 0:
    print(f"search ({args.search_batch}, {args.dim}) vectors:  {search_time*1000:.2f} ms,  {search_rate:.1f} vectors/sec,  {search_rate*vector_size:.1f} MB/s")
    
def print_search_stats():
    print(f"{args.index} search size:      ({args.search_batch}, {args.dim})")
    print(f"{args.index} search time:      {avg_search_time*1000:.2f} ms")
    print(f"{args.index} search rate:      {avg_search_rate:.1f} vectors/sec") 
    print(f"{args.index} search bandwidth: {avg_search_rate*vector_size:.1f} MB/s") 
    
print("\n")
print_index_stats()
print("")
print_search_stats()

# https://github.com/facebookresearch/faiss/wiki/FAQ#why-does-the-ram-usage-not-go-down-when-i-delete-an-index
memory_usage = faiss.get_mem_usage_kb() / 1024
print(f"\nPeak memory usage:     {memory_usage:.1f} MB")


if args.save:
    if not os.path.isfile(args.save):  # csv header
        with open(args.save, 'w') as file:
            file.write(f"timestamp, hostname, api, device, index, dtype, vector_dim, num_vectors, ")
            file.write(f"index_batch, index_time, index_rate, index_bandwidth, ")
            file.write(f"search_batch, search_time, search_rate, search_bandwidth, memory\n")
    with open(args.save, 'a') as file:
        file.write(f"{datetime.datetime.now().strftime('%Y%m%d %H:%M:%S')}, {socket.gethostname()}, faiss, ")
        file.write(f"{'cpu' if args.cpu else 'cuda'}, {args.index}, {args.dtype}, {args.dim}, {args.index_size}, ")
        file.write(f"{args.index_batch}, {avg_index_time}, {avg_index_rate}, {avg_index_rate*vector_size}, ")
        file.write(f"{args.search_batch}, {avg_search_time}, {avg_search_rate}, {avg_search_rate*vector_size}, {memory_usage}\n")
        