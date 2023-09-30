#!/usr/bin/env python3
import os
import sys
import time
import tqdm
import torch

import numpy as np
import ctypes as C

from faiss_lite import (
    cudaKNN, 
    cudaL2Norm, 
    cudaAllocMapped, 
    DistanceMetrics, 
    assert_cuda
)

from cuda.cudart import (
    cudaStreamCreateWithFlags, 
    cudaStreamNonBlocking, 
    cudaStreamSynchronize,
)

from .utils import AttrDict, torch_dtype, tqdm_redirect_stdout


class cudaVectorIndex:
    """
    Flat vector store that uses FAISS CUDA kernels for KNN search
    
      -- uses zero-copy mapped CUDA memory
      -- supports np.float16 and np.float32 vectors
      -- returned distances are float32, indexes are int64
      
    It's aimed at storing high-dimensional embeddings,
    with a fixed reserve allocation due to their size.
    """
    def __init__(self, dim, dtype=np.float32, reserve=1<<30, metric='cosine', max_search_queries=1, **kwargs):
        """
        Allocate memory for a vector index.
        
        Parameters:
        
          dim (int) -- dimension of the vectors (i.e. the size of the embedding)
          dtype (np.dtype) -- data type of the vectors, either float32 or float16
          reserve (int) -- maximum amount of memory (in bytes) to allocate for storing vectors (default 1GB)
          metric (str) -- the distance metric to use (recommend 'l2', 'inner_product', or 'cosine')
          max_search_queries (int) -- the maximum number of queries to search for at a time
        """
        self.shape = (0, dim)
        self.dtype = dtype
        self.dsize = np.dtype(dtype).itemsize
        self.metric = metric
        self.stats = AttrDict()
        
        self.reserved_size = reserve
        self.reserved = int(self.reserved_size / (dim * self.dsize))
        self.max_search_queries = max_search_queries

        err, self.stream = cudaStreamCreateWithFlags(cudaStreamNonBlocking)
        assert_cuda(err)
        self.torch_stream = torch.cuda.ExternalStream(int(self.stream), device='cuda:0')
        #torch.cuda.set_stream(self.torch_stream)
        
        print(f"-- creating CUDA stream {self.stream}")
        print(f"-- creating CUDA vector index ({self.reserved},{dim}) dtype={dtype} metric={metric}")
        
        self.vectors = cudaAllocMapped((self.reserved, dim), dtype) # inputs
        self.queries = cudaAllocMapped((max_search_queries, dim), dtype)

        self.indexes = cudaAllocMapped((max_search_queries, self.reserved), np.int64) # outputs
        self.distances = cudaAllocMapped((max_search_queries, self.reserved), np.float32)
        
        if metric == 'l2':
            self.vector_norms = cudaAllocMapped(self.reserved, np.float32)
     
    def __len__(self):
        return self.shape[0]
     
    def size(self):
        return self.shape[0] * self.shape[1] * self.dsize
     
    def sync(self):
        assert_cuda(cudaStreamSynchronize(self.stream))
        
    def add(self, vector, sync=True):
        """
        Add a vector and return its index.
        If the L2 metric is being used, its L2 norm will be computed.
        If sync=True, the CPU will wait for the CUDA stream to complete.
        """
        index = self.shape[0]
        
        if isinstance(vector, np.ndarray):
            np.copyto(dst=self.vectors.array[index], src=vector, casting='no')
        elif isinstance(vector, torch.Tensor):
            with torch.cuda.StreamContext(self.torch_stream):
                self.vectors.tensor[index] = vector
        else:
            raise ValueError(f"vector needs to be a np.ndarray or torch.Tensor (was type {type(vector)})")
            
        if self.metric == 'l2':
            assert(cudaL2Norm(
                C.cast(self.vectors.ptr + index * self.shape[1] * self.dsize, C.c_void_p),
                self.dsize, 1, self.shape[1],
                C.cast(self.vector_norms.ptr + index * 4, C.POINTER(C.c_float)),
                True, C.c_void_p(int(self.stream)) if self.stream else None
            ))
            
            if sync:
                self.sync()

        elif self.metric == 'cosine':
            with torch.cuda.StreamContext(self.torch_stream), torch.inference_mode():
                torch.nn.functional.normalize(
                    self.vectors.tensor[index], dim=-1,
                    out=self.vectors.tensor[index]
                )
            
            if sync:
                self.sync()
                
        self.shape = (index + 1, self.shape[1])
        return index
        
    def search(self, queries, k=1, sync=True, return_tensors='np'):
        """
        Returns the indexes and distances of the k-closest vectors using the given distance metric.
        Each query should be of the same dimension and dtype that this class was created with.
        
        Returns (indexes, distances) tuple where each of these has shape (queries, K)
        The returned distances are always with float32 dtype and the indexes are int64
        If return_tensors is 'np' ndarray's will be returned, or 'pt' for PyTorch tensors.
        
        Note that for 'inner_product' and 'cosine' metrics, these are similarity metrics not
        distances, so they are in descending order (not ascending).  'cosine' is normalized
        between [0,1], where 1.0 is the highest probability of a match.
        
        If sync=True, the CPU will wait for the CUDA stream to complete.  Otherwise, the function
        is aynchronous and None is returned.  In this case, cudaStreamSynchronize() should be called.
        """
        if isinstance(queries, list):
            queries = np.asarray(queries)
        
        if len(queries.shape) == 1:
            queries.shape = (1, queries.shape[0])
            
        if queries.shape[0] > self.max_search_queries:
            raise ValueError(f"the number of queries exceeds the max_search_queries of {self.max_search_queries}")

        if queries.shape[1] != self.shape[1]:
            raise ValueError(f"queries must match the vector dimension ({self.shape[1]})")
        
        if queries.dtype != self.dtype and queries.dtype != torch_dtype(self.dtype):
            raise ValueError(f"queries need to use {self.dtype} dtype (was type {queries.dtype})")
            
        if isinstance(queries, np.ndarray):
            np.copyto(dst=self.queries.array[:queries.shape[0]], src=queries, casting='no')
        elif isinstance(queries, torch.Tensor):
            with torch.cuda.StreamContext(self.torch_stream):
                self.queries.tensor[:queries.shape[0]] = queries 
           
        time_begin = time.perf_counter()
        
        if self.metric == 'cosine':
            with torch.cuda.StreamContext(self.torch_stream), torch.inference_mode():
                torch.nn.functional.normalize(
                    self.queries.tensor[:queries.shape[0]], dim=-1,
                    out=self.queries.tensor[:queries.shape[0]]
                )
            
        metric=self.metric if self.metric != 'cosine' else 'inner_product'
        
        assert(cudaKNN(
            C.cast(self.vectors.ptr, C.c_void_p),
            C.cast(self.queries.ptr, C.c_void_p),
            self.dsize,
            self.shape[0],
            queries.shape[0],
            self.shape[1],
            k,
            DistanceMetrics[metric],
            C.cast(self.vector_norms.ptr, C.POINTER(C.c_float)) if self.metric == 'l2' else None,
            C.cast(self.distances.ptr, C.POINTER(C.c_float)),
            C.cast(self.indexes.ptr, C.POINTER(C.c_longlong)),
            C.cast(int(self.stream), C.c_void_p) if self.stream else None
        ))

        if sync:
            self.sync()
            
            self.stats.query_shape = queries.shape
            self.stats.index_shape = self.shape
            self.stats.search_time = time.perf_counter() - time_begin
            
            if return_tensors == 'np':
                return (
                    np.copy(self.indexes.array[:queries.shape[0], :k]).squeeze(),
                    np.copy(self.distances.array[:queries.shape[0], :k]).squeeze()
                )
        else:
            self.stats.search_time = time.perf_counter() - time_begin
            return None
            
    def validate(self, k=4):
        """
        Sanity check search
        """
        correct=True
        metric=self.metric if self.metric != 'cosine' else 'inner_product'

        for n in tqdm.tqdm(range(self.shape[0]), file=sys.stdout):
            with tqdm_redirect_stdout():
                assert(cudaKNN(
                    C.cast(self.vectors.ptr, C.c_void_p),
                    C.cast(self.vectors.ptr+n*self.shape[1]*self.dsize, C.c_void_p),
                    self.dsize,
                    self.shape[0],
                    1,
                    self.shape[1],
                    k,
                    DistanceMetrics[metric],
                    C.cast(self.vector_norms.ptr, C.POINTER(C.c_float)) if self.metric == 'l2' else None,
                    C.cast(self.distances.ptr, C.POINTER(C.c_float)),
                    C.cast(self.indexes.ptr, C.POINTER(C.c_longlong)),
                    C.cast(int(self.stream), C.c_void_p) if self.stream else None
                ))
                self.sync()
                if self.indexes.array[0][0] != n:
                    print(f"incorrect or duplicate index [{n}]  indexes={self.indexes.array[0,:k]}  distances={self.distances.array[0,:k]}")
                    #assert(self.indexes[0][0]==n)
                    correct=False
                
        return correct
        
        
'''
# https://ai.stackexchange.com/questions/36191/how-to-calculate-a-meaningful-distance-between-multidimensional-tensors
# https://ai.stackexchange.com/questions/37233/similarities-between-2d-vectors-to-flatten-or-to-not
def cumsum_3d(a):
    a = torch.cumsum(a, -1)
    a = torch.cumsum(a, -2)
    a = torch.cumsum(a, -3)
    return a

def norm_3d(a):
    return a / torch.sum(a, dim=(-1,-2,-3), keepdim=True)

def emd_3d(a, b):
    a = norm_3d(a)
    b = norm_3d(b)
    return torch.mean(torch.square(cumsum_3d(a) - cumsum_3d(b)), dim=(-1,-2,-3))

def cumsum_2d(a):
    a = torch.cumsum(a, -1)
    a = torch.cumsum(a, -2)
    return a

def norm_2d(a):
    return a / torch.sum(a, dim=(-1,-2), keepdim=True)

def emd_2d(a, b):
    a = norm_2d(a)
    b = norm_2d(b)
    return torch.mean(torch.square(cumsum_2d(a) - cumsum_2d(b)), dim=(-1,-2))
    
cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

et = torch.cat(embeddings)
print('et', et.shape, et.dtype, et.device)

for n, embedding in enumerate(embeddings):
    print(n, embedding.shape, embedding.dtype, type(embedding))
    
    distances = cos(embedding, et)
    """
    distances = torch.zeros((len(embeddings)), dtype=torch.float32, device='cuda:0')
    
    print(f"distances:  {distances.shape}  {distances.dtype}")
    
    for m, embedding2 in enumerate(embeddings):
        distances[m] = emd_2d(embedding, embedding2)
        
    print(distances)
    """
    
    indexes = torch.argsort(distances, descending=True).squeeze()
    
    print(f"-- search results for {n} {metadata[n]}")
    for k in range(args.k):
        print(f"   * {indexes[k]} {metadata[indexes[k]]}  dist={distances[indexes[k]]}")
'''


"""
# pylibraft - https://github.com/rapidsai/raft/blob/4f0a2d2d6e30eea0c036ca3b531e03e44e760fbe/python/pylibraft/pylibraft/distance/pairwise_distance.pyx#L93
time_dist_begin = time.perf_counter()
pairwise_distance(
    vector, 
    cudaArrayInterface(self.data, self.shape, self.dtype),
    cudaArrayInterface(self.search_data, (vector.shape[0], self.shape[0]), self.dtype),
    metric=metric
)

cudaDeviceSynchronize()
time_dist_end = time.perf_counter()

if k == 1:
    topk = np.argmin(self.search_array, axis=1)
    
time_sort_end = time.perf_counter()

print(f'dist: {(time_dist_end - time_dist_begin) * 1000:.3f} ms')
print(f'sort: {(time_sort_end - time_dist_end) * 1000:.3f} ms')
"""