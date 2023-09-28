#!/usr/bin/env python3
import os
import time

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
    cudaStreamSynchronize
)


class cudaVectorIndex:
    """
    Flat vector store that uses FAISS CUDA kernels for KNN search
    
      -- uses zero-copy mapped CUDA memory
      -- supports np.float16 and np.float32 vectors
      -- returned distances are float32, indexes are int64
      
    It's aimed at storing high-dimensional embeddings,
    with a fixed reserve allocation due to their size.
    """
    def __init__(self, dim, reserve=1024, max_search_queries=1, dtype=np.float32, metric='l2'):
        """
        Allocate memory for a vector index.
        
        Parameters:
        
          dim (int) -- dimension of the vectors (i.e. the size of the embedding)
          reserve (int) -- maximum amount of memory (in MB) to allocate for storing vectors
          max_search_queries (int) -- the maximum number of queries to search for at a time
          dtype (np.dtype) -- data type of the vectors, either float32 or float16
          metric (str) -- the distance metric to use (recommend 'l2' or 'inner_product')
        """
        self.shape = (0, dim)
        self.dtype = dtype
        self.dsize = np.dtype(dtype).itemsize
        
        self.metric = metric
        
        self.reserved_size = reserve * 1024 * 1024
        self.reserved = int(self.reserved_size / (dim * self.dsize))
        
        self.max_search_queries = max_search_queries
        
        self.stream = None
        err, self.stream = cudaStreamCreateWithFlags(cudaStreamNonBlocking)
        assert_cuda(err)
        
        print(f"-- creating CUDA stream {self.stream}")
        print(f"-- creating CUDA vector index ({reserve},{dim})")
        
        self.vectors, self.vector_ptr = cudaAllocMapped((self.reserved, dim), dtype) # inputs
        self.queries, self.query_ptr = cudaAllocMapped((max_search_queries, dim), dtype)

        self.indexes, self.index_ptr = cudaAllocMapped((max_search_queries, self.reserved), np.int64) # outputs
        self.distances, self.distance_ptr = cudaAllocMapped((max_search_queries, self.reserved), np.float32)
        
        if metric == 'l2':
            self.vector_norms, self.vector_norms_ptr = cudaAllocMapped(self.reserved, np.float32)
            
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
            self.vectors.tensor[index] = vector
        else:
            raise ValueError(f"vector needs to be a np.ndarray or torch.Tensor (was type {type(vector)})")
            
        if self.metric == 'l2':
            assert(cudaL2Norm(
                C.cast(self.vector.ptr + index * self.shape[1] * self.dsize, C.c_void_p),
                self.dsize, 1, self.shape[1],
                C.cast(self.vector_norms.ptr + index * 4, C.POINTER(C.c_float)),
                True, C.c_void_p(self.stream.getPtr()) if self.stream else None
            ))
            
            if sync:
                assert_cuda(cudaStreamSynchronize(self.stream))
                
            #print(f'vector_norms[{index}]', self.vector_norms.shape, self.vector_norms.dtype)
        
        self.shape = (index + 1, self.shape[1])
        return index
        
    def search(self, queries, k=1, sync=True, return_tensors='np'):
        """
        Returns the indexes and distances of the k-closest vectors using the given distance metric.
        Each query should be of the same dimension and dtype that this class was created with.
        
        Returns (indexes, distances) tuple where each of these has shape (queries, K)
        The returned distances are always with float32 dtype and the indexes are int64
        If return_tensors is 'np' ndarray's will be returned, or 'pt' for PyTorch tensors.
        
        If sync=True, the CPU will wait for the CUDA stream to complete.  Otherwise, the function
        is aynchronous and None is returned.  In this case, cudaStreamSynchronize() should be called.
        """
        if isinstance(queries, list):
            queries = np.asarray(queries)
        
        if len(queries.shape) == 1:
            queries.shape = (1, queries.shape[0])
            
        if queries.shape[0] > self.max_search_queries:
            raise ValueError(f"the number of queries exceeds the max_search_queries of {self.max_search_queries}")
            
        #print('queries', queries.shape)
        
        if queries.shape[1] != self.shape[1]:
            raise ValueError(f"queries must match the vector dimension ({self.shape[1]})")
        
        if queries.dtype != self.dtype:
            raise ValueError(f"queries need to use {self.dtype} dtype")
            
        if isinstance(queries, np.ndarray):
            np.copyto(dst=self.queries.array[:queries.shape[0]], src=queries, casting='no')
        elif isinstance(queries, torch.Tensor):
            self.queries.tensor[:queries.shape[0]] = queries 
            
        assert(cudaKNN(
            C.cast(self.vectors.ptr, C.c_void_p),
            C.cast(self.queries.ptr, C.c_void_p),
            self.dsize,
            self.shape[0],
            queries.shape[0],
            self.shape[1],
            k,
            DistanceMetrics[self.metric],
            C.cast(self.vector_norms.ptr, C.POINTER(C.c_float)) if self.metric == 'l2' else None,
            C.cast(self.distances.ptr, C.POINTER(C.c_float)),
            C.cast(self.indexes.ptr, C.POINTER(C.c_longlong)),
            C.cast(self.stream.getPtr(), C.c_void_p) if self.stream else None
        ))

        if sync:
            assert_cuda(cudaStreamSynchronize(self.stream))
            
            if return_tensors == 'np':
                return (
                    np.copy(self.indexes[:queries.shape[0], :k]).squeeze(),
                    np.copy(self.distances[:queries.shape[0], :k]).squeeze()
                )
            
        else:
            return None
            
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

    def validate(self, k=4):
        """
        Sanity check search
        """
        correct=True
        
        for n in range(self.shape[0]):
            assert(cudaKNN(
                C.cast(self.vector_ptr, C.c_void_p),
                C.cast(self.vector_ptr+n*self.shape[1]*self.dsize, C.c_void_p),
                self.dsize,
                self.shape[0],
                1,
                self.shape[1],
                k,
                DistanceMetrics[self.metric],
                C.cast(self.vector_norms_ptr, C.POINTER(C.c_float)) if self.metric == 'l2' else None,
                C.cast(self.distance_ptr, C.POINTER(C.c_float)),
                C.cast(self.index_ptr, C.POINTER(C.c_longlong)),
                C.cast(self.stream.getPtr(), C.c_void_p) if self.stream else None
            ))
            assert_cuda(cudaStreamSynchronize(self.stream))
            if self.indexes[0][0] != n:
                print(f"incorrect or duplicate index[{n}]  indexes={self.indexes[0,:k]}  distances={self.distances[0,:k]}")
                #assert(self.indexes[0][0]==n)
                correct=False
                
        return correct