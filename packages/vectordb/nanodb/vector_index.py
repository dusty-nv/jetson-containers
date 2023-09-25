#!/usr/bin/env python3
import os
import time

import numpy as np
import ctypes as C

from cuda.cudart import cudaDeviceSynchronize

from faiss_lite import cudaKNN, cudaL2Norm, cudaAllocMapped, DistanceMetrics


class cudaVectorIndex:
    """
    Flat vector store that uses FAISS CUDA kernels for KNN search
    
      -- uses zero-copy mapped CUDA memory
      -- supports np.float16 and np.float32 vectors
      -- returned distances are float32, indexes are int64
      
    It's aimed at storing high-dimensional embeddings,
    with a fixed reserve allocation due to their size.
    """
    def __init__(self, dim, reserve=1024, max_batch_size=1, dtype=np.float32, metric='l2'):
        """
        Allocate memory for a vector index.
        
        Parameters:
        
          dim (int) -- dimension of the vectors (i.e. the size of the embedding)
          reserve (int) -- maximum number of vectors able to be stored
          max_batch_size (int) -- maximum number of search() queries
          dtype (np.dtype) -- data type of the vectors, either float32 or float16
          metric (str) -- the distance metric to use (recommend 'l2' or 'inner_product')
        """
        self.shape = (0, dim)
        self.dtype = dtype
        self.dsize = np.dtype(dtype).itemsize
        
        self.metric = metric
        self.reserved = reserve
        self.reserved_size = dim * reserve * dtype.itemsize
        self.max_batch_size = max_batch_size
        
        print(f"-- creating CUDA vector index ({reserve},{dim})")
        
        self.vectors, self.vector_ptr = cudaAllocMapped((reserve, dim), dtype) # inputs
        self.queries, self.query_ptr = cudaAllocMapped((max_batch_size, dim), dtype)

        self.indexes, self.index_ptr = cudaAllocMapped((max_batch_size, reserve), np.int64) # outputs
        self.distances, self.distance_ptr = cudaAllocMapped((max_batch_size, reserve), np.float32)
        
        if metric == 'l2':
            self.vector_norms, self.vector_norms_ptr = cudaAllocMapped(reserve, np.float32)
            
    def add(self, vector, sync=True):
        """
        Add a vector and return its index.
        If the L2 metric is being used, its L2 norm will be computed.
        If sync=True, the CPU will wait for the CUDA stream to complete.
        """
        index = self.shape[0]
        np.copyto(dst=self.vectors[index], src=vector, casting='no')
        
        if self.metric == 'l2':
            assert(cudaL2Norm(
                C.cast(self.vector_ptr + index * self.shape[1] * self.dsize, C.c_void_p),
                self.dsize, 1, self.shape[1],
                C.cast(self.vector_norms_ptr + index * 4, C.POINTER(C.c_float)),
                True, None
            ))
            if sync:
                cudaDeviceSynchronize()
            #print(f'vector_norms[{index}]', self.vector_norms.shape, self.vector_norms.dtype)
        
        self.shape = (index + 1, self.shape[1])
        return index
        
    def search(self, queries, k=1, sync=True):
        """
        Returns the indexes and distances of the k-closest vectors using the given distance metric.
        Each query should be of the same dimension and dtype that this class was created with.
        Returns (indexes, distances) tuple where each of these has shape (queries, K)
        The returned distances are always with np.float32 dtype and the indexes are np.int64
        If sync=True, the CPU will wait for the CUDA stream to complete.
        """
        queries = np.asarray(queries)
        
        if len(queries.shape) == 1:
            queries.shape = (1, queries.shape[0])
            
        if queries.shape[0] > self.max_batch_size:
            raise ValueError(f"the number of queries exceeds the max_batch_size of {self.max_batch_size}")
            
        #print('queries', queries.shape)
        
        if queries.shape[1] != self.shape[1]:
            raise ValueError(f"queries must match the vector dimension ({self.shape[1]})")
        
        if queries.dtype != self.dtype:
            raise ValueError(f"queries need to use {self.dtype} dtype")
            
        np.copyto(dst=self.queries[:queries.shape[0]], src=queries, casting='no')
        
        assert(cudaKNN(
            C.cast(self.vector_ptr, C.c_void_p),
            C.cast(self.query_ptr, C.c_void_p),
            self.dsize,
            self.shape[0],
            queries.shape[0],
            self.shape[1],
            k,
            DistanceMetrics[self.metric],
            C.cast(self.vector_norms_ptr, C.POINTER(C.c_float)) if self.metric == 'l2' else None,
            C.cast(self.distance_ptr, C.POINTER(C.c_float)),
            C.cast(self.index_ptr, C.POINTER(C.c_longlong)),
            None
        ))

        if sync:
            cudaDeviceSynchronize()
        
        #print('indexes', self.indexes.shape, (queries.shape[0], k))
        
        return (
            np.copy(self.indexes[:queries.shape[0], :k]).squeeze(),
            np.copy(self.distances[:queries.shape[0], :k]).squeeze()
        )
        
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

    def validate(self):
        """
        Sanity check search
        """
        for n in range(self.shape[0]):
            assert(cudaKNN(
                C.cast(self.vector_ptr, C.c_void_p),
                C.cast(self.vector_ptr+n*self.shape[1]*self.dsize, C.c_void_p),
                self.dsize,
                self.shape[0],
                1,
                self.shape[1],
                1,
                DistanceMetrics[self.metric],
                C.cast(self.vector_norms_ptr, C.POINTER(C.c_float)) if self.metric == 'l2' else None,
                C.cast(self.distance_ptr, C.POINTER(C.c_float)),
                C.cast(self.index_ptr, C.POINTER(C.c_longlong)),
                None
            ))
            cudaDeviceSynchronize()
            if self.indexes[0][0] != n:
                print(f"incorrect index[{n}]\n", self.indexes, "\n", self.distances)
                #assert(self.indexes[0][0]==n)
                return false
                
        return True