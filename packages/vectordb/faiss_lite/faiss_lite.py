#!/usr/bin/env python3
import os
import math
import ctypes as C

_lib = C.CDLL('/opt/faiss_lite/build/libfaiss_lite.so')

# https://github.com/facebookresearch/faiss/blob/main/faiss/MetricType.h
METRICS = {
    'inner_product': 0,
    'l2': 1,
    'l1': 2,
    'Linf': 3,
    'canberra': 20,
    'braycurtis': 21,
    'jensenshannon': 22,
    'jaccard': 23,
}


def _cudaKNN(name='cudaKNN'):
    func = _lib[name]
    
    func.argtypes = [
        C.c_void_p, # vectors
        C.c_void_p, # queries
        C.c_int,    # dsize
        C.c_int,    # n
        C.c_int,    # m
        C.c_int,    # d
        C.c_int,    # k
        C.c_int,    # metric
        C.POINTER(C.c_float), # vector_norms
        C.POINTER(C.c_float), # out_distances
        C.POINTER(C.c_longlong), # out_indices
        C.c_void_p, # cudaStream_t
    ]
    
    func.restype = C.c_bool
    return func
 
def _cudaL2Norm(name='cudaL2Norm'):
    func = _lib[name]
    
    func.argtypes = [
        C.c_void_p, # vectors
        C.c_int,    # dsize
        C.c_int,    # n
        C.c_int,    # d
        C.POINTER(C.c_float), # output
        C.c_bool,   # squared
        C.c_void_p, # cudaStream_t
    ]
    
    func.restype = C.c_bool
    return func
    
cudaKNN = _cudaKNN()
cudaL2Norm = _cudaL2Norm()


def dtype_to_ctype(dtype):
    if dtype == np.float32:
        return C.c_float
    elif dtype == np.float16:
        return C.c_ushort
    elif dtype == np.int32:
        return C.c_int
    elif dtype == np.int64:
        return C.c_longlong
    else:
        raise RuntimeError(f"unsupported dtype:  {dtype}")
        
def cudaAlloc(shape, dtype):
    """
    Allocate cudaMallocManaged() memory and map it to a numpy array
    """
    ctype = dtype_to_ctype(dtype)
    dsize = np.dtype(dtype).itemsize

    if isinstance(shape, int):
        size = shape * dsize
        shape = [shape]
    else:
        size = math.prod(shape) * dsize
    
    print(f"-- allocating {size} bytes ({size/(1024*1024):.2f} MB) with cudaMallocManaged()")
    
    _, ptr = cudaMallocManaged(size, cudaMemAttachGlobal)
    array = np.ctypeslib.as_array(C.cast(ptr, C.POINTER(ctype)), shape=shape)
    
    if dtype == np.float16:
        array.dtype = np.float16

    return array, ptr
        

if __name__ == '__main__':
    import time
    import argparse
    import numpy as np
    
    from cuda.cudart import cudaGetDeviceProperties, cudaMallocManaged, cudaMemAttachGlobal, cudaDeviceSynchronize
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-d', '--dim', type=int, default=5120, help='the dimensionality of the embedding vectors') 
    parser.add_argument('-n', '--num-vectors', type=int, default=64, help='the number of vectors to add to the index')
    parser.add_argument('-k', type=int, default=4, help='the number of nearest-neighbors to find')
    parser.add_argument('--dtype', type=str, default='float32', choices=['float32', 'float16'], help='datatype of the vectors')
    parser.add_argument('--metric', type=str, default='l2', help='the distance metric to use during search')
    parser.add_argument('--num-queries', type=int, default=1) 
    
    parser.add_argument('--seed', type=int, default=1234, help='change the random seed used')
    parser.add_argument('--runs', type=int, default=25)
    
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

    vectors, vector_ptr = cudaAlloc((args.num_vectors, args.dim), dtype)
    queries, query_ptr = cudaAlloc((args.num_queries, args.dim), dtype)
  
    distances, distance_ptr = cudaAlloc((args.num_queries, args.k), np.float32)
    indices, index_ptr = cudaAlloc((args.num_queries, args.k), np.int64)

    for n in range(args.num_vectors):
        vectors[n] = xb[n]
     
    for n in range(args.num_queries):
        queries[n] = xq[n]
       
    print('vectors', vectors, vectors.shape, vectors.dtype)
    print('queries', queries, queries.shape, queries.dtype)
    
    vector_norms_ptr = None
    
    if args.metric == 'l2':
        vector_norms, vector_norms_ptr = cudaAlloc(args.num_vectors, np.float32)

        result = cudaL2Norm(
            C.cast(vector_ptr, C.c_void_p),
            dsize, args.num_vectors, args.dim,
            C.cast(vector_norms_ptr, C.POINTER(C.c_float)),
            True, None
        )
        
        cudaDeviceSynchronize()
        print('vector_norms', vector_norms, vector_norms.shape, vector_norms.dtype)

        # check the time for single vectors
        avg_l2_time = 0
        
        for n in range(args.num_vectors):
            time_begin = time.perf_counter()
            cudaL2Norm(
                C.cast(vector_ptr+n*args.dim*dsize, C.c_void_p),
                dsize, 1, args.dim,
                C.cast(vector_norms_ptr+n*4, C.POINTER(C.c_float)),
                True, None
            )
            cudaDeviceSynchronize()
            time_elapsed = (time.perf_counter() - time_begin) * 1000
            print(f"l2_norm time for (1,{args.dim}) vector:  {time_elapsed:.3f} ms")
            if n > 0:
                avg_l2_time += time_elapsed / (args.num_vectors-1)
            
    ctype = C.POINTER(dtype_to_ctype(dtype))
      
    # sanity check
    for n in range(args.num_vectors):
        result = cudaKNN(
            C.cast(vector_ptr, C.c_void_p),
            C.cast(vector_ptr+n*args.dim*dsize, C.c_void_p),
            dsize,
            args.num_vectors,
            1,
            args.dim,
            args.k,
            METRICS[args.metric],
            C.cast(vector_norms_ptr, C.POINTER(C.c_float)),
            C.cast(distance_ptr, C.POINTER(C.c_float)),
            C.cast(index_ptr, C.POINTER(C.c_longlong)),
            None
        )
        cudaDeviceSynchronize()
        #print(f"n={n}  {indices[0]}  dist={distances[0]}")
        assert(result)
        assert(indices[0][0]==n)
    
    avg_time = 0
    
    for r in range(args.runs):
        time_begin = time.perf_counter()

        result = cudaKNN(
            C.cast(vector_ptr, C.c_void_p),
            C.cast(query_ptr, C.c_void_p),
            dsize,
            args.num_vectors,
            args.num_queries,
            args.dim,
            args.k,
            METRICS[args.metric],
            C.cast(vector_norms_ptr, C.POINTER(C.c_float)),
            C.cast(distance_ptr, C.POINTER(C.c_float)),
            C.cast(index_ptr, C.POINTER(C.c_longlong)),
            None
        )
        time_enqueue = (time.perf_counter() - time_begin) * 1000
        assert(result)
        
        cudaDeviceSynchronize()
        time_elapsed = (time.perf_counter() - time_begin) * 1000
        
        print(f"cudaKNN  enqueue:  {time_enqueue:.3f} ms   process:  {time_elapsed:.3f}")
        
        if r > 0:
            avg_time += time_elapsed / (args.runs-1)
    
    print("")
    print(f"N={args.num_vectors} M={args.num_queries} D={args.dim} K={args.k} metric={args.metric} dtype={args.dtype}\n")
    print(f"average search time:   {avg_time:.3f} ms")
    
    if args.metric == 'l2':
        print(f"average l2 norm time:  {avg_l2_time:.3f} ms")
        
    
