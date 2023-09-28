/**
 * expose faiss cudaKNN and cudaL2Norm functions
 */
#include <stdio.h>
#include <unistd.h>

#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/impl/Distance.cuh>
#include <faiss/gpu/impl/L2Norm.cuh>

faiss::gpu::StandardGpuResourcesImpl* resources;


/**
 * vector - CUDA device pointer to NxD array
 * queries - CUDA device pointer to MxD array
 * vector_norms - CUDA device pointer to cached L2 norms of the vectors for METRIC_L2 (optional)
 * out_distances - CUDA device pointer to MxK float array
 * out_indices - CUDA device pointer to MxK int64 array
 */
template<typename T>
bool _cudaKNN(
	T* vectors, T* queries,
	int n, int m, int d, int k, 
	faiss::MetricType metric,
	float* vector_norms,
	float* out_distances,
	int64_t* out_indices,
	cudaStream_t stream=0)
{
	if( !resources )
		resources = new faiss::gpu::StandardGpuResourcesImpl(); // todo fix memory leak
	
	faiss::gpu::DeviceTensor<T, 2, true> vector_t(vectors, {n, d}); 
	faiss::gpu::DeviceTensor<T, 2, true> query_t(queries, {m, d}); 
	
	faiss::gpu::DeviceTensor<float, 1, true> vector_norms_t(vector_norms, {n}); 
	faiss::gpu::DeviceTensor<float, 2, true> out_distances_t(out_distances, {m, k});
	faiss::gpu::DeviceTensor<int64_t, 2, true> out_indices_t(out_indices, {m, k});

	// https://github.com/facebookresearch/faiss/blob/main/faiss/gpu/impl/Distance.cuh
	bfKnnOnDevice(
		resources,
		0,  		  // device
		stream,	  // cudaStream_t
		vector_t,
		true, // row major
		(vector_norms != NULL) ? &vector_norms_t : NULL,
		query_t,
		true, // row major
		k,
		metric,
		2.0f,	  // metric arg (La)
		out_distances_t,
		out_indices_t,
		false
	);

	return true;
}

extern "C" 
bool cudaKNN(
	void* vectors, void* queries, int dsize,
	int n, int m, int d, int k, 
	faiss::MetricType metric,
	float* vector_norms,
	float* out_distances,
	int64_t* out_indices,
	cudaStream_t stream=0 )
{
	//printf("cudaKNN(vectors=%p, queries=%p, dsize=%i, n=%i, m=%i, d=%i, k=%i, metric=%i, vector_norms=%p, out_distances=%p, out_indices=%p, stream=%p);\n", vectors, queries, dsize, n, m, d, k, metric, vector_norms, out_distances, out_indices, stream);
	
	if( dsize == sizeof(float) )
		return _cudaKNN<float>((float*)vectors, (float*)queries, n, m, d, k, metric, vector_norms, out_distances, out_indices, stream);
	else if( dsize == sizeof(half) )
		return _cudaKNN<half>((half*)vectors, (half*)queries, n, m, d, k, metric, vector_norms, out_distances, out_indices, stream);
	
	printf("cudaKNN() -- invalid datatype size (%i)\n", dsize);
	return false;
}


/**
 * vector - CUDA device pointer to NxD array
 * output - CUDA device pointer to Nx1 array
 */
template<typename T>
bool _cudaL2Norm(
	T* vectors, 
	int n, int d,
	float* output,
	bool squared=true,
	cudaStream_t stream=0 )
{
	faiss::gpu::DeviceTensor<T, 2, true> vector_t(vectors, {n, d}); 
	faiss::gpu::DeviceTensor<float, 1, true> output_t(output, {n}); 

	runL2Norm(vector_t, true, output_t, squared, stream);
	
	return true;
}


extern "C" 
bool cudaL2Norm(
	void* vectors, int dsize,
	int n, int d,
	float* output, 
	bool squared=true, 
	cudaStream_t stream=0 ) 
{
	//printf("cudaL2Norm(vectors=%p, dsize=%i, n=%i, d=%i, output=%p, squared=%s, stream=%p);\n", vectors, dsize, n, d, output, squared ? "true" : "false", stream);
	
	if( dsize == sizeof(float) )
		return _cudaL2Norm<float>((float*)vectors, n, d, output, squared, stream);
	else if( dsize == sizeof(half) )
		return _cudaL2Norm<half>((half*)vectors, n, d, output, squared, stream);
	
	printf("cudaKNN() -- invalid datatype size (%i)\n", dsize);
	return false;
}	
