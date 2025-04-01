/**
 * expose piecewise distance functions
 */
#include <time.h>
#include <stdio.h>
#include <unistd.h>

#include <cuda.h>
#include <cuda_fp16.h>

#include <faiss/MetricType.h>


extern "C" 
bool cudaKNN(
	void* vectors, void* queries, int dsize,
	int n, int m, int d, int k, 
	faiss::MetricType metric,
	float* vector_norms,
	float* out_distances,
	int64_t* out_indices,
	cudaStream_t stream=NULL);

extern "C" 
bool cudaL2Norm(
	void* vectors, int dsize,
	int n, int d,
	float* output, 
	bool squared=true, 
	cudaStream_t stream=NULL );
	
	
#define CUDA(x)				cudaCheckError((x), #x, __FILE__, __LINE__)
#define CUDA_SUCCESS(x)			(CUDA(x) == cudaSuccess)
#define CUDA_FAILED(x)			(CUDA(x) != cudaSuccess)
#define CUDA_VERIFY(x)			if(CUDA_FAILED(x))	return false;

//#define CUDA_TRACE
#define LOG_CUDA "[cuda]   "

inline cudaError_t cudaCheckError(cudaError_t retval, const char* txt, const char* file, int line )
{
#if !defined(CUDA_TRACE)
	if( retval == cudaSuccess)
		return cudaSuccess;
#endif

	if( retval == cudaSuccess )
	{
		printf(LOG_CUDA "%s\n", txt);
	}
	else
	{
		printf(LOG_CUDA "%s\n", txt);
	}
	
	if( retval != cudaSuccess )
	{
		printf(LOG_CUDA "   %s (error %u) (hex 0x%02X)\n", cudaGetErrorString(retval), retval, retval);
		printf(LOG_CUDA "   %s:%i\n", file, line);	
	}

	return retval;
}

inline double time_diff( const timespec& start, const timespec& end )
{
	timespec result;
	
	if ((end.tv_nsec-start.tv_nsec)<0) {
		result.tv_sec = end.tv_sec-start.tv_sec-1;
		result.tv_nsec = 1000000000+end.tv_nsec-start.tv_nsec;
	} else {
		result.tv_sec = end.tv_sec-start.tv_sec;
		result.tv_nsec = end.tv_nsec-start.tv_nsec;
	}
	
	return result.tv_sec * 1000.0 + result.tv_nsec * 0.000001;  // return milliseconds
}

template<typename T>
T* cudaAlloc(int elements) 
{
	const size_t size = elements * sizeof(T);
	T* mem = NULL;
	
	printf(LOG_CUDA "allocating %zu bytes  (%i elements, %zu bytes each, %.2f MB)\n", size, elements, sizeof(T), float(size)/(1024.0f*1024.0f));
	
	if( CUDA_FAILED(cudaMallocManaged(&mem, size, cudaMemAttachGlobal)) )
		return NULL;
		
	return mem;
}

template<typename T>
double test(int N, int M, int D, int K, faiss::MetricType metric=faiss::METRIC_L2, int runs=10, cudaStream_t stream=0)
{
	T* vectors = cudaAlloc<T>(N * D);
	T* queries = cudaAlloc<T>(M * D);
	
	float* distances = cudaAlloc<float>(M * K);
	int64_t* indices = cudaAlloc<int64_t>(M * K);
	
	float* vectorNorms = NULL;
	
	if( metric == faiss::METRIC_L2 )
	{
		vectorNorms = cudaAlloc<float>(N);
		printf("cudaL2Norm(vectors=%p, n=%i, d=%i, output=%p, stream=%p)\n", vectors, N, D, vectorNorms, stream);
		
		timespec time_begin, time_enqueue, time_end;
		clock_gettime(CLOCK_REALTIME, &time_begin);
		
		const bool result = cudaL2Norm(
			vectors, sizeof(T),
			N, D, vectorNorms,
			true, stream);
			
		clock_gettime(CLOCK_REALTIME, &time_enqueue);
		CUDA(cudaStreamSynchronize(stream));
		clock_gettime(CLOCK_REALTIME, &time_end);

		if( !result )
			printf("cudaKNN() returned false\n");
		
		const double enqueue_time = time_diff(time_begin, time_enqueue);
		const double process_time = time_diff(time_begin, time_end);

		printf("cudaL2Norm   enqueue:  %.3f ms   process:  %.3f\n", enqueue_time, process_time);
	}
	
	printf("cudaKNN(vectors=%p, queries=%p, dsize=%zu, n=%i, m=%i, d=%i, k=%i, metric=%i, out_distances=%p, out_indices=%p, stream=%p)\n", vectors, queries, sizeof(T), N, M, D, K, metric, distances, indices, stream);
	
	double time_avg = 0.0;
	
	for( int r=0; r < runs; r++ )
	{
		timespec time_begin, time_enqueue, time_end;
		clock_gettime(CLOCK_REALTIME, &time_begin);
		
		const bool result = cudaKNN(
			vectors, queries, sizeof(T),
			N, M, D, K,
			metric, 
			vectorNorms,
			distances,
			indices,
			stream
		);

		clock_gettime(CLOCK_REALTIME, &time_enqueue);
		CUDA(cudaStreamSynchronize(stream));
		clock_gettime(CLOCK_REALTIME, &time_end);

		if( !result )
			printf("cudaKNN() returned false\n");
		
		const double enqueue_time = time_diff(time_begin, time_enqueue);
		const double process_time = time_diff(time_begin, time_end);

		if( r > 0 )
			time_avg += process_time;
		
		printf("cudaKNN   enqueue:  %.3f ms   process:  %.3f ms\n", enqueue_time, process_time);
	}

	CUDA(cudaFree(vectors));
	CUDA(cudaFree(queries));
	CUDA(cudaFree(distances));
	CUDA(cudaFree(indices));
	
	if( vectorNorms != NULL )
		CUDA(cudaFree(vectorNorms));
	
	return time_avg / double(runs-1);
}

void benchmark(int N, int M, int D, int K, double results[4], int runs=10, cudaStream_t stream=0)
{
	results[0] = test<float>(N, M, D, K, faiss::METRIC_L2, runs, stream);
	results[1] = test<float>(N, M, D, K, faiss::METRIC_INNER_PRODUCT, runs, stream);
	
	results[2] = test<half>(N, M, D, K, faiss::METRIC_L2, runs, stream);
	results[3] = test<half>(N, M, D, K, faiss::METRIC_INNER_PRODUCT, runs, stream);
}

int main( int argc, char* argv[] )
{
	int D[] = {5120, 1310720, 2621440};
	int N[] = {64, 512};
	int M[] = {1};
	int K[] = {4};

	cudaStream_t stream = NULL;
	CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));  // https://docs.nvidia.com/cuda/cuda-runtime-api/stream-sync-behavior.html#stream-sync-behavior
	
	double results[3][2][1][1][4];
	
	for( int d=0; d < 3; d++ )
		for( int n=0; n < 2; n++ )
			for( int m=0; m < 1; m++ )
				for( int k=0; k < 1; k++ )
					benchmark(N[n], M[m], D[d], K[k], results[d][n][m][k], 10, stream);
	
	for( int d=0; d < 3; d++ )
	{
		for( int n=0; n < 2; n++ )
		{
			for( int m=0; m < 1; m++ )
			{
				for( int k=0; k < 1; k++ )
				{
					printf("\nAverage time for (%i,%i) search queries over (%i,%i) vectors (k=%i)\n", M[m], D[d], N[n], D[d], K[k]);
					printf("  -- fp32, L2_norm:        %.3f ms\n", results[d][n][m][k][0]);
					printf("  -- fp32, inner_product:  %.3f ms\n", results[d][n][m][k][1]);
					printf("  -- fp16, L2_norm:        %.3f ms\n", results[d][n][m][k][2]);
					printf("  -- fp16, inner_product:  %.3f ms\n", results[d][n][m][k][3]);
				}
			}
		}
	}
	
	CUDA(cudaStreamDestroy(stream));
}
