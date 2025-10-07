#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>
#include <cutensor.h>

// --- Compatibility shims: v1 vs v2+ ---
#if defined(CUTENSOR_VERSION) && (CUTENSOR_VERSION >= 20000)
  // cuTENSOR v2+ (opaque handle)
  #define CUTENSOR_CREATE(h)   cutensorCreate(h)
  #define CUTENSOR_DESTROY(h)  cutensorDestroy(h)
  #define CUTENSOR_IS_V2 1
#else
  // cuTENSOR v1.x
  #define CUTENSOR_CREATE(h)   cutensorInit(h)
  #define CUTENSOR_DESTROY(h)  do{}while(0)
  #define CUTENSOR_IS_V2 0
#endif

#define CHECK_CUDA(call) do { \
  cudaError_t _e = (call); \
  if (_e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
    exit(1); \
  } \
} while(0)

#define CHECK_CUTENSOR(call) do { \
  cutensorStatus_t _s = (call); \
  if (_s != CUTENSOR_STATUS_SUCCESS) { \
    fprintf(stderr, "cuTENSOR error %s:%d: %s\n", __FILE__, __LINE__, cutensorGetErrorString(_s)); \
    exit(1); \
  } \
} while(0)

int main() {
  // --- Handle + version (works on both) ---
  cutensorHandle_t handle;
  CHECK_CUTENSOR(CUTENSOR_CREATE(&handle));

#if CUTENSOR_IS_V2
  // v2+: single-integer version
  int ver = (int)cutensorGetVersion();
  printf("cuTENSOR version: %d\n", ver);
#else
  // v1: major/minor/patch via pointers
  int major=0, minor=0, patch=0;
  cutensorGetVersion(&major, &minor, &patch);
  printf("cuTENSOR version: %d.%d.%d\n", major, minor, patch);
#endif

  // ---------- Minimal test ----------
  // We keep a real contraction in v1 (existing API),
  // and a lightweight runtime sanity path in v2+ (until v2 contraction APIs are added).

#if !CUTENSOR_IS_V2
  // ======= v1 contraction path (keeps your existing logic) =======
  const int64_t m = 4, n = 4, k = 4;
  const float alpha = 1.0f, beta = 0.0f;

  std::vector<float> hA(m*k, 1.0f);
  std::vector<float> hB(k*n, 2.0f);
  std::vector<float> hC(m*n, 0.0f);

  float *A, *B, *C;
  CHECK_CUDA(cudaMalloc(&A, hA.size()*sizeof(float)));
  CHECK_CUDA(cudaMalloc(&B, hB.size()*sizeof(float)));
  CHECK_CUDA(cudaMalloc(&C, hC.size()*sizeof(float)));
  CHECK_CUDA(cudaMemcpy(A, hA.data(), hA.size()*sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(B, hB.data(), hB.size()*sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(C, hC.data(), hC.size()*sizeof(float), cudaMemcpyHostToDevice));

  int32_t modeA[2] = {'m','k'};
  int32_t modeB[2] = {'k','n'};
  int32_t modeC[2] = {'m','n'};
  int64_t extentA[2] = {m, k};
  int64_t extentB[2] = {k, n};
  int64_t extentC[2] = {m, n};

  cutensorTensorDescriptor_t descA, descB, descC;
  int64_t strideA[2] = {k, 1};
  int64_t strideB[2] = {n, 1};
  int64_t strideC[2] = {n, 1};

  CHECK_CUTENSOR(cutensorInitTensorDescriptor(&handle, &descA,
                  2, extentA, strideA, CUDA_R_32F, CUTENSOR_OP_IDENTITY));
  CHECK_CUTENSOR(cutensorInitTensorDescriptor(&handle, &descB,
                  2, extentB, strideB, CUDA_R_32F, CUTENSOR_OP_IDENTITY));
  CHECK_CUTENSOR(cutensorInitTensorDescriptor(&handle, &descC,
                  2, extentC, strideC, CUDA_R_32F, CUTENSOR_OP_IDENTITY));

  cutensorOperationDescriptor_t opDesc;
  CHECK_CUTENSOR(cutensorInitContraction(&handle, &opDesc,
                  &descA, modeA, CUDA_R_32F,
                  &descB, modeB, CUDA_R_32F,
                  &descC, modeC, CUDA_R_32F,
                  &descC, modeC, CUDA_R_32F,
                  CUTENSOR_OP_MUL, CUTENSOR_COMPUTE_32F));

  cutensorContractionFind_t find;
  CHECK_CUTENSOR(cutensorInitContractionFind(&handle, &find, CUTENSOR_ALGO_DEFAULT));

  size_t worksize = 0;
  CHECK_CUTENSOR(cutensorContractionGetWorkspace(&handle, &opDesc, &find,
                                                 CUTENSOR_WORKSPACE_RECOMMENDED, &worksize));
  void* workspace = nullptr;
  if (worksize > 0) CHECK_CUDA(cudaMalloc(&workspace, worksize));

  cutensorContractionPlan_t plan;
  CHECK_CUTENSOR(cutensorInitContractionPlan(&handle, &plan, &opDesc, &find, worksize));

  CHECK_CUTENSOR(cutensorContraction(&handle, &plan,
                     &alpha, A, B, &beta, C, C,
                     workspace, worksize, 0));

  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaMemcpy(hC.data(), C, hC.size()*sizeof(float), cudaMemcpyDeviceToHost));

  printf("C[0] = %.1f (expected 8.0)\n", hC[0]);

  if (workspace) cudaFree(workspace);
  cudaFree(A); cudaFree(B); cudaFree(C);
#else
  // ======= v2+ temporary path =======
  // We only validate that the runtime is present and a handle can be created.
  // (The v2 contraction API names differ; add them here if you want the full op.)
  printf("cuTENSOR v2+ detected: basic runtime sanity passed (handle created)\n");
#endif

  CUTENSOR_DESTROY(&handle);
  return 0;
}
