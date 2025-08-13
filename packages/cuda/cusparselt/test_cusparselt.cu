#include <iostream>
#include <cuda_runtime.h>
#include <cusparseLt.h>

static const char* status_str(cusparseStatus_t s){
  switch(s){
    case CUSPARSE_STATUS_SUCCESS: return "SUCCESS";
    case CUSPARSE_STATUS_NOT_INITIALIZED: return "NOT_INITIALIZED";
    case CUSPARSE_STATUS_ALLOC_FAILED: return "ALLOC_FAILED";
    case CUSPARSE_STATUS_INVALID_VALUE: return "INVALID_VALUE";
    case CUSPARSE_STATUS_ARCH_MISMATCH: return "ARCH_MISMATCH";
    case CUSPARSE_STATUS_MAPPING_ERROR: return "MAPPING_ERROR";
    case CUSPARSE_STATUS_EXECUTION_FAILED: return "EXECUTION_FAILED";
    case CUSPARSE_STATUS_INTERNAL_ERROR: return "INTERNAL_ERROR";
    case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED: return "MATRIX_TYPE_NOT_SUPPORTED";
    case CUSPARSE_STATUS_ZERO_PIVOT: return "ZERO_PIVOT";
    case CUSPARSE_STATUS_NOT_SUPPORTED: return "NOT_SUPPORTED";
    case CUSPARSE_STATUS_INSUFFICIENT_RESOURCES: return "INSUFFICIENT_RESOURCES";
    default: return "UNKNOWN";
  }
}

int main() {
    int ndev = 0;
    cudaGetDeviceCount(&ndev);
    if (ndev <= 0) { std::cerr << "No CUDA devices visible\n"; return 2; }
    cudaDeviceProp p{};
    cudaGetDeviceProperties(&p, 0);
    std::cout << "GPU: " << p.name << "  CC " << p.major << "." << p.minor << "\n";

    cusparseLtHandle_t handle;
    cusparseStatus_t s = cusparseLtInit(&handle);
    if (s != CUSPARSE_STATUS_SUCCESS) {
        std::cerr << "cusparselt FAILED: " << (int)s << " (" << status_str(s) << ")\n";
        if (s == CUSPARSE_STATUS_ARCH_MISMATCH) {
            std::cout << "cusparselt FAILED: ARCH_MISMATCH - wrong binary for this GPU\n";
            return 1; // fail on ARCH_MISMATCH to indicate compatibility issue
        }
        return 1;
    }
    std::cout << "cusparselt OK\n";
    cusparseLtDestroy(&handle);
    return 0;
}
