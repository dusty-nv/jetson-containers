// test_nvshmem.cu
#include <iostream>
#include <nvshmem/nvshmem.h>
#include <nvshmem/nvshmemx.h>

__global__ void ping_self(int* buf) {
#if __CUDA_ARCH__ >= 700
    // Simple write to self (can run with single PE)
    nvshmem_int_p(buf, 123, nvshmem_my_pe());
#else
    // Lower architectures don't use device atomic/LDST path, avoiding atomicAdd_system symbols
    if (threadIdx.x == 0) *buf = 123;
#endif
}

int main() {
    cudaSetDevice(0);
    nvshmem_init();

    int* buf = (int*)nvshmem_malloc(sizeof(int));
    if (!buf) { std::cerr << "malloc failed\n"; return 1; }

    // Grid launch (1 block is sufficient)
    ping_self<<<1, 32>>>(buf);
    cudaDeviceSynchronize();

    int host = -1;
    cudaMemcpy(&host, buf, sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "device wrote: " << host << "\n";

    nvshmem_free(buf);
    nvshmem_finalize();
    return 0;
}
