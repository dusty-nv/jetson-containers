#include <iostream>
#include <cusparseLt.h>

int main() {
    cusparseLtHandle_t handle;
    cusparseStatus_t s = cusparseLtInit(&handle);
    if (s != CUSPARSE_STATUS_SUCCESS) {
        std::cerr << "cusparselt FAILED: " << s << "\n";
        return 1;
    }
    std::cout << "cusparselt OK\n";
    cusparseLtDestroy(&handle);
    return 0;
}