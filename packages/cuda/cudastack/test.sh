#!/usr/bin/env bash
set -e

export SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TEST_DIR="${SCRIPT_DIR}/test"


echo "Testing CUDA Stack installation..."
echo "=================================="

# Track test results
TESTS_RUN=0
TESTS_PASSED=0
TESTS_FAILED=0

# Helper function to run test scripts
run_test() {
    local test_name=$1
    local test_script=$2

    if [ ! -f "${test_script}" ]; then
        echo "⚠ Test script not found: ${test_script}"
        return
    fi

    # Check if script is not empty (some test scripts are just stubs)
    if [ ! -s "${test_script}" ] || [ "$(wc -l < "${test_script}")" -le 1 ]; then
        echo "⊘ Skipping empty test: ${test_name}"
        return
    fi

    echo ""
    echo "Running ${test_name}..."
    echo "------------------------"

    TESTS_RUN=$((TESTS_RUN + 1))

    if bash "${test_script}"; then
        echo "✓ ${test_name} passed"
        TESTS_PASSED=$((TESTS_PASSED + 1))
    else
        echo "✗ ${test_name} failed"
        TESTS_FAILED=$((TESTS_FAILED + 1))
    fi
}

# Test cuDNN
if [ "${WITH_CUDNN:-1}" = "1" ]; then
    run_test "cuDNN" "${TEST_DIR}/test_cudnn.sh"
fi

# Test NCCL
if [ "${WITH_NCCL:-1}" = "1" ]; then
    run_test "NCCL" "${TEST_DIR}/test_nccl.sh"
fi

# Test TensorRT
if [ "${WITH_TENSORRT:-0}" = "1" ]; then
    run_test "TensorRT" "${TEST_DIR}/test_tensorrt.sh"
fi

# Test CUDSS
if [ "${WITH_CUDSS:-1}" = "1" ]; then
    run_test "CUDSS" "${TEST_DIR}/test_cudss.sh"
fi

# Test cuSPARSELt
if [ "${WITH_CUSPARSELT:-1}" = "1" ]; then
    run_test "cuSPARSELt" "${TEST_DIR}/test_cusparselt.sh"
fi

# Test cuTensor
# if [ "${WITH_CUTENSOR:-1}" = "1" ]; then
#     run_test "cuTensor" "${TEST_DIR}/test_cutensor.sh"
# fi

# Test GDRCopy
if [ "${WITH_GDRCOPY:-1}" = "1" ]; then
    run_test "GDRCopy" "${TEST_DIR}/test_gdrcopy.sh"
fi

# Test NVPL (SBSA only)
if [ "${WITH_NVPL:-0}" = "1" ]; then
    run_test "NVPL" "${TEST_DIR}/test_nvpl.sh"
fi

# Test NVSHMEM
if [ "${WITH_NVSHMEM:-1}" = "1" ]; then
    run_test "NVSHMEM" "${TEST_DIR}/test_nvshmem.sh"
fi

# Print summary
echo ""
echo "=================================="
echo "Test Summary:"
echo "  Total tests run: ${TESTS_RUN}"
echo "  Passed: ${TESTS_PASSED}"
echo "  Failed: ${TESTS_FAILED}"
echo "=================================="

if [ ${TESTS_FAILED} -gt 0 ]; then
    echo "✗ Some tests failed!"
    exit 1
else
    echo "✓ All CUDA stack tests passed!"
    exit 0
fi
