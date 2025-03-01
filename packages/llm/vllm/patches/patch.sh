#!/bin/bash

# Set the CUDA architecture to use (configurable via environment variable)
CUDA_ARCHS_TO_USE=${CUDA_ARCHS_TO_USE:-"8.7"}

# Modify CMakeLists.txt
echo "Updating CMakeLists.txt..."
# Update CUDA_SUPPORTED_ARCHS
sed -i "s/set(CUDA_SUPPORTED_ARCHS \"[^\"]*\")/set(CUDA_SUPPORTED_ARCHS \"${CUDA_ARCHS_TO_USE}\")/" CMakeLists.txt
# Update all cuda_archs_loose_intersection calls to use "${CUDA_ARCHS}" for both args
sed -i 's/\(cuda_archs_loose_intersection(\s*\([^,]*\),\s*\)"[^"]*"\s*,\s*"[^"]*"\s*\)/\1"${CUDA_ARCHS}", "${CUDA_ARCHS}")/' CMakeLists.txt

# Modify vllm_flash_attn.cmake
echo "Updating vllm_flash_attn.cmake..."
# Ensure patch_vllm_flash_attn is defined (assuming the patch file exists)
sed -i '/set(patch_vllm_flash_attn git apply \/tmp\/vllm\/fa.diff)/d' cmake/external_projects/vllm_flash_attn.cmake
sed -i '/cmake\/external_projects\/vllm_flash_attn\.cmake/a set(patch_vllm_flash_attn git apply /tmp/vllm/fa.diff)' cmake/external_projects/vllm_flash_attn.cmake
# Add PATCH_COMMAND and UPDATE_DISCONNECTED after BINARY_DIR in FetchContent_Declare
sed -i '/BINARY_DIR/ a\
          PATCH_COMMAND ${patch_vllm_flash_attn}\
          UPDATE_DISCONNECTED 1
' cmake/external_projects/vllm_flash_attn.cmake

# Modify __init__.py
echo "Updating guided_decoding/__init__.py..."
# Remove the CPU architecture check block (safely delete lines matching the pattern)
sed -i '/from vllm\.platforms import current_platform/d' vllm/model_executor/guided_decoding/__init__.py
sed -i '/if current_platform\.get_cpu_architecture() is not CpuArchEnum\.X86:/,/^ *$/d' vllm/model_executor/guided_decoding/__init__.py

# Generate diff for verification
echo "Generating diff..."
git diff > /tmp/vllm_diff.txt
echo "Changes applied. See /tmp/vllm_diff.txt for the diff."