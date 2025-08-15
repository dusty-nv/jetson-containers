#!/usr/bin/env bash

# Function to detect system architecture and SBSA status
detect_system_info() {
    echo "=== System Architecture Detection ==="

    # Detect CUDA architecture
    local detected_arch="unknown"
    if command -v nvidia-smi >/dev/null 2>&1; then
        local gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>/dev/null | head -n1 || true)
        if [[ "$gpu_name" == *"Orin"* ]] || [[ "$gpu_name" == *"Jetson"* ]]; then
            detected_arch="tegra-aarch64"
        elif [[ "$gpu_name" == *"A100"* ]] || [[ "$gpu_name" == *"H100"* ]] || [[ "$gpu_name" == *"L40"* ]]; then
            detected_arch="aarch64"
        else
            detected_arch="x86_64"
        fi
        echo "• GPU: $gpu_name"
    fi
    echo "• Detected CUDA_ARCH: $detected_arch"

    # Detect SBSA status
    local is_sbsa="False"
    if [[ "$detected_arch" == "aarch64" ]] && [[ "$detected_arch" != "tegra-aarch64" ]]; then
        is_sbsa="True"
    fi
    echo "• Detected IS_SBSA: $is_sbsa"

    # Detect distribution
    local detected_distro="unknown"
    if [ -f /etc/os-release ]; then
        detected_distro=$(grep "^ID=" /etc/os-release | cut -d= -f2 | tr -d '"')
    fi
    echo "• Detected DISTRO: $detected_distro"

    echo "====================================="
    echo ""
}

# Detect system information and capture the detected architecture
detect_output=$(detect_system_info)
echo "$detect_output"

# Extract the detected architecture from the output
detected_arch=$(echo "$detect_output" | grep "• Detected CUDA_ARCH:" | cut -d: -f2 | xargs)

# If tegra, exit with message
if [[ "$detected_arch" == "tegra-aarch64" ]]; then
  echo "Not supported on Tegra architecture"
  exit 0
fi

nvcc /test/test_nvshmem.cu \
  -I/usr/include/nvshmem \
  -I/usr/local/cuda/include \
  -L/usr/lib/aarch64-linux-gnu \
  -lnvshmem_host -lnvshmem -lcudart \
  -Xlinker -rpath=/usr/lib/aarch64-linux-gnu \
  -Wno-deprecated-gpu-targets \
  -Wno-deprecated-declarations \
  -o /test/test_nvshmem

/test/test_nvshmem
