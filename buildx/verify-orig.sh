#!/bin/bash

# Set UTF-8 locale
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8

# Header formatting
print_header() {
    echo -e "\n\e[1;34m======== $1 ========\e[0m"
}

# Check component with color output and show path
check_component() {
    echo -ne "\e[1;33m$1:\e[0m "
    if eval "$2" &>/dev/null; then
        echo -e "\e[1;32mOK\e[0m"
        eval "$3"
        
        # Try to find binary location for commands
        if command -v $1 &>/dev/null; then
            echo -e "  \e[90m→ Binary: $(which $1)\e[0m"
        fi
        
        # For Python modules, show their location
        if [[ "$2" == *"python3 -c 'import "* ]]; then
            module_name=$(echo "$2" | sed -n "s/.*import \([a-zA-Z0-9_\.]*\).*/\1/p" | cut -d. -f1)
            if [ -n "$module_name" ]; then
                module_path=$(python3 -c "import $module_name, os; print(os.path.dirname($module_name.__file__))" 2>/dev/null)
                if [ -n "$module_path" ]; then
                    echo -e "  \e[90m→ Module: $module_path\e[0m"
                fi
            fi
        fi
        
        # For libraries, show their location if mentioned in the check
        if [[ "$2" == *"ldconfig"* ]]; then
            lib_name=$(echo "$2" | grep -o "lib[a-zA-Z0-9]*" | head -1)
            if [ -n "$lib_name" ]; then
                lib_path=$(ldconfig -p | grep "$lib_name" | head -1 | awk '{print $4}')
                if [ -n "$lib_path" ]; then
                    echo -e "  \e[90m→ Library: $lib_path\e[0m"
                fi
            fi
        fi
    else
        echo -e "\e[1;31mNOT FOUND\e[0m"
    fi
}

# Current date, time and user info
print_header "CONTAINER INFORMATION"
echo "Current Date/Time (UTC): 2025-03-27 11:51:03"
echo "Local Date/Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Current User: kairin"
echo "Hostname: $(hostname)"
echo "Container Image: kairin/001:2025-03-27-full-stack"

# System Information
print_header "SYSTEM INFORMATION"
echo "Kernel: $(uname -r)"
echo "Architecture: $(uname -m)"
if [ -f /etc/os-release ]; then
    source /etc/os-release
    echo "OS: $PRETTY_NAME"
fi
echo "Memory: $(free -h | grep Mem | awk '{print $2}')"
echo "CPU Cores: $(nproc)"

# NVIDIA Components
print_header "NVIDIA COMPONENTS"
check_component "CUDA" "nvcc --version" "nvcc --version | head -n1"
check_component "CUDA libraries" "ldconfig -p | grep -q libcuda" "echo CUDA libs: $(ldconfig -p | grep libcuda | wc -l)"
check_component "cuDNN" "ldconfig -p | grep -q libcudnn" "echo cuDNN libs: $(ldconfig -p | grep libcudnn | wc -l)"
check_component "TensorRT" "ldconfig -p | grep -q libnvinfer" "echo TensorRT: $(dpkg -l | grep libnvinfer | awk '{print $3}' | head -n1)"
check_component "NVIDIA drivers" "nvidia-smi" "nvidia-smi --query-gpu=driver_version,name,memory.total,memory.free --format=csv"
check_component "NCCL" "ldconfig -p | grep -q libnccl" "echo NCCL libs: $(ldconfig -p | grep libnccl | wc -l)"

# Python and ML Framework
print_header "PYTHON & BASIC LIBRARIES"
check_component "Python" "which python3" "python3 --version"
check_component "pip" "which pip3" "pip3 --version"
check_component "NumPy" "python3 -c 'import numpy'" "python3 -c 'import numpy; print(f\"NumPy {numpy.__version__}\")'"
check_component "SciPy" "python3 -c 'import scipy'" "python3 -c 'import scipy; print(f\"SciPy {scipy.__version__}\")'"
check_component "Pandas" "python3 -c 'import pandas'" "python3 -c 'import pandas; print(f\"Pandas {pandas.__version__}\")'"
check_component "Matplotlib" "python3 -c 'import matplotlib'" "python3 -c 'import matplotlib; print(f\"Matplotlib {matplotlib.__version__}\")'"
check_component "Jupyter" "which jupyter" "jupyter --version | head -n1"

# Deep Learning Frameworks
print_header "DEEP LEARNING FRAMEWORKS"
check_component "PyTorch" "python3 -c 'import torch'" "python3 -c 'import torch; print(f\"PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}\")'"
check_component "TorchVision" "python3 -c 'import torchvision'" "python3 -c 'import torchvision; print(f\"TorchVision {torchvision.__version__}\")'"
check_component "TorchAudio" "python3 -c 'import torchaudio'" "python3 -c 'import torchaudio; print(f\"TorchAudio {torchaudio.__version__}\")'"
check_component "TensorFlow" "python3 -c 'import tensorflow'" "python3 -c 'import tensorflow; print(f\"TensorFlow {tensorflow.__version__}\")'" || true
check_component "JAX" "python3 -c 'import jax'" "python3 -c 'import jax; print(f\"JAX {jax.__version__}\")'" || true
check_component "ONNX" "python3 -c 'import onnx'" "python3 -c 'import onnx; print(f\"ONNX {onnx.__version__}\")'"
check_component "ONNX Runtime" "python3 -c 'import onnxruntime'" "python3 -c 'import onnxruntime; print(f\"ONNX Runtime {onnxruntime.__version__}\")'" || true
check_component "Triton" "python3 -c 'import triton'" "python3 -c 'import triton; print(f\"Triton {triton.__version__}\")'" || true

# NVIDIA ML Tools
print_header "NVIDIA ML TOOLS"
check_component "DALI" "python3 -c 'import nvidia.dali'" "python3 -c 'import nvidia.dali; print(f\"NVIDIA DALI available\")'" || true
check_component "APEX" "python3 -c 'import apex'" "python3 -c 'import apex; print(f\"NVIDIA APEX available\")'" || true
check_component "cupy" "python3 -c 'import cupy'" "python3 -c 'import cupy; print(f\"CuPy {cupy.__version__}\")'" || true

# Hugging Face Components
print_header "HUGGING FACE COMPONENTS"
check_component "Transformers" "python3 -c 'import transformers'" "python3 -c 'import transformers; print(f\"Transformers {transformers.__version__}\")'"
check_component "XFormers" "python3 -c 'import xformers'" "python3 -c 'import xformers; print(f\"XFormers {xformers.__version__}\")'"
check_component "Hugging Face Hub" "python3 -c 'import huggingface_hub'" "python3 -c 'import huggingface_hub; print(f\"Hugging Face Hub {huggingface_hub.__version__}\")'"
check_component "Diffusers" "python3 -c 'import diffusers'" "python3 -c 'import diffusers; print(f\"Diffusers {diffusers.__version__}\")'" || true
check_component "Accelerate" "python3 -c 'import accelerate'" "python3 -c 'import accelerate; print(f\"Accelerate {accelerate.__version__}\")'" || true
check_component "Datasets" "python3 -c 'import datasets'" "python3 -c 'import datasets; print(f\"Datasets {datasets.__version__}\")'" || true

# Media processing
print_header "MEDIA PROCESSING"
check_component "FFmpeg" "which ffmpeg" "ffmpeg -version | head -n1"
check_component "GStreamer" "which gst-launch-1.0" "gst-launch-1.0 --version | head -n1"
check_component "OpenCV" "python3 -c 'import cv2'" "python3 -c 'import cv2; print(f\"OpenCV version: {cv2.__version__}\")'"
check_component "Pillow" "python3 -c 'import PIL'" "python3 -c 'import PIL; print(f\"Pillow version: {PIL.__version__}\")'"

# Development Tools
print_header "DEVELOPMENT TOOLS"
check_component "gcc" "which gcc" "gcc --version | head -n1"
check_component "g++" "which g++" "g++ --version | head -n1"
check_component "CMake" "which cmake" "cmake --version | head -n1"
check_component "Make" "which make" "make --version | head -n1"
check_component "Git" "which git" "git --version"
check_component "Ninja" "which ninja" "ninja --version"
check_component "Bazel" "which bazel" "bazel --version"

# HuggingFace CLI and tools
print_header "HUGGINGFACE TOOLS"
check_component "huggingface-cli" "which huggingface-cli" "huggingface-cli --version || echo 'Version info not available'"
check_component "huggingface-downloader" "which huggingface-downloader" "huggingface-downloader --help | head -n1"
check_component "huggingface-benchmark" "which huggingface-benchmark.py" "ls -la $(which huggingface-benchmark.py 2>/dev/null)" || true

# Environment variables
print_header "ENVIRONMENT VARIABLES"
echo "PYTHONPATH: $PYTHONPATH"
echo "CUDA_HOME: $CUDA_HOME"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "TRANSFORMERS_CACHE: $TRANSFORMERS_CACHE"
echo "HUGGINGFACE_HUB_CACHE: $HUGGINGFACE_HUB_CACHE"
echo "HF_HOME: $HF_HOME"
echo "JAX_CACHE_DIR: $JAX_CACHE_DIR"
echo "XFORMERS_FORCE_DISABLE_TRITON: $XFORMERS_FORCE_DISABLE_TRITON"
echo "DIFFUSERS_FORCE_DISABLE_TRITON: $DIFFUSERS_FORCE_DISABLE_TRITON"

# Additional useful paths
print_header "IMPORTANT PATHS"
echo -e "\e[1;33mPython Site Packages:\e[0m"
python3 -c "import site; print('\n'.join(site.getsitepackages()))"
echo
echo -e "\e[1;33mPython Distribution Packages:\e[0m"
find /usr/local/lib/python*/ -maxdepth 1 -name "dist-packages" 2>/dev/null || echo "No dist-packages found"
echo
echo -e "\e[1;33mShared Libraries:\e[0m"
echo "/usr/local/lib"
echo "/usr/lib"
echo
echo -e "\e[1;33mExecutables:\e[0m"
echo "/usr/local/bin"
echo "/usr/bin"

# CUDA capability
print_header "CUDA CAPABILITIES"
check_component "PyTorch CUDA" "python3 -c 'import torch; torch.cuda.is_available()'" "python3 -c 'import torch; print(f\"Available GPU(s): {torch.cuda.device_count()}\"); [print(f\"GPU {i}: {torch.cuda.get_device_name(i)}\") for i in range(torch.cuda.device_count())]'"
check_component "CUDA device list" "nvidia-smi" "nvidia-smi -L"

# Python installed packages
print_header "PYTHON PACKAGES SUMMARY"
echo "Total packages installed: $(pip list | tail -n +3 | wc -l)"
echo "To see all packages: pip list"
echo "To see package locations: pip show <package_name>"

# Disk space
print_header "DISK SPACE"
df -h | grep -E "(Filesystem|/$|/tmp$|/var$)"

print_header "VERIFICATION COMPLETE"
echo "Current Date (UTC): 2025-03-27 11:51:03"
echo "Current User: kairin"
echo -e "\e[1;32mContainer verification completed.\e[0m"
