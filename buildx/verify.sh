#!/bin/bash

# Function to print colored text
print_color() {
    local color=$1
    local text=$2
    echo -e "\033[${color}m${text}\033[0m"
}

# Function to print section headers with ASCII art
print_section_header() {
    local title=$1
    local color=$2
    print_color $color "================================="
    print_color $color "== $title"
    print_color $color "================================="
}

# Print system information
print_section_header "SYSTEM INFORMATION" "32"
print_color 34 "Kernel: $(uname -r)"
print_color 34 "Architecture: $(uname -m)"
print_color 34 "OS: $(lsb_release -d | awk -F'\t' '{print $2}')"
print_color 34 "Memory: $(free -h | awk '/^Mem:/ {print $2}')"
print_color 34 "CPU Cores: $(nproc)"

# Print NVIDIA components
print_section_header "NVIDIA COMPONENTS" "33"
print_color 35 "CUDA: $(nvcc --version | grep release | awk '{print $6,$7}')"
print_color 35 "cuDNN: $(cat /usr/include/cudnn_version.h | grep CUDNN_MAJOR -A 2 | awk '{print $3}' | tr '\n' '.')"
print_color 35 "TensorRT: $(dpkg -l | grep nvinfer | awk '{print $3}')"
print_color 35 "NCCL: $(dpkg -l | grep nccl | awk '{print $3}')"

# Print Python and basic libraries
print_section_header "PYTHON & BASIC LIBRARIES" "36"
print_color 37 "Python: $(python3 --version)"
print_color 37 "NumPy: $(python3 -c 'import numpy; print(numpy.__version__)')"
print_color 37 "SciPy: $(python3 -c 'import scipy; print(scipy.__version__)')"
print_color 37 "Pandas: $(python3 -c 'import pandas; print(pandas.__version__)' 2>/dev/null || echo 'NOT FOUND')"
print_color 37 "Matplotlib: $(python3 -c 'import matplotlib; print(matplotlib.__version__)')"

# Print deep learning frameworks
print_section_header "DEEP LEARNING FRAMEWORKS" "31"
print_color 33 "PyTorch: $(python3 -c 'import torch; print(torch.__version__)')"
print_color 33 "TorchVision: $(python3 -c 'import torchvision; print(torchvision.__version__)')"
print_color 33 "TorchAudio: $(python3 -c 'import torchaudio; print(torchaudio.__version__)' 2>/dev/null || echo 'NOT FOUND')"
print_color 33 "TensorFlow: $(python3 -c 'import tensorflow as tf; print(tf.__version__)')"
print_color 33 "JAX: $(python3 -c 'import jax; print(jax.__version__)')"
print_color 33 "ONNX: $(python3 -c 'import onnx; print(onnx.__version__)')"
print_color 33 "ONNX Runtime: $(python3 -c 'import onnxruntime; print(onnxruntime.__version__)' 2>/dev/null || echo 'NOT FOUND')"
print_color 33 "Triton: $(python3 -c 'import triton; print(triton.__version__)' 2>/dev/null || echo 'NOT FOUND')"

# Print media processing libraries
print_section_header "MEDIA PROCESSING" "35"
print_color 36 "FFmpeg: $(ffmpeg -version | head -n 1 | awk '{print $3}' 2>/dev/null || echo 'NOT FOUND')"
print_color 36 "GStreamer: $(gst-inspect-1.0 --version | head -n 1 | awk '{print $2}' 2>/dev/null || echo 'NOT FOUND')"
print_color 36 "OpenCV: $(python3 -c 'import cv2; print(cv2.__version__)')"
print_color 36 "Pillow: $(python3 -c 'import PIL; print(PIL.__version__)')"

# Print development tools
print_section_header "DEVELOPMENT TOOLS" "32"
print_color 34 "gcc: $(gcc --version | head -n 1 | awk '{print $3}')"
print_color 34 "g++: $(g++ --version | head -n 1 | awk '{print $3}')"
print_color 34 "CMake: $(cmake --version | head -n 1 | awk '{print $3}')"
print_color 34 "Make: $(make --version | head -n 1 | awk '{print $3}')"
print_color 34 "Git: $(git --version | awk '{print $3}')"
print_color 34 "Ninja: $(ninja --version)"
print_color 34 "Bazel: $(bazel --version 2>/dev/null || echo 'NOT FOUND')"

# Print HuggingFace components
print_section_header "HUGGING FACE COMPONENTS" "31"
print_color 33 "Transformers: $(python3 -c 'import transformers; print(transformers.__version__)')"
print_color 33 "XFormers: $(python3 -c 'import xformers; print(xformers.__version__)')"
print_color 33 "Hugging Face Hub: $(python3 -c 'import huggingface_hub; print(huggingface_hub.__version__)')"
print_color 33 "Diffusers: $(python3 -c 'import diffusers; print(diffusers.__version__)')"
print_color 33 "Accelerate: $(python3 -c 'import accelerate; print(accelerate.__version__)' 2>/dev/null || echo 'NOT FOUND')"
print_color 33 "Datasets: $(python3 -c 'import datasets; print(datasets.__version__)' 2>/dev/null || echo 'NOT FOUND')"

# Print disk space information
print_section_header "DISK SPACE" "34"
df -h | grep -E '^Filesystem|overlay'

print_color 32 "======== VERIFICATION COMPLETE ========"
print_color 34 "Current Date (UTC): $(date -u)"
print_color 34 "Current User: $(whoami)"
print_color 32 "Container verification completed."
