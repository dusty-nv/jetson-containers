#!/bin/bash
# Environment information script for GitHub Actions workflows
# Records comprehensive environment variables and system info

echo "=== Complete Environment Variables ==="
echo "All environment variables:"
env | sort
echo ""
echo "=== Key Environment Variables ==="
echo "PATH: $PATH"
echo "HOME: $HOME"
echo "USER: $USER"
echo "PWD: $PWD"
echo "SHELL: $SHELL"
echo "LANG: $LANG"
echo "LC_ALL: $LC_ALL"
echo "TZ: $TZ"
echo "GITHUB_* variables:"
env | grep "^GITHUB_" | sort
echo ""
echo "=== System Information ==="
echo "Docker version:"
docker --version || echo "Docker not available"
echo "Python version:"
python3 --version || echo "Python3 not available"
echo "CUDA version:"
nvcc --version || echo "NVCC not available"
echo "Available disk space:"
df -h
echo "Memory usage:"
free -h
echo "CPU info:"
lscpu | head -20 || echo "lscpu not available"
echo "GPU info:"
nvidia-smi || echo "nvidia-smi not available"
