# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Jetson Containers is a modular container build system that provides optimized AI/ML packages for NVIDIA Jetson edge AI platforms. The project supports JetPack 6.2+ (CUDA 12.6+) and JetPack 7 (CUDA 13.x), including Ubuntu 24.04 containers and ARM SBSA for GH200/GB200.

## Common Development Commands

### Building Containers
```bash
# Basic build
jetson-containers build pytorch

# Named build with multiple packages
jetson-containers build --name=my_container pytorch transformers ros:humble-desktop

# Build with specific versions
CUDA_VERSION=12.6 jetson-containers build pytorch
LSB_RELEASE=24.04 CUDA_VERSION=12.9 PYTORCH_VERSION=2.8 jetson-containers build vllm

# Skip tests during build
jetson-containers build --skip-tests=all pytorch

# Push to registry
jetson-containers build --push=dockerhub_username pytorch
```

### Running Containers
```bash
# Run with autotag (finds compatible version)
jetson-containers run $(autotag pytorch)

# Run with specific command
jetson-containers run $(autotag pytorch) python3 script.py

# Run with CSI camera support
jetson-containers run --csi2webcam $(autotag pytorch)
```

### Testing
```bash
# Test specific packages
jetson-containers build --test-only=pytorch,torchvision pytorch torchvision

# Run individual package tests (inside container)
python3 /test.py  # Most packages
bash /test.sh     # Some packages
```

### Code Quality
```bash
# Install pre-commit hooks
pre-commit install

# Run formatters
pre-commit run --all-files

# Manual formatting
black jetson_containers/
flake8 jetson_containers/
```

## Architecture

### Package System
- **Location**: `packages/` directory organized by category (llm/, ml/, robotics/, etc.)
- **Components**: Each package has:
  - `Dockerfile` with metadata header
  - `config.py` or `config.yml` for dependencies and build logic
  - `test.py` or `test.sh` for automated testing
  - `README.md` with usage instructions

### Build System (`jetson_containers/`)
- **Dependency Resolution**: Automatically resolves package dependencies based on JetPack/CUDA versions
- **Multi-stage Builds**: Efficient Docker layer caching
- **Version Management**: Dynamic version selection based on platform compatibility
- **Logging**: All builds logged to `logs/` with timestamps and reproduction scripts

### Key Environment Variables
- `CUDA_VERSION`: Override CUDA version (12.6, 12.9, etc.)
- `LSB_RELEASE`: Override Ubuntu version (22.04, 24.04)
- `PYTORCH_VERSION`: Override PyTorch version
- `PYTHON_VERSION`: Override Python version
- `HUGGINGFACE_TOKEN`: HuggingFace API token for model downloads

### Testing Strategy
- Tests run automatically during builds
- Version verification tests ensure correct installations
- Import/functionality tests validate package operations
- Performance benchmarks available for optimization
- Test logs saved in `logs/*/test/`

### CI/CD
- GitHub Actions workflows for automated builds
- Self-hosted runners on Jetson hardware
- Monthly builds for changed packages
- Automated pushing to Docker Hub

## Code Style Requirements
- Python code formatted with Black (line length 120)
- Flake8 for linting (max line length 120, ignoring E203, W503)
- Pre-commit hooks enforce formatting on whitelisted packages
- To add a package to formatting checks, update `.pre-commit-config.yaml`

## Development Tips
- Use `autotag` to find/build compatible containers automatically
- Build logs contain exact commands for reproduction
- The `--simulate` flag shows what would be built without executing
- Volume mount `/data` is automatically available for persistent storage
- Default runtime is `nvidia` with GPU access enabled