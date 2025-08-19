#!/usr/bin/env bash
set -ex

# setting envariable I_CONFIRM_THIS_IS_NOT_A_LICENSE_VIOLATION to 1
export I_CONFIRM_THIS_IS_NOT_A_LICENSE_VIOLATION=1
echo "Building torchcodec ${TORCHCODEC_VERSION} from source"

# Since we're forcing a build, we'll install torchcodec from source
# First, ensure we have the required dependencies
echo "Installing build dependencies..."

# Install required packages for building
apt-get update && apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    libavformat-dev \
    libavcodec-dev \
    libavutil-dev \
    libswscale-dev \
    libswresample-dev \
    libavfilter-dev \
    libavdevice-dev

# Clone and build torchcodec from source
echo "Cloning torchcodec ${TORCHCODEC_VERSION}"
BRANCH_VERSION=$(echo "$TORCHCODEC_VERSION" | sed 's/^\([0-9]*\.[0-9]*\)\.0$/\1/')
git clone --branch=v${TORCHCODEC_VERSION} --recursive --depth=1 https://github.com/pytorch/torchcodec /opt/torchcodec ||
git clone --branch=main --recursive --depth=1 https://github.com/pytorch/torchcodec /opt/torchcodec

cd /opt/torchcodec

# Set environment variables for building
export BUILD_VERSION=${TORCHCODEC_VERSION}
export TORCH_CUDA_ARCH_LIST="5.3;6.2;7.2;8.7;8.9;9.0+PTX"

# Build torchcodec
echo "Building torchcodec..."
python3 setup.py --verbose bdist_wheel --dist-dir /opt

cd ../
rm -rf /opt/torchcodec

# Install the built wheel
pip3 install /opt/torchcodec*.whl

# Verify installation
pip3 show torchcodec && python3 -c 'import torchcodec; print(f"torchcodec version: {torchcodec.__version__}")'

# Upload to repository if configured
twine upload --verbose /opt/torchcodec*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
