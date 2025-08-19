#!/usr/bin/env bash
set -ex

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of torchcodec ${TORCHCODEC_VERSION}"
	exit 1
fi
echo "Installing torchcodec ${TORCHCODEC_VERSION}"
# Install torchcodec based on official documentation
# Check if CUDA is available and install appropriate version
if command -v nvidia-smi >/dev/null 2>&1; then
	echo "CUDA detected, installing CUDA-enabled torchcodec"
	# Get CUDA version for the appropriate index URL
	CUDA_VERSION=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' | head -1)
	CUDA_VERSION_SHORT=$(echo $CUDA_VERSION | sed 's/\.//')

	# Try CUDA-enabled installation first
    echo "Attempting CUDA installation for ${CUDA_VERSION} (${CUDA_VERSION_SHORT})"
	#pip3 install torchcodec --index-url=https://download.pytorch.org/whl/cu${CUDA_VERSION_SHORT}
else
	echo "No CUDA detected, installing CPU-only torchcodec"
	#pip3 install torchcodec==${TORCHCODEC_VERSION}
fi

# Verify installation
#pip3 show torchcodec && python3 -c 'import torchcodec; print(f"torchcodec version: {torchcodec.__version__}")'
