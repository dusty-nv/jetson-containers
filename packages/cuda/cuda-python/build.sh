#!/usr/bin/env bash
set -ex

echo "Building cuda-python ${CUDA_PYTHON_VERSION}"
   
git clone --branch v${CUDA_PYTHON_VERSION} --depth=1 https://github.com/NVIDIA/cuda-python /opt/cuda-python
cd /opt/cuda-python

# NOW CUDA-PYTHON HAS 3 STRUCTURES

# Get the number of CPU cores
export MAX_JOBS=$(nproc)  # or use `sysctl -n hw.ncpu` on macOS

echo "Building with MAX_JOBS=${MAX_JOBS} cores..."

# Build cuda_core wheel
cd cuda_core
pip3 wheel . --no-deps --wheel-dir /opt --verbose

# Build cuda_bindings wheel
cd /opt/cuda-python/cuda_bindings
pip3 wheel . --no-deps --wheel-dir /opt --verbose

cd /opt/
rm -rf /opt/cuda-python

pip3 install /opt/cuda*.whl
pip3 show cuda_core && pip3 show cuda_bindings && python3 -c 'import cuda; print(cuda.__version__)'

twine upload --verbose /opt/cuda*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
