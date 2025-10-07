#!/usr/bin/env bash
echo "Building cuda-python $CUDA_PYTHON_VERSION"
set -ex

SRC=/opt/cuda-python
WHL=/opt/wheels

export MAX_JOBS="$(nproc)"
export CMAKE_BUILD_PARALLEL_LEVEL=$MAX_JOBS
echo "Building with MAX_JOBS=$MAX_JOBS and CMAKE_BUILD_PARALLEL_LEVEL=$CMAKE_BUILD_PARALLEL_LEVEL"


git clone --branch v$CUDA_PYTHON_VERSION --depth=1 https://github.com/NVIDIA/cuda-python $SRC || git clone --depth=1 https://github.com/NVIDIA/cuda-python $SRC

# NOW CUDA-PYTHON HAS 3 STRUCTURES
if [ $(vercmp $CUDA_PYTHON_VERSION "12.6") -gt 0 ]; then
  # Build cuda_core wheel
  cd $SRC/cuda_core
  pip3 wheel . --no-deps --wheel-dir $WHL --verbose

  # Build cuda_bindings wheel
  cd $SRC/cuda_bindings
  pip3 wheel . --no-deps --wheel-dir $WHL --verbose
else
  cd $SRC

  sed 's|^numpy.=.*|numpy|g' -i requirements.txt
  sed 's|^numba.=.*|numba|g' -i requirements.txt

  pip3 install -r requirements.txt
  python3 setup.py bdist_wheel --verbose --dist-dir $WHL
fi

cd /
rm -rf $SRC

pip3 install $WHL/cuda*.whl
twine upload --verbose $WHL/cuda*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"

python3 -c 'import cuda'
pip3 show cuda_core cuda_bindings || pip3 show cuda-python
