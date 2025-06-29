#!/usr/bin/env bash
echo "Building cutlass $CUTLASS_VERSION"
set -ex

SRC=/opt/cutlass
WHL=/opt/wheels

export MAX_JOBS="$(nproc)"
export CMAKE_BUILD_PARALLEL_LEVEL=$MAX_JOBS
echo "Building with MAX_JOBS=$MAX_JOBS and CMAKE_BUILD_PARALLEL_LEVEL=$CMAKE_BUILD_PARALLEL_LEVEL"


git clone --branch v$CUTLASS_VERSION --depth=1 https://github.com/NVIDIA/cutlass $SRC

cd $SRC/python
pip3 wheel --no-deps --wheel-dir $WHL

cd /
rm -rf $SRC

pip3 install $WHL/cutlass*.whl
twine upload --verbose $WHL/cutlass*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"

python3 -c 'import cutlass; print(cutlass.__version__)'
pip3 show cutlass || pip3 show cutlass
