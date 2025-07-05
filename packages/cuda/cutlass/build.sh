#!/usr/bin/env bash
echo "Building cutlass $CUTLASS_VERSION"
set -ex

SRC=/opt/cutlass
WHL=/opt/wheels

export MAX_JOBS="$(nproc)"
export CMAKE_BUILD_PARALLEL_LEVEL=$MAX_JOBS
echo "Building with MAX_JOBS=$MAX_JOBS and CMAKE_BUILD_PARALLEL_LEVEL=$CMAKE_BUILD_PARALLEL_LEVEL"


git clone --branch v$CUTLASS_VERSION --depth=1 https://github.com/NVIDIA/cutlass $SRC || \
git clone --depth=1 https://github.com/NVIDIA/cutlass $SRC

cd $SRC
echo "Building cutlass wheel in $SRC"
pip3 wheel . -w $WHL

cd $SRC/python

python3 setup_library.py bdist_wheel --dist-dir $WHL
python3 setup_pycute.py bdist_wheel --dist-dir $WHL

cd $SRC

pip3 install $WHL/nvidia_cutlass*.whl $WHL/pycute*.whl
twine upload --verbose $WHL/nvidia_cutlass*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
twine upload --verbose $WHL/pycute*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"

python3 -c 'import cutlass; print(cutlass.__version__)'
