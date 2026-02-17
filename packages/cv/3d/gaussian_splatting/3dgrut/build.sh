#!/usr/bin/env bash
set -ex

# Clone the repository if it doesn't exist
git clone --branch=v${THREEGRUT_VERSION} --depth=1 --recursive https://github.com/nv-tlabs/3dgrut /opt/3dgrut || \
git clone --depth=1 --recursive https://github.com/nv-tlabs/3dgrut /opt/3dgrut

# Navigate to the directory containing PyMeshLab's setup.py
cd /opt/3dgrut

sed -i 's/python_requires=">=3\.11"/python_requires=">=3.10"/' /opt/3dgrut/setup.py

export MAX_JOBS="$(nproc)"
export CMAKE_BUILD_PARALLEL_LEVEL=$MAX_JOBS
echo "Building with MAX_JOBS=$MAX_JOBS and CMAKE_BUILD_PARALLEL_LEVEL=$CMAKE_BUILD_PARALLEL_LEVEL"

wget $WGET_FLAGS https://github.com/shader-slang/slang/releases/download/v2026.10.5/slang-2026.10.5-linux-aarch64.tar.gz && \
tar -xzvf slang*.tar.gz -C /usr/local
sed -i 's|\.type|.scalar_type|g' threedgrt_tracer/src/particlePrimitives.cu

# Set GCC-11 and G++-11 as the default
update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 110 && \
update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 110

sed -i '/^--find-links/d' requirements.txt
sed -i 's/kaolin==0.17.0/kaolin>=0.17.0/' requirements.txt
sed -i '/^opencv-python$/d' requirements.txt

uv pip install --no-build-isolation -r requirements.txt
uv pip install --no-build-isolation -e .

MAX_JOBS=$(nproc) \
uv build --wheel --no-build-isolation . --out-dir /opt/3dgrut/wheels --verbose
# uv pip install /opt/3dgrut/wheels/threedgrut-*.whl

cd /opt/3dgrut
# Optionally upload to a repository using Twine
twine upload --verbose /opt/3dgrut/wheels/threedgrut-*.whl || echo "Failed to upload wheel to ${TWINE_REPOSITORY_URL}"
uv pip install --force-reinstall opencv-contrib-python
