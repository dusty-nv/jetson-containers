#!/usr/bin/env bash
set -ex

# Clone the repository if it doesn't exist
git clone --branch=v${THREEGRUT_VERSION} --depth=1 --recursive https://github.com/nv-tlabs/3dgrut /opt/3dgrut || \
git clone --depth=1 --recursive https://github.com/nv-tlabs/3dgrut /opt/3dgrut

# Navigate to the directory containing PyMeshLab's setup.py
cd /opt/3dgrut

export MAX_JOBS="$(nproc)"
export CMAKE_BUILD_PARALLEL_LEVEL=$MAX_JOBS
echo "Building with MAX_JOBS=$MAX_JOBS and CMAKE_BUILD_PARALLEL_LEVEL=$CMAKE_BUILD_PARALLEL_LEVEL"

wget $WGET_FLAGS https://github.com/shader-slang/slang/releases/download/v2026.10.5/slang-2026.10.5-linux-aarch64.tar.gz && \
tar -xzvf slang*.tar.gz -C /usr/local
sed -i 's|\.type|.scalar_type|g' threedgrt_tracer/src/particlePrimitives.cu

# Set GCC-11 and G++-11 as the default
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 110 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 110

sed -i '/^--find-links/d' requirements.txt
sed -i 's/kaolin==0.17.0/kaolin>=0.17.0/' requirements.txt

pip install -r requirements.txt
pip install -e .

MAX_JOBS=$(nproc) \
pip3 wheel . -w /opt/3dgrut/wheels --verbose

cd /tmp/
wget $WGET_FLAGS https://es.download.nvidia.com/XFree86/aarch64/570.169/NVIDIA-Linux-aarch64-570.169.run
chmod +x NVIDIA-Linux-aarch64-570.169.run
sh NVIDIA-Linux-aarch64-570.169.run --extract-only
cd NVIDIA-Linux-aarch64-570.169/
cp -R ./libnvoptix.so.570.169 /usr/lib/aarch64-linux-gnu/
cp -R ./libnvidia-rtcore.so.570.169 /usr/lib/aarch64-linux-gnu/
cp -R ./nvoptix.bin /usr/lib/aarch64-linux-gnu/
ln -sf /usr/lib/aarch64-linux-gnu/libnvoptix.so.570.169 /usr/lib/aarch64-linux-gnu/libnvoptix.so.1
# Clean up
rm -rf /tmp/NVIDIA-Linux-aarch64-570.169.run /tmp/NVIDIA-Linux-aarch64-570.169
# pip3 install /opt/3dgrut/wheels/threedgrut-*.whl

cd /opt/3dgrut
# Optionally upload to a repository using Twine
twine upload --verbose /opt/3dgrut/wheels/threedgrut-*.whl || echo "Failed to upload wheel to ${TWINE_REPOSITORY_URL}"
