#!/usr/bin/env bash
set -ex

# Clone the repository if it doesn't exist
git clone --branch=v${NIXL_VERSION} --depth=1 --recursive https://github.com/ai-dynamo/nixl /usr/local/nixl || \
git clone --depth=1 --recursive https://github.com/ai-dynamo/nixl /usr/local/nixl

# -----------------------------------------------------------------------------
cd /opt/
echo "Installing dependencies for NixL build..."
export MAX_JOBS=$(nproc)
ARCH=$(uname -m)
export CPATH=/usr/local/cuda/include:$CPATH
export LIBRARY_PATH=/usr/local/cuda/lib64:$LIBRARY_PATH



# -----------------------------------------------------------------------------
echo "ETCD INSTALLATION..."
# Navigate to the directory containing nixl's setup.py
cd /opt/
git clone https://github.com/etcd-cpp-apiv3/etcd-cpp-apiv3.git &&\
	cd etcd-cpp-apiv3 && mkdir build && cd build && \
	cmake .. && make -j$(nproc) && make install
ldconfig
echo "ETCD INSTALLATION COMPLETED"


# -----------------------------------------------------------------------------
echo "UCX INSTALLATION STARTED"
cd /usr/local/src
rm -rf /usr/lib/ucx
rm -rf /opt/hpcx/ucx
git clone https://github.com/openucx/ucx.git && \
cd ucx && \
git checkout v1.19.x && \
./autogen.sh && ./configure     \
    --enable-shared             \
    --disable-static            \
    --disable-doxygen-doc       \
    --enable-optimizations      \
    --enable-cma                \
    --enable-devel-headers      \
    --with-cuda=/usr/local/cuda \
    --with-verbs                \
    --with-dm                   \
    --with-gdrcopy=/usr/local   \
    --with-efa                  \
    --enable-mt &&              \
make -j &&                      \
make -j install-strip &&        \
ldconfig

echo "UCX INSTALLATION COMPLETED"

# -----------------------------------------------------------------------------
echo "Installing DPDK and DOCA SDK dependencies..."
cd /opt/
ARCH_SUFFIX=$(if [ "${ARCH}" = "aarch64" ]; then echo "arm64-sbsa"; else echo "${ARCH}"; fi) && \
export PKG_CONFIG_PATH="/opt/mellanox/doca/lib/${ARCH_SUFFIX}-linux-gnu/pkgconfig:/opt/mellanox/dpdk/lib/${ARCH_SUFFIX}-linux-gnu/pkgconfig:$PKG_CONFIG_PATH" && \
curl -fsSL https://linux.mellanox.com/public/repo/doca/3.0.0/ubuntu24.04/${ARCH_SUFFIX}/GPG-KEY-Mellanox.pub | \
gpg --dearmor | tee /usr/share/keyrings/mellanox-archive-keyring.gpg && \
echo "deb [signed-by=/usr/share/keyrings/mellanox-archive-keyring.gpg] https://linux.mellanox.com/public/repo/doca/3.0.0/ubuntu24.04/${ARCH_SUFFIX} ./" | \
tee /etc/apt/sources.list.d/mellanox.list && \
DEBIAN_FRONTEND=noninteractive apt update -y && \
apt install -y --no-install-recommends \
    mlnx-dpdk mlnx-dpdk-dev \
    doca-sdk-common doca-sdk-dma doca-sdk-dpdk-bridge \
    doca-sdk-eth doca-sdk-flow doca-sdk-rdma doca-all \
    doca-sdk-gpunetio libdoca-sdk-gpunetio-dev

echo "DPDK and DOCA SDK dependencies installed successfully."

# -----------------------------------------------------------------------------
echo "Installing Python dependencies..."
cd /usr/local/nixl
pip3 install --upgrade meson pybind11 patchelf auditwheel
# -----------------------------------------------------------------------------
echo "Building NixL..."
export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH
export CMAKE_PREFIX_PATH=/usr/local:$CMAKE_PREFIX_PATH
rm -rf build && \
mkdir build && \
meson setup build/ --prefix=/usr/local/nixl --buildtype=release && \
cd build && \
ninja && \
ninja install

# -----------------------------------------------------------------------------
echo "System architecture detected: ${ARCH}"
LIB_DIR="/usr/local/nixl/lib/${ARCH}-linux-gnu"
if [ -d "$LIB_DIR" ]; then

  echo "✅ Found NixL directory for ${ARCH}: ${LIB_DIR}"
  export PLUGIN_DIR="${LIB_DIR}/plugins"
  export NIXL_PLUGIN_DIR="/usr/local/nixl/lib/${ARCH}-linux-gnu/plugins"
  echo "/usr/local/nixl/lib/${ARCH}-linux-gnu" > /etc/ld.so.conf.d/nixl.conf && \
  echo "/usr/local/nixl/lib/${ARCH}-linux-gnu/plugins" >> /etc/ld.so.conf.d/nixl.conf && \
  ldconfig
  echo "NixL configuration for ${ARCH} applied successfully."
else
  echo "⚠️  Warning: NixL directory not found for architecture ${ARCH}."
  echo "   Checked path: ${LIB_DIR}"
  echo "   No configuration changes were made."
fi
ldconfig
# -----------------------------------------------------------------------------
cd /usr/local/nixl/
rm -rf /tmp/wheels
rm -rf /usr/local/nixl/wheels/
pip3 wheel --no-deps --wheel-dir=/tmp/wheels .
UNREPAIRED_WHEEL=$(find /tmp/wheels -name "nixl-*linux_aarch64.whl")
WHL_PLATFORM="manylinux_2_39_$(uname -m)"
pip3 install auditwheel
auditwheel repair --exclude libcuda.so.1 $UNREPAIRED_WHEEL --plat $WHL_PLATFORM --wheel-dir /usr/local/nixl/wheels/
/usr/local/nixl/contrib/wheel_add_ucx_plugins.py --ucx-lib-dir /usr/lib /usr/local/nixl/wheels/*.whl
pip3 install /usr/local/nixl/wheels/nixl-*.whl

cd /usr/local/nixl/

# Optionally upload to a repository using Twine
twine upload --verbose /usr/local/nixl/wheels/nixl*.whl || echo "Failed to upload wheel to ${TWINE_REPOSITORY_URL}"
