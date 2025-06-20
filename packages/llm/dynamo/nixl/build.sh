#!/usr/bin/env bash
set -ex

# Clone the repository if it doesn't exist
git clone --branch=v${NIXL_VERSION} --depth=1 --recursive https://github.com/ai-dynamo/nixl /opt/nixl || \
git clone --depth=1 --recursive https://github.com/ai-dynamo/nixl /opt/nixl

cd /opt/

export MAX_JOBS=$(nproc)

git clone https://github.com/openucx/ucx.git && \
cd ucx && \
git checkout v1.19.x && \
./autogen.sh && ./configure \
   --enable-shared \
   --disable-static \
   --disable-doxygen-doc \
   --enable-optimizations \
   --enable-cma \
   --enable-devel-headers \
   --with-cuda=/usr/local/cuda \
   --with-verbs \
   --with-dm \
   --with-gdrcopy=/usr/local \
   --with-efa \
   --enable-mt && \
make -j && \
make -j install-strip && \
ldconfig

# Navigate to the directory containing nixl's setup.py
cd /opt/nixl

pip3 install --upgrade meson pybind11 patchelf

export MESON_ARGS="-Ddisable_mooncake_backend=false"
rm -rf build && \
mkdir build && \
meson setup build/ --prefix=/usr/local/nixl && \
cd build && \
ninja && \
ninja install

export NIXL_PREFIX=/usr/local/nixl

ARCH=$(uname -m)
echo "System architecture detected: ${ARCH}"
LIB_DIR="/usr/local/nixl/lib/${ARCH}-linux-gnu"
if [ -d "$LIB_DIR" ]; then

  echo "✅ Found NixL directory for ${ARCH}: ${LIB_DIR}"
  PLUGIN_DIR="${LIB_DIR}/plugins"
  export NIXL_PLUGIN_DIR="${PLUGIN_DIR}"
  echo "${LIB_DIR}" > /etc/ld.so.conf.d/nixl.conf
  echo "${PLUGIN_DIR}" >> /etc/ld.so.conf.d/nixl.conf
  ldconfig
  echo "NixL configuration for ${ARCH} applied successfully."
else
  echo "⚠️  Warning: NixL directory not found for architecture ${ARCH}."
  echo "   Checked path: ${LIB_DIR}"
  echo "   No configuration changes were made."
fi
ldconfig

cd /opt/nixl/src/bindings/rust && \
cargo build --release --locked
cd /opt/nixl/
pip3 wheel --wheel-dir=/opt/nixl/wheels . --verbose
pip3 install /opt/nixl/wheels/nixl*.whl

cd /opt/nixl

# Optionally upload to a repository using Twine
twine upload --verbose /opt/nixl/wheels/nixl*.whl || echo "Failed to upload wheel to ${TWINE_REPOSITORY_URL}"
