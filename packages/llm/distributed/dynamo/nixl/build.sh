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
if [[ "${IS_SBSA,,}" == "0" || "${IS_SBSA,,}" == "false" ]]; then
  sed -i 's|/usr/local/cuda/targets/x86_64-linux/|/usr/local/cuda/targets/aarch64-linux/|g' meson_options.txt
elif [[ "${IS_SBSA,,}" == "1" || "${IS_SBSA,,}" == "true" ]]; then
  sed -i 's|/usr/local/cuda/targets/x86_64-linux/|/usr/local/cuda/targets/sbsa-linux/|g' meson_options.txt
else
  echo "⚠️  IS_SBSA value '${IS_SBSA}' not recognized. No changes applied to meson_options.txt"
fi
export MESON_ARGS="-Ddisable_mooncake_backend=false"
rm -rf build && \
mkdir build && \
meson setup build/ --prefix=/usr/local/nixl && \
cd build && \
ninja && \
ninja install

export NIXL_PREFIX=/usr/local/nixl
export NIXL_PLUGIN_DIR=/usr/local/nixl/lib/aarch64-linux-gnu/plugins
echo "/usr/local/nixl/lib/aarch64-linux-gnu" > /etc/ld.so.conf.d/nixl.conf && \
echo "/usr/local/nixl/lib/aarch64-linux-gnu/plugins" >> /etc/ld.so.conf.d/nixl.conf && \
ldconfig

cd src/bindings/rust && cargo build --release --locked
cd /opt/nixl/
pip3 wheel --wheel-dir=/opt/nixl/wheels . --verbose
pip3 install /opt/nixl/wheels/nixl*.whl

cd /opt/nixl

# Optionally upload to a repository using Twine
twine upload --verbose /opt/nixl/wheels/nixl*.whl || echo "Failed to upload wheel to ${TWINE_REPOSITORY_URL}"
