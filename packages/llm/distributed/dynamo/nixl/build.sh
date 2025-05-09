#!/usr/bin/env bash
set -ex

# Clone the repository if it doesn't exist
git clone --branch=v${NIXL_VERSION} --depth=1 --recursive https://github.com/ai-dynamo/nixl /opt/nixl || \
git clone --depth=1 --recursive https://github.com/ai-dynamo/nixl /opt/nixl

cd /opt/

export MAX_JOBS=$(nproc)

wget https://github.com/openucx/ucx/releases/download/v1.18.1/ucx-1.18.1.tar.gz
tar xzf ucx-1.18.1.tar.gz
cd ucx-1.18.1
./configure \
  --enable-shared \
  --disable-static \
  --disable-doxygen-doc \
  --enable-optimizations \
  --enable-cma \
  --enable-devel-headers \
  --with-verbs \
  --with-dm \
  --enable-mt

# --with-gdrcopy=
# --with-cuda=<cuda install>
make -j
make -j install-strip
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
pip3 wheel --wheel-dir=/opt/nixl/wheels . --verbose
pip3 install /opt/nixl/wheels/nixl*.whl

cd /opt/nixl

# Optionally upload to a repository using Twine
twine upload --verbose /opt/nixl/wheels/nixl*.whl || echo "Failed to upload wheel to ${TWINE_REPOSITORY_URL}"
