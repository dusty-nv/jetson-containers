#!/usr/bin/env bash
set -ex

echo "Building IsaacSim ${ISAACSIM_VERSION}"

git clone --branch=${ISAACSIM_VERSION} --depth=1 --recursive https://github.com/isaac-sim/IsaacSim /opt/isaacsim  || \
git clone --depth=1 --recursive https://github.com/isaac-sim/IsaacSim /opt/isaacsim

apt-get update && apt-get install -y gcc-11 g++-11
update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 200
update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 200
cd /opt/isaacsim
git lfs install
git lfs pull
yes yes | ./build.sh
arch=$(uname -m)
case "$arch" in
  x86_64|amd64) platform="linux-x86_64" ;;
  aarch64|arm64) platform="linux-aarch64" ;;
  *) echo "Unsupported arch: $arch"; return 1 2>/dev/null || exit 1 ;;
esac
export OMNI_KIT_ALLOW_ROOT=1
# Isaac Sim root directory
export ISAACSIM_PATH="${PWD}/_build/linux-${arch}/release"
# Isaac Sim python executable
export ISAACSIM_PYTHON_EXE="${ISAACSIM_PATH}/python.sh"

${ISAACSIM_PYTHON_EXE} -c "print('Isaac Sim configuration is now complete.')"

uv pip install --force-reinstall torch torchvision torchaudio nvidia-cuda-nvrtc --index-url ${UV_DEFAULT_INDEX}
