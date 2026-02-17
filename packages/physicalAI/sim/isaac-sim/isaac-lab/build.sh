#!/usr/bin/env bash
set -ex

echo "Building IsaacLab ${ISAACLAB_VERSION}"

git clone --branch=${ISAACLAB_VERSION} --depth=1 --recursive https://github.com/isaac-sim/IsaacLab /opt/IsaacLab  || \
git clone --depth=1 --recursive https://github.com/isaac-sim/IsaacLab /opt/IsaacLab

export TERM=xterm-256color
arch="$(uname -m)"
case "$arch" in
  x86_64|amd64) isaac_arch="x86_64" ;;
  aarch64|arm64) isaac_arch="aarch64" ;;
  *) echo "Unsupported arch: $arch"; exit 1 ;;
esac

export ISAACSIM_PATH="/opt/isaacsim/_build/linux-${isaac_arch}/release"
echo "ISAACSIM_PATH=$ISAACSIM_PATH"
# sanity checks
test -x "${ISAACSIM_PATH}/python.sh" || { echo "python.sh not found in ${ISAACSIM_PATH}"; exit 1; }

# the key: link INSIDE the IsaacLab repo
ln -sfn "${ISAACSIM_PATH}" /opt/IsaacLab/_isaac_sim
ls -l /opt/IsaacLab/_isaac_sim/python.sh

cd /opt/IsaacLab
chmod a+x isaaclab.sh
./isaaclab.sh --install # or "./isaaclab.sh -i"

uv pip install --force-reinstall torch torchvision torchaudio nvidia-cuda-nvrtc --index-url ${UV_DEFAULT_INDEX}
