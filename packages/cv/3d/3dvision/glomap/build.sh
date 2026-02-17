#!/usr/bin/env bash
set -euxo pipefail

# Requirements:
#   GLOMAP_VERSION (e.g. 0.5.0)
#   CUDAARCHS (e.g. "87" for Orin, or "86;87")
# Optional:
#   GLOMAP_SRC   (default /opt/glomap)
#   STAGE_DIR    (default /tmp/glomap-stage)

GLOMAP_SRC="${GLOMAP_SRC:-/opt/glomap}"
STAGE_DIR="${STAGE_DIR:-/tmp/glomap-stage}"

# 0) Minimal dependencies (adjust to your base system)
if command -v apt-get >/dev/null 2>&1; then
  apt-get update
  apt-get install -y --no-install-recommends \
    build-essential cmake ninja-build git pkg-config \
    libeigen3-dev
fi

# 1) Clone source
if [ ! -d "$GLOMAP_SRC" ]; then
  echo "Cloning glomap version ${GLOMAP_VERSION}"
  git clone --branch "v${GLOMAP_VERSION}" --depth=1 --recursive https://github.com/colmap/glomap "$GLOMAP_SRC" \
    || git clone --depth=1 --recursive https://github.com/colmap/glomap "$GLOMAP_SRC"
fi

cd "$GLOMAP_SRC"

# 2) Remove -Werror (known issue)
#   - In the main tree (before configuring)
sed -i 's|-Werror||g' glomap/CMakeLists.txt || true

# 3) Configure + build
rm -rf build
mkdir -p build
cd build

cmake .. -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  "-DCMAKE_CUDA_ARCHITECTURES=${CUDAARCHS}" \
  -DCMAKE_INSTALL_PREFIX=/usr/local

# After configuring, the poselib submodule is in _deps: remove -Werror there as well
sed -i 's|-Werror||g' _deps/poselib-src/CMakeLists.txt || true

ninja -j"$(nproc)"

# 4) Installation in STAGING via DESTDIR (does not touch the system)
rm -rf "$STAGE_DIR"
mkdir -p "$STAGE_DIR"
DESTDIR="$STAGE_DIR" ninja install

echo "Staged contents:"
ls -lahR "$STAGE_DIR/usr/local" | head -n 200 || true

# 5) (Optional) also install on the current host
ninja install -j $(nproc)

# 6) Upload staged /usr/local tree (bin/, lib/, include/, etc.) to tarpack
tarpack upload "glomap-${GLOMAP_VERSION}" "$STAGE_DIR/usr/local" \
  || echo "failed to upload tarball"

# 7) Light checks
ldconfig -p | grep -i glomap || true

echo "GLOMAP packaged with tarpack."
