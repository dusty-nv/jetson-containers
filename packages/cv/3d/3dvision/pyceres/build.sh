#!/usr/bin/env bash
set -ex

# ----------------------------
# Config (override via env)
# ----------------------------
PYCERES_SRC="${PYCERES_SRC:-/opt/pyceres}"
PYCERES_VERSION="${PYCERES_VERSION:-2.6}"

CERES_VERSION="${CERES_VERSION:-2.2.0}"
CERES_PREFIX="${CERES_PREFIX:-/usr/local}"
CERES_BUILD_DIR="${CERES_BUILD_DIR:-/tmp/ceres-build}"
CERES_SRC_DIR="${CERES_SRC_DIR:-/tmp/ceres-solver}"

PIP_WHEEL_DIR="${PIP_WHEEL_DIR:-/opt/wheels}"

# ----------------------------
# Detect Ubuntu version
# ----------------------------
UBUNTU_VERSION=""
if [ -f /etc/os-release ]; then
  . /etc/os-release
  UBUNTU_VERSION="${VERSION_ID}"
fi

is_ubuntu_lt_2404() {
  python3 - <<PY
from packaging.version import Version
import sys
sys.exit(0 if Version("${UBUNTU_VERSION}") < Version("24.04") else 1)
PY
}

# ----------------------------
# Build Ceres from source ONLY if Ubuntu < 24.04
# ----------------------------
if command -v apt-get >/dev/null 2>&1 && is_ubuntu_lt_2404; then
  echo "Ubuntu ${UBUNTU_VERSION} detected (< 24.04) — building Ceres from source"

  apt-get update
  apt-get remove -y libceres-dev ceres-solver || true
  apt-get autoremove -y || true

  apt-get install -y --no-install-recommends \
    build-essential cmake ninja-build git pkg-config \
    libeigen3-dev libgoogle-glog-dev libgflags-dev \
    libsuitesparse-dev libatlas-base-dev

  rm -rf /var/lib/apt/lists/*

  rm -rf "${CERES_SRC_DIR}" "${CERES_BUILD_DIR}"
  git clone --depth 1 --branch "${CERES_VERSION}" \
    https://github.com/ceres-solver/ceres-solver.git "${CERES_SRC_DIR}"

  cmake -S "${CERES_SRC_DIR}" -B "${CERES_BUILD_DIR}" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${CERES_PREFIX}" \
    -DBUILD_TESTING=OFF \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_SHARED_LIBS=ON

  cmake --build "${CERES_BUILD_DIR}" -j"$(nproc)"
  cmake --install "${CERES_BUILD_DIR}"
  ldconfig || true
else
  echo "Ubuntu ${UBUNTU_VERSION} detected (>= 24.04) — using system Ceres"
fi

# Ensure CMake can find Ceres (both cases)
export CMAKE_PREFIX_PATH="${CERES_PREFIX}:${CMAKE_PREFIX_PATH:-}"
export Ceres_DIR="${CERES_PREFIX}/lib/cmake/Ceres"

# ----------------------------
# Clone pyceres (if needed)
# ----------------------------
if [ ! -d "${PYCERES_SRC}" ]; then
  echo "Cloning pyceres version ${PYCERES_VERSION}"
  git clone --branch="v${PYCERES_VERSION}" --depth=1 --recursive \
    https://github.com/cvg/pyceres "${PYCERES_SRC}" || \
  git clone --depth=1 --recursive \
    https://github.com/cvg/pyceres "${PYCERES_SRC}"
fi

cd "${PYCERES_SRC}"
export MAX_JOBS="$(nproc)"

# ----------------------------
# Build & install pyceres
# ----------------------------
uv build --wheel . --out-dir "${PIP_WHEEL_DIR}" --verbose
uv pip install "${PIP_WHEEL_DIR}"/pyceres*.whl

# ----------------------------
# Optional upload
# ----------------------------
twine upload --verbose "${PIP_WHEEL_DIR}"/pyceres*.whl || echo "Failed to upload wheel to ${TWINE_REPOSITORY_URL:-<unset>}"
