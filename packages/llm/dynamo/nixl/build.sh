#!/usr/bin/env bash
# Build NIXL from source, UCX (with CUDA/GDRCopy), and produce a manylinux wheel.
# Works on aarch64 (Jetson/ARM) and x86_64. Requires CUDA already installed.

set -euo pipefail
IFS=$'\n\t'
set -x

# ------------------------------ Config ------------------------------
: "${NIXL_VERSION:=0.6.0}"             # or pass in environment
: "${PYTHON_BIN:=python3}"             # venv python is fine
: "${CUDA_HOME:=/usr/local/cuda}"
: "${UCX_TAG:=v1.19.x}"               # UCX branch/tag
: "${INSTALL_PREFIX:=/usr/local}"     # where UCX will install
: "${NIXL_PREFIX:=/usr/local/nixl}"   # where NIXL installs
: "${TWINE_REPOSITORY_URL:=}"         # optional
ARCH="$(uname -m)"
MAX_JOBS="$(nproc)"

export CPATH="${CUDA_HOME}/include:${CPATH:-}"
export LIBRARY_PATH="${CUDA_HOME}/lib64:${LIBRARY_PATH:-}"
export PKG_CONFIG_PATH="${INSTALL_PREFIX}/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
export CMAKE_PREFIX_PATH="${INSTALL_PREFIX}:${CMAKE_PREFIX_PATH:-}"

# GDRCopy (adjust if different)
# We keep your legacy layout, but make it explicit:
: "${GDRCOPY_PREFIX:=/usr/local}"
mkdir -p "${GDRCOPY_PREFIX}/lib" || true
# If libgdrapi is under ${GDRCOPY_PREFIX}/lib/aarch64-linux-gnu, symlink a flat one:
if compgen -G "${GDRCOPY_PREFIX}/lib/aarch64-linux-gnu/libgdrapi.so*" > /dev/null; then
  ln -sf ${GDRCOPY_PREFIX}/lib/aarch64-linux-gnu/libgdrapi.so* "${GDRCOPY_PREFIX}/lib/" || true
fi
ldconfig

# Runtime search
export LD_LIBRARY_PATH="${INSTALL_PREFIX}/lib/ucx:${INSTALL_PREFIX}/lib:${CUDA_HOME}/lib64:${GDRCOPY_PREFIX}/lib:${GDRCOPY_PREFIX}/lib/aarch64-linux-gnu:${LD_LIBRARY_PATH:-}"

# -------------------------- Clone NIXL repo --------------------------
if [[ ! -d "${NIXL_PREFIX}" ]]; then
  git clone --branch "v${NIXL_VERSION}" --depth=1 --recursive https://github.com/ai-dynamo/nixl "${NIXL_PREFIX}" || \
  git clone --depth=1 --recursive https://github.com/ai-dynamo/nixl "${NIXL_PREFIX}"
fi

# -------------------------- Optional: ETCD ---------------------------
# If you want etcd-cpp-apiv3 support, build it; otherwise skip this block.
if [[ ! -f "${INSTALL_PREFIX}/lib/libetcd-cpp-api.so" && ! -f "${INSTALL_PREFIX}/lib64/libetcd-cpp-api.so" ]]; then
  pushd /opt
  if [[ ! -d etcd-cpp-apiv3 ]]; then
    git clone https://github.com/etcd-cpp-apiv3/etcd-cpp-apiv3.git
  fi
  cd etcd-cpp-apiv3
  rm -rf build && mkdir build && cd build
  cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}"
  make -j"${MAX_JOBS}"
  make install
  ldconfig
  popd
fi

# ------------------------------ UCX -------------------------------
# Build UCX with CUDA + GDRCopy so GPU device API is available
pushd /usr/local/src
rm -rf ucx
git clone https://github.com/openucx/ucx.git
cd ucx
git checkout "${UCX_TAG}"
./autogen.sh

if [ -f "./contrib/configure-release-mt" ]; then
    ./contrib/configure-release-mt \
        --prefix="${INSTALL_PREFIX}" \
        --enable-shared \
        --disable-static \
        --disable-doxygen-doc \
        --enable-optimizations \
        --enable-cma \
        --enable-devel-headers \
        --with-cuda="${CUDA_HOME}" \
        --with-verbs \
        --with-dm \
        --with-gdrcopy="${GDRCOPY_PREFIX}" \
        --enable-mt
else
    CPPFLAGS="-I${GDRCOPY_PREFIX}/include" \
    LDFLAGS="-L${GDRCOPY_PREFIX}/lib -L${GDRCOPY_PREFIX}/lib/aarch64-linux-gnu" \
    LIBS="-lgdrapi -lcuda" \
    ./configure \
      --prefix="${INSTALL_PREFIX}" \
      --enable-shared \
      --disable-static \
      --disable-doxygen-doc \
      --enable-optimizations \
      --enable-cma \
      --enable-devel-headers \
      --with-cuda="${CUDA_HOME}" \
      --with-verbs \
      --with-dm \
      --with-gdrcopy="${GDRCOPY_PREFIX}" \
      --enable-mt
fi

make -j"$(nproc)" || make -j"$(nproc)"
make -j install-strip || make -j install
ldconfig
popd

# Quick UCX sanity: expect to see cuda/gdr entries
ucx_info -v | egrep -i 'cuda|gdr|gdrcopy' || true
ucx_info -d | egrep -i 'cuda_(copy|ipc)|gdr' || true


# -------------------------- Build NIXL (meson) -----------------------
pushd "${NIXL_PREFIX}"
# toolchain deps
${PYTHON_BIN} -m pip install -U tomlkit meson ninja pybind11 patchelf auditwheel

# Check CUDA version and rename wheel if CUDA 13
if [ -x "${CUDA_HOME}/bin/nvcc" ]; then
    CUDA_MAJOR_VERSION=$("${CUDA_HOME}/bin/nvcc" --version | grep release | sed 's/.*release //' | cut -d. -f1)
    if [ "$CUDA_MAJOR_VERSION" == "13" ]; then
        echo "Detected CUDA 13, renaming wheel to nixl-cu13"
        if [ -f "./contrib/tomlutil.py" ]; then
             ./contrib/tomlutil.py --wheel-name nixl-cu13 pyproject.toml
        else
             echo "Warning: ./contrib/tomlutil.py not found, skipping wheel rename"
        fi
    fi
fi

rm -rf build
meson setup build --prefix="${NIXL_PREFIX}" --buildtype=release
meson compile -C build -j "${MAX_JOBS}"
meson install -C build

# Ensure system can see the installed libs
LIB_DIR="${NIXL_PREFIX}/lib/${ARCH}-linux-gnu"
PLUGINS_DIR="${LIB_DIR}/plugins"
echo "${LIB_DIR}"            | tee    /etc/ld.so.conf.d/nixl.conf >/dev/null
echo "${PLUGINS_DIR}"        | tee -a /etc/ld.so.conf.d/nixl.conf >/dev/null
ldconfig
popd

# ----------------------- Build & Repair Wheel -----------------------
# Prefer the official helper script; it handles auditwheel & UCX/NIXL plugins.
pushd "${NIXL_PREFIX}"

# Use uv if present, else fallback to pip/auditwheel manually
if command -v uv &>/dev/null; then
  # Official script expects UCX plugins in /usr/lib64/ucx by default.
  # On your box UCX installs to /usr/local/lib/ucx — pass it explicitly.
  OUTDIR="/opt/nixl/dist"
  mkdir -p "${OUTDIR}" || true
  sed -i 's/\r$//' "${NIXL_PREFIX}"/contrib/build-wheel.sh "${NIXL_PREFIX}"/contrib/wheel_add_ucx_plugins.py || true
  chmod +x "${NIXL_PREFIX}"/contrib/build-wheel.sh "${NIXL_PREFIX}"/contrib/wheel_add_ucx_plugins.py
  cd "${NIXL_PREFIX}/"
  ./contrib/build-wheel.sh \
    --python-version "$(${PYTHON_BIN} -c 'import sys;print(f"{sys.version_info.major}.{sys.version_info.minor}")')" \
    --platform "manylinux_2_39_${ARCH}" \
    --output-dir "${OUTDIR}" \
    --ucx-plugins-dir "${INSTALL_PREFIX}/lib/ucx" \
    --nixl-plugins-dir "${PLUGINS_DIR}"
  WHEEL_PATH="$(ls -1 ${OUTDIR}/nixl*.whl | head -n1)"
else
  # Fallback path (no uv)
  rm -rf /tmp/wheels "${NIXL_PREFIX}/wheels"
  ${PYTHON_BIN} -m pip wheel --no-deps --out-dir /tmp/wheels .
  UNREPAIRED_WHEEL="$(ls -1 /tmp/wheels/nixl*-"linux_${ARCH}.whl" 2>/dev/null || true)"
  if [[ -z "${UNREPAIRED_WHEEL}" ]]; then
    # newer pip tags may already be manylinux; just pick the first
    UNREPAIRED_WHEEL="$(ls -1 /tmp/wheels/nixl*.whl | head -n1)"
  fi
  WHL_PLATFORM="manylinux_2_39_${ARCH}"
  ${PYTHON_BIN} -m pip install -U auditwheel patchelf
  auditwheel repair \
    --exclude 'libcuda*' \
    --exclude 'libcufile*' \
    --exclude 'libssl*' \
    --exclude 'libcrypto*' \
    "${UNREPAIRED_WHEEL}" \
    --plat "${WHL_PLATFORM}" \
    --wheel-dir "${NIXL_PREFIX}/wheels"

  # CRITICAL: use UCX plugin dir from the UCX install, NOT /usr/lib
  "${NIXL_PREFIX}/contrib/wheel_add_ucx_plugins.py" \
    --ucx-plugins-dir "${INSTALL_PREFIX}/lib/ucx" \
    --nixl-plugins-dir "${PLUGINS_DIR}" \
    "${NIXL_PREFIX}/wheels/"*.whl

  WHEEL_PATH="$(ls -1 ${NIXL_PREFIX}/wheels/nixl*.whl | head -n1)"
fi

# Install locally (optional)
${PYTHON_BIN} -m pip install -U "${WHEEL_PATH}"

# Optional: upload
if [[ -n "${TWINE_REPOSITORY_URL}" ]]; then
  ${PYTHON_BIN} -m pip install -U twine
  twine upload --verbose "${WHEEL_PATH}" || echo "⚠️  Failed to upload wheel to ${TWINE_REPOSITORY_URL}"
fi

popd

echo "✅ Done. Built & repaired wheel at: ${WHEEL_PATH}"
