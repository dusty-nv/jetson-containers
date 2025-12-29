#!/usr/bin/env bash
set -euxo pipefail

# Required env:
#   PYCOLMAP_VERSION (e.g. 3.9)
#   CUDAARCHS (e.g. "86;87" or "87")
# Optional:
#   PYCOLMAP_SRC (default /opt/pycolmap)
#   STAGE_DIR     (default /tmp/colmap-stage)
#   PIP_WHEEL_DIR (default /tmp/wheels)

PYCOLMAP_SRC="${PYCOLMAP_SRC:-/opt/pycolmap}"
STAGE_DIR="${STAGE_DIR:-/tmp/colmap-stage}"
PIP_WHEEL_DIR="${PIP_WHEEL_DIR:-/tmp/wheels}"

# 1) Get source
if [ ! -d "$PYCOLMAP_SRC" ]; then
  echo "Cloning COLMAP ${PYCOLMAP_VERSION}"
  git clone --branch "v${PYCOLMAP_VERSION}" --depth=1 --recursive https://github.com/colmap/colmap "$PYCOLMAP_SRC" \
    || git clone --depth=1 --recursive https://github.com/colmap/colmap "$PYCOLMAP_SRC"
fi

mkdir -p "$PYCOLMAP_SRC/build"
cd "$PYCOLMAP_SRC/build"

# 2) Configure & build C++ targets
cmake \
  -DCUDA_ENABLED=ON \
  -DCMAKE_CUDA_ARCHITECTURES="${CUDAARCHS}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/usr/local \
  ..

cmake --build . -- -j"$(nproc)"

# 3) Stage install into a DESTDIR (no root required)
rm -rf "$STAGE_DIR"
mkdir -p "$STAGE_DIR"
# CMake honors DESTDIR for a "fake" root
DESTDIR="$STAGE_DIR" cmake --install .

# (Optional) also install to the current host
cmake --install .

# 4) Build Python wheel
cd "$PYCOLMAP_SRC"
mkdir -p "$PIP_WHEEL_DIR"
export MAX_JOBS="$(nproc)"
uv build --wheel . --out-dir "$PIP_WHEEL_DIR" --verbose

# (Optional) install locally for immediate use
uv pip install "$PIP_WHEEL_DIR"/pycolmap-*.whl

# 5) Prepare upload layout(s)
echo "Staged native files:"
ls -lahR "$STAGE_DIR/usr/local" | head -n 200 || true

tarpack upload "coolmap-${PYCOLMAP_VERSION}" "$STAGE_DIR/usr/local" || echo "failed to upload tarball for native artifacts"

# 6) Sanity checks (host)
ldconfig -p | grep -i colmap || true

echo "Done."

twine upload --verbose /opt/wheels/pycolmap*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
