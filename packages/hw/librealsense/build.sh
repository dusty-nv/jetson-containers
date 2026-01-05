#!/usr/bin/env bash
set -ex

git clone --branch=v${LIBREALSENSE_VERSION} --depth=1 --recursive https://github.com/IntelRealSense/librealsense /opt/librealsense  || \
git clone --depth=1 --recursive https://github.com/IntelRealSense/librealsense /opt/librealsense

# Version comparison function: returns 0 if version1 >= version2, 1 otherwise
version_ge() {
    local v1="$1"
    local v2="$2"
    # Use sort -V (version sort) to compare: if v2 is the first (smallest), then v1 >= v2
    local first=$(printf '%s\n%s\n' "$v1" "$v2" | sort -V | head -n1)
    [ "$first" = "$v2" ] || [ "$v1" = "$v2" ]
}

cd /opt/librealsense

# --- 1. Install GCC 11 (Required for v2.50.0 on Ubuntu 24.04) ---
apt-get update && apt-get install -y gcc-11 g++-11
update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 200
update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 200

if version_ge "$LIBREALSENSE_VERSION" "2.50.0"; then

    # --- 2. Patch Pybind11 to v2.10.4 (Fixes Python 3.12 compatibility) ---
    sed -i 's/GIT_TAG "8de7772cc72daca8e947b79b83fea46214931604"/GIT_TAG "v2.10.4"/g' third-party/pybind11/CMakeLists.txt || true

    # --- 3. Fix missing <iostream> for std::cerr (Fixes GCC 11 build error) ---
    sed -i '10i #include <iostream>' wrappers/python/pyrs_device.cpp || true

    # --- 4. Fix undefined behavior: delete[] void* (Fixes warnings/errors) ---
    sed -i 's/delete\[\] ptr/delete[] (char*)ptr/g' wrappers/python/pyrs_internal.cpp || true

    # --- 5. Force C++14 (Required by Pybind11 v2.10+) ---
    sed -i 's/set(CMAKE_CXX_STANDARD 11)/set(CMAKE_CXX_STANDARD 14)/g' CMakeLists.txt || true
fi

export MAX_JOBS="$(nproc)"
mkdir build
cd build
cmake \
-DBUILD_EXAMPLES=ON \
-DFORCE_RSUSB_BACKEND=ON \
-DBUILD_WITH_CUDA=ON \
-DCMAKE_BUILD_TYPE=release \
-DBUILD_SHARED_LIBS=OFF \
-DBUILD_PYTHON_BINDINGS=ON \
-DPYTHON_EXECUTABLE=$(which python3) \
../
# -DPYTHON_INSTALL_DIR=$(python3 -c 'import sys; print(f"/usr/lib/python{sys.version_info.major}.{sys.version_info.minor}/dist-packages")') \
cmake --build . -j"$(($(nproc)-1))"
make install
cd ../
cp ./config/99-realsense-libusb.rules /etc/udev/rules.d/

# make wrappers
cd /opt/librealsense/wrappers/python

ROOT=/opt/librealsense
BUILD_REL=$ROOT/build/release
WRAP_PY=$ROOT/wrappers/python

# Wrapper setup for Python bindings
# 1) Locate the compiled extension
# For version 2.50.0, the .so is in build/wrappers/python
# For versions > 2.50.0, the .so is in build/release/
if [ "$LIBREALSENSE_VERSION" = "2.50.0" ]; then
    SO=$(ls "$ROOT/build/wrappers/python"/pyrealsense2.cpython-*.so | head -n1)
    [ -f "$SO" ] || { echo "No .so in $ROOT/build/wrappers/python"; exit 1; }
else
    SO=$(ls "$BUILD_REL"/pyrealsense2.cpython-*.so | head -n1)
    [ -f "$SO" ] || { echo "No .so in $BUILD_REL"; exit 1; }
fi
echo "Using $SO"

# 2) Recreate the package folder
PKG_DIR="$WRAP_PY/pyrealsense2"
rm -rf "$PKG_DIR"
mkdir -p "$PKG_DIR"

# 3) COPY the .so WITHOUT RENAMING
cp "$SO" "$PKG_DIR/"

# 4) __init__.py that re-exports the compiled submodule and exposes __version__
cat > "$PKG_DIR/__init__.py" << 'EOF'
# Re-export everything from the compiled extension that sits next to this file
from .pyrealsense2 import *  # type: ignore[attr-defined]

# Try to provide a __version__ from our generated file; fall back if needed
try:
    from ._version import __version__
except Exception:  # pragma: no cover
    try:
        from .pyrealsense2 import __version__  # if the extension ever exposes it
    except Exception:
        __version__ = "unknown"
EOF

# 5) Generate _version.py from the already-built module (run it from build dir)
# For version 2.50.0, set PYTHONPATH to the build directory containing the .so
# For versions > 2.50.0, use BUILD_REL which is build/release
if [ "$LIBREALSENSE_VERSION" = "2.50.0" ]; then
    VERSION_PATH="$ROOT/build/wrappers/python"
else
    VERSION_PATH="$BUILD_REL"
fi

VERSION=$(PYTHONPATH="$VERSION_PATH" python3 -c "import pyrealsense2 as rs; print(getattr(rs,'__version__','unknown'))" 2>/dev/null || echo "unknown")
# If version detection failed, fall back to LIBREALSENSE_VERSION
if [ "$VERSION" = "unknown" ] || [ -z "$VERSION" ]; then
    VERSION="$LIBREALSENSE_VERSION"
    echo "Warning: Could not detect version from module, using LIBREALSENSE_VERSION: $VERSION"
fi
printf '__version__ = "%s"\n' "$VERSION" > "$PKG_DIR/_version.py"
echo "Detected version: $VERSION"

# 6) Build the wheel using wrappers/python/pyproject.toml
python3 -m pip install -U pip build hatchling
cd "$WRAP_PY"
python3 -m build --wheel --outdir "$ROOT/wheels"

# 7) Install and verify
pip install --force-reinstall /opt/librealsense/wheels/pyrealsense2*.whl

# Optionally upload to a repository using Twine
twine upload --verbose /opt/librealsense/wheels/pyrealsense2*.whl || echo "Failed to upload wheel to ${TWINE_REPOSITORY_URL}"
