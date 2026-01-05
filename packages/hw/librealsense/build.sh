#!/usr/bin/env bash
set -ex

git clone --branch=v${LIBREALSENSE_VERSION} --depth=1 --recursive https://github.com/IntelRealSense/librealsense /opt/librealsense  || \
git clone --depth=1 --recursive https://github.com/IntelRealSense/librealsense /opt/librealsense

cd /opt/librealsense

apt-get update && apt-get install -y gcc-11 g++-11
update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 200
update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 200

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
rm -rf librealsense

# make wrappers
cd /opt/librealsense/wrappers/python

ROOT=/opt/librealsense
BUILD_REL=$ROOT/build/release
WRAP_PY=$ROOT/wrappers/python

# 1) Locate the compiled extension
SO=$(ls "$BUILD_REL"/pyrealsense2.cpython-*.so | head -n1)
[ -f "$SO" ] || { echo "No .so in $BUILD_REL"; exit 1; }
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
VERSION=$(PYTHONPATH="$BUILD_REL" python3 -c "import pyrealsense2 as rs; print(getattr(rs,'__version__','unknown'))")
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
