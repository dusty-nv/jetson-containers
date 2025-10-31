#!/usr/bin/env bash
echo "Building OpenCV Python variants ${OPENCV_VERSION}"
set -ex
cd /opt

# install dependencies
bash $TMP/install_deps.sh


git clone --branch "${OPENCV_VERSION}" --recursive https://github.com/opencv/opencv \
  || git clone --recursive https://github.com/opencv/opencv

git clone --branch "${OPENCV_VERSION}" --recursive https://github.com/opencv/opencv_contrib \
  || git clone --recursive https://github.com/opencv/opencv_contrib

git clone --branch "${OPENCV_PYTHON}" --recursive https://github.com/opencv/opencv-python \
  || git clone --recursive https://github.com/opencv/opencv-python && export ENABLE_ROLLING=1

cd /opt/opencv-python/opencv || git checkout --recurse-submodules origin/4.x
git checkout --recurse-submodules ${OPENCV_VERSION} || git checkout --recurse-submodules origin/4.x
cat modules/core/include/opencv2/core/version.hpp
cd ../opencv_contrib
git checkout --recurse-submodules ${OPENCV_VERSION} || git checkout --recurse-submodules origin/4.x
cd ../opencv_extra
git checkout --recurse-submodules ${OPENCV_VERSION} || git checkout --recurse-submodules origin/4.x

cd ../

# apply patches to setup.py
git apply $TMP/patches.diff || echo "failed to apply git patches"
git diff

# OpenCV looks for the cuDNN version in cudnn_version.h, but it's been renamed to cudnn_version_v8.h
ln -sfnv /usr/include/$(uname -i)-linux-gnu/cudnn_version_v*.h /usr/include/$(uname -i)-linux-gnu/cudnn_version.h

# patches for FP16/half casts
function patch_opencv()
{
    sed -i 's|weight != 1.0|(float)weight != 1.0f|' opencv/modules/dnn/src/cuda4dnn/primitives/normalize_bbox.hpp
    sed -i 's|nms_iou_threshold > 0|(float)nms_iou_threshold > 0.0f|' opencv/modules/dnn/src/cuda4dnn/primitives/region.hpp
    grep 'weight' opencv/modules/dnn/src/cuda4dnn/primitives/normalize_bbox.hpp
    grep 'nms_iou_threshold' opencv/modules/dnn/src/cuda4dnn/primitives/region.hpp
}

patch_opencv
cd /opt
patch_opencv
cd /opt/opencv-python

# default build flags
OPENCV_BUILD_ARGS="\
   -DCPACK_BINARY_DEB=ON \
   -DBUILD_EXAMPLES=OFF \
   -DBUILD_opencv_python2=OFF \
   -DBUILD_opencv_python3=ON \
   -DBUILD_opencv_java=OFF \
   -DCMAKE_BUILD_TYPE=RELEASE \
   -DCMAKE_INSTALL_PREFIX=/usr/local \
   -DWITH_FFMPEG=ON \
   -DCUDA_ARCH_BIN=${CUDA_ARCH_BIN} \
   -DCUDA_ARCH_PTX= \
   -DCUDA_FAST_MATH=ON \
   -DCUDNN_INCLUDE_DIR=/usr/include/$(uname -i)-linux-gnu \
   -DEIGEN_INCLUDE_PATH=/usr/include/eigen3 \
   -DWITH_EIGEN=ON \
   -DOPENCV_DNN_CUDA=ON \
   -DOPENCV_ENABLE_NONFREE=ON \
   -DOPENCV_GENERATE_PKGCONFIG=ON \
   -DOpenGL_GL_PREFERENCE=GLVND \
   -DWITH_CUBLAS=ON \
   -DWITH_CUDA=ON \
   -DWITH_CUDNN=ON \
   -DWITH_GSTREAMER=ON \
   -DWITH_LIBV4L=ON \
   -DWITH_GTK=ON \
   -DWITH_OPENGL=ON \
   -DWITH_OPENCL=OFF \
   -DWITH_IPP=OFF \
   -DWITH_TBB=ON \
   -DBUILD_TIFF=ON \
   -DBUILD_PERF_TESTS=OFF \
   -DBUILD_TESTS=OFF"

# architecture-specific build flags
if [ "$(uname -m)" == "aarch64" ]; then
    OPENCV_BUILD_ARGS="${OPENCV_BUILD_ARGS} -DENABLE_NEON=ON"
fi

# cv2.abi3.so: undefined symbol: glRenderbufferStorageEXT
# https://github.com/opencv/opencv_contrib/issues/2307
OPENCV_BUILD_ARGS="${OPENCV_BUILD_ARGS} -DBUILD_opencv_rgbd=OFF"

# setup environment variables
export CMAKE_BUILD_PARALLEL_LEVEL=$(nproc)
export CMAKE_POLICY_VERSION_MINIMUM="3.5"
export CMAKE_LIBRARY_PATH=/usr/local/cuda/lib64/stubs
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
# export ENABLE_ROLLING=1 # Build from last commit
# export OPENCV_PYTHON_SKIP_GIT_COMMANDS=1

# Install dependencies for building the wheel
uv pip install scikit-build

# Function to build opencv-python-headless or opencv-python (simpler builds)
build_simple_variant() {
    local variant_name=$1
    local enable_headless=$2
    
    echo "=========================================="
    echo "Building ${variant_name}..."
    echo "=========================================="
    
    # Set environment variables
    export ENABLE_CONTRIB=0
    export ENABLE_HEADLESS=${enable_headless}
    
    # Create version.py file
    cat <<EOF > /opt/opencv-python/cv2/version.py
opencv_version = "${OPENCV_VERSION}"
contrib = False
headless = ${enable_headless}
rolling = False
EOF
    
    # Clean previous build artifacts
    cd /opt/opencv-python
    rm -rf build/ dist/ *.egg-info/ 2>/dev/null || true
    
    # Build the wheel (no contrib modules)
    CMAKE_ARGS="${OPENCV_BUILD_ARGS}" \
    uv build --wheel --out-dir /opt --verbose --no-build-isolation .
    
    echo "Completed building ${variant_name}"
    echo ""
}

# Build opencv-contrib-python with full working configuration
echo "=========================================="
echo "Building opencv-contrib-python (full configuration)..."
echo "=========================================="

export ENABLE_CONTRIB=1
export ENABLE_HEADLESS=0

cat <<EOF > /opt/opencv-python/cv2/version.py
opencv_version = "${OPENCV_VERSION}"
contrib = True
headless = False
rolling = False
EOF

CMAKE_ARGS="${OPENCV_BUILD_ARGS} -DOPENCV_EXTRA_MODULES_PATH=/opt/opencv-python/opencv_contrib/modules" \
uv build --wheel --out-dir /opt --verbose --no-build-isolation .

echo "Completed building opencv-contrib-python"
echo ""

# Build the other two variants (simpler builds without special patches/config)
build_simple_variant "opencv-python-headless" "1"
build_simple_variant "opencv-python" "0"

# List all generated wheels
echo "=========================================="
echo "All wheels built. Listing generated wheels:"
echo "=========================================="
ls -lh /opt/*.whl

# Install and test opencv-contrib-python wheel (the main working version)
echo "Installing and testing opencv-contrib-python..."
uv pip install /opt/opencv_contrib_python*.whl
python3 -c "import cv2; print('OpenCV version:', str(cv2.__version__)); print(cv2.getBuildInformation())"

# Upload all three variants
echo "Uploading all OpenCV Python wheels..."
# Upload each variant separately to avoid pattern overlap
twine upload --verbose /opt/opencv_python_headless*.whl || echo "failed to upload opencv-python-headless"
twine upload --verbose /opt/opencv_python-*.whl || echo "failed to upload opencv-python"
twine upload --verbose /opt/opencv_contrib_python*.whl || echo "failed to upload opencv-contrib-python"
echo "Upload completed for all variants"

# [FIX] Ensure the build directory is clean to avoid CMake caching issues from previous failed runs.
echo "Configuring C++ Debian package build..."
rm -rf /opt/opencv/build
mkdir /opt/opencv/build
cd /opt/opencv/build

# [FIX] Set the PKG_CONFIG_PATH environment variable.
# This is the crucial step that allows CMake to find system libraries like FFmpeg on Ubuntu.
# export PKG_CONFIG_PATH="/usr/lib/$(uname -i)-linux-gnu/pkgconfig:${PKG_CONFIG_PATH}"

# Now, running cmake will succeed because it can find the correct paths.
cmake \
    ${OPENCV_BUILD_ARGS} \
    -DOPENCV_EXTRA_MODULES_PATH=/opt/opencv_contrib/modules \
    ../

echo "Building C++ Debian packages..."
make -j$(nproc)
make install
make package

# upload packages to apt server
mkdir -p /tmp/debs/
cp *.deb /tmp/debs/

tarpack upload OpenCV-${OPENCV_VERSION} /tmp/debs/ || echo "failed to upload tarball"

# Cleanup
cd /
rm -rf /opt/opencv-python
echo "installed" > "$TMP/.opencv"
