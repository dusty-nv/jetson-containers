#!/usr/bin/env bash
echo "Building opencv-python ${OPENCV_VERSION}"
set -ex
cd /opt

# install dependencies
bash $TMP/install_deps.sh

clone_with_fallback () {
  local ver="$1" url="$2" dir="$3"

  if git clone --branch "$ver" --recursive "$url" "$dir"; then
    echo 0        # se clonó la rama → HAREMOS checkout más tarde
  else
    git clone --recursive "$url" "$dir"
    echo 1        # se clonó main → NO se hace checkout
  fi
}

#------------------------------------------------------------------
# 1. Clonamos los tres repos y guardamos si necesitan checkout (0/1)
#------------------------------------------------------------------
cd /opt

need_ck_opencv=$(clone_with_fallback  "$OPENCV_VERSION" \
                 https://github.com/opencv/opencv            opencv)

need_ck_contrib=$(clone_with_fallback "$OPENCV_VERSION" \
                 https://github.com/opencv/opencv_contrib    opencv_contrib)

need_ck_py=$(clone_with_fallback      "$OPENCV_PYTHON" \
                 https://github.com/opencv/opencv-python     opencv-python)


if [ "$need_ck_opencv" -eq 0 ]; then
  cd opencv-python/opencv
  git checkout --recurse-submodules "$OPENCV_VERSION"
  cat modules/core/include/opencv2/core/version.hpp
  cd ../../
fi

if [ "$need_ck_contrib" -eq 0 ]; then
  cd opencv_contrib
  git checkout --recurse-submodules "$OPENCV_VERSION"
  cd ../
fi

if [ "$need_ck_contrib" -eq 0 ]; then
  cd opencv_extra
  git checkout --recurse-submodules "$OPENCV_VERSION"
  cd ../
fi

cd $TMP
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

# setup environment and build wheel
export CMAKE_BUILD_PARALLEL_LEVEL=$(nproc)
export CMAKE_POLICY_VERSION_MINIMUM="3.5"
export ENABLE_CONTRIB=1

CMAKE_ARGS="${OPENCV_BUILD_ARGS} -DOPENCV_EXTRA_MODULES_PATH=/opt/opencv-python/opencv_contrib/modules" \
pip3 wheel --wheel-dir=/opt --verbose .

ls /opt
cd /
rm -rf /opt/opencv-python

# install/test/upload wheel
pip3 install /opt/opencv*.whl
python3 -c "import cv2; print('OpenCV version:', str(cv2.__version__)); print(cv2.getBuildInformation())"
twine upload --verbose /opt/opencv*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"

# build C++ deb packages
mkdir /opt/opencv/build
cd /opt/opencv/build

cmake \
    ${OPENCV_BUILD_ARGS} \
    -DOPENCV_EXTRA_MODULES_PATH=/opt/opencv_contrib/modules \
    ../

make -j$(nproc)
make install
make package

# upload packages to apt server
mkdir -p /tmp/debs/
cp *.deb /tmp/debs/

tarpack upload OpenCV-${OPENCV_VERSION} /tmp/debs/ || echo "failed to upload tarball"
echo "installed" > "$TMP/.opencv"
