#!/usr/bin/env bash
set -ex
echo "Building opencv-python ${OPENCV_VERSION}"

bash /tmp/opencv/opencv_install_deps.sh

git clone --branch ${OPENCV_PYTHON} --recursive https://github.com/opencv/opencv-python /opt/opencv-python

cd /opt/opencv-python/opencv
git checkout --recurse-submodules ${OPENCV_VERSION}
cat modules/core/include/opencv2/core/version.hpp

cd ../opencv_contrib
git checkout --recurse-submodules ${OPENCV_VERSION}

cd ../opencv_extra
git checkout --recurse-submodules ${OPENCV_VERSION}

cd ../

# apply patches to setup.py
git apply /tmp/opencv/patches.diff || echo "failed to apply git patches"
git diff

# OpenCV looks for the cuDNN version in cudnn_version.h, but it's been renamed to cudnn_version_v8.h
ln -s /usr/include/$(uname -i)-linux-gnu/cudnn_version_v8.h /usr/include/$(uname -i)-linux-gnu/cudnn_version.h

# patches for FP16/half casts
sed -i 's|weight != 1.0|(float)weight != 1.0f|' opencv/modules/dnn/src/cuda4dnn/primitives/normalize_bbox.hpp
sed -i 's|nms_iou_threshold > 0|(float)nms_iou_threshold > 0.0f|' opencv/modules/dnn/src/cuda4dnn/primitives/region.hpp
grep 'weight' opencv/modules/dnn/src/cuda4dnn/primitives/normalize_bbox.hpp
grep 'nms_iou_threshold' opencv/modules/dnn/src/cuda4dnn/primitives/region.hpp
    
export ENABLE_CONTRIB=1
export CMAKE_BUILD_PARALLEL_LEVEL=$(nproc)
export CMAKE_ARGS="\
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
   -DENABLE_NEON=ON \
   -DOPENCV_DNN_CUDA=ON \
   -DOPENCV_ENABLE_NONFREE=ON \
   -DOPENCV_EXTRA_MODULES_PATH=/opt/opencv-python/opencv_contrib/modules \
   -DOPENCV_GENERATE_PKGCONFIG=ON \
   -DOpenGL_GL_PREFERENCE=GLVND \
   -DWITH_CUBLAS=ON \
   -DWITH_CUDA=ON \
   -DWITH_CUDNN=ON \
   -DWITH_GSTREAMER=ON \
   -DWITH_LIBV4L=ON \
   -DWITH_GTK=ON \
   -DWITH_OPENGL=OFF \
   -DWITH_OPENCL=OFF \
   -DWITH_IPP=OFF \
   -DWITH_TBB=ON \
   -DBUILD_TIFF=ON \
   -DBUILD_PERF_TESTS=OFF \
   -DBUILD_TESTS=OFF"

pip3 wheel --wheel-dir=/opt --verbose .

ls /opt
cd /
rm -rf /opt/opencv-python

pip3 install --no-cache-dir --verbose /opt/opencv*.whl
python3 -c "import cv2; print('OpenCV version:', str(cv2.__version__)); print(cv2.getBuildInformation())"
twine upload --verbose /opt/opencv*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
