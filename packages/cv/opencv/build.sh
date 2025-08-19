#!/usr/bin/env bash
echo "Building opencv-python ${OPENCV_VERSION}"
set -ex
cd /opt

# ------------------------------------------------------------------------------
# 1) Dependencias base
# ------------------------------------------------------------------------------
bash $TMP/install_deps.sh

# ------------------------------------------------------------------------------
# 2) Clonar repos necesarios
#    Nota: añadimos opencv_extra (no estaba en tu script original)
# ------------------------------------------------------------------------------
git clone --branch "${OPENCV_VERSION}" --recursive https://github.com/opencv/opencv \
  || git clone --recursive https://github.com/opencv/opencv

git clone --branch "${OPENCV_VERSION}" --recursive https://github.com/opencv/opencv_contrib \
  || git clone --recursive https://github.com/opencv/opencv_contrib

git clone --branch "${OPENCV_PYTHON}" --recursive https://github.com/opencv/opencv-python \
  || git clone --recursive https://github.com/opencv/opencv-python

git clone --branch "${OPENCV_VERSION}" --recursive https://github.com/opencv/opencv_extra \
  || git clone --recursive https://github.com/opencv/opencv_extra

# Forzar las ramas solicitadas (si existen) o caer a 4.x
cd /opt/opencv-python/opencv || git checkout --recurse-submodules origin/4.x
git checkout --recurse-submodules ${OPENCV_VERSION} || git checkout --recurse-submodules origin/4.x
cat modules/core/include/opencv2/core/version.hpp || true

cd /opt/opencv-python/opencv_contrib
git checkout --recurse-submodules ${OPENCV_VERSION} || git checkout --recurse-submodules origin/4.x

cd /opt/opencv-python/opencv_extra
git checkout --recurse-submodules ${OPENCV_VERSION} || git checkout --recurse-submodules origin/4.x

cd /opt

# ------------------------------------------------------------------------------
# 3) Parcheo menor dnn (mantiene tus cambios originales)
# ------------------------------------------------------------------------------
function patch_opencv_dnn_fp16()
{
    sed -i 's|weight != 1.0|(float)weight != 1.0f|' opencv/modules/dnn/src/cuda4dnn/primitives/normalize_bbox.hpp || true
    sed -i 's|nms_iou_threshold > 0|(float)nms_iou_threshold > 0.0f|' opencv/modules/dnn/src/cuda4dnn/primitives/region.hpp || true
    grep 'weight' opencv/modules/dnn/src/cuda4dnn/primitives/normalize_bbox.hpp || true
    grep 'nms_iou_threshold' opencv/modules/dnn/src/cuda4dnn/primitives/region.hpp || true
}
# aplicar tanto al opencv de opencv-python como al repo "opencv" de /opt
pushd /opt/opencv-python; patch_opencv_dnn_fp16; popd
pushd /opt;             patch_opencv_dnn_fp16; popd

# ------------------------------------------------------------------------------
# 4) Symlink para cudnn (mantiene tu workaround)
# ------------------------------------------------------------------------------
ln -sfnv /usr/include/$(uname -i)-linux-gnu/cudnn_version_v*.h /usr/include/$(uname -i)-linux-gnu/cudnn_version.h || true

# ------------------------------------------------------------------------------
# 5) Parche CUDA 13 para cuda_info.cpp (clockRate, kernelExecTimeoutEnabled, etc)
#    -> Se aplica en AMBOS árboles: /opt/opencv-python/opencv y /opt/opencv
# ------------------------------------------------------------------------------
cat > /tmp/opencv-cuda13.patch <<'PATCH'
*** a/modules/core/src/cuda_info.cpp
--- b/modules/core/src/cuda_info.cpp
***************
*** 420,436 ****
     int cv::cuda::DeviceInfo::clockRate() const
     {
-        return deviceProps().get(device_id_)->clockRate;
+    #if CUDART_VERSION >= 13000
+        int v = 0; cudaDeviceGetAttribute(&v, cudaDevAttrClockRate, device_id_);
+        return v; // kHz
+    #else
+        return deviceProps().get(device_id_)->clockRate;
+    #endif
     }
***************
*** 483,498 ****
     bool cv::cuda::DeviceInfo::kernelExecTimeoutEnabled() const
     {
-        return deviceProps().get(device_id_)->kernelExecTimeoutEnabled != 0;
+    #if CUDART_VERSION >= 13000
+        int v = 0; cudaDeviceGetAttribute(&v, cudaDevAttrKernelExecTimeout, device_id_);
+        return v != 0;
+    #else
+        return deviceProps().get(device_id_)->kernelExecTimeoutEnabled != 0;
+    #endif
     }
***************
*** 518,532 ****
     cv::cuda::DeviceInfo::ComputeMode cv::cuda::DeviceInfo::computeMode() const
     {
-        static ComputeMode tbl[] = { Default, Exclusive, Prohibited, ExclusiveProcess };
-        return tbl[deviceProps().get(device_id_)->computeMode];
+    #if CUDART_VERSION >= 13000
+        int v = 0; cudaDeviceGetAttribute(&v, cudaDevAttrComputeMode, device_id_);
+        static ComputeMode tbl[] = { Default, Exclusive, Prohibited, ExclusiveProcess };
+        return tbl[v];
+    #else
+        static ComputeMode tbl[] = { Default, Exclusive, Prohibited, ExclusiveProcess };
+        return tbl[deviceProps().get(device_id_)->computeMode];
+    #endif
     }
***************
*** 552,563 ****
     int cv::cuda::DeviceInfo::maxTexture1DLinear() const
     {
-        return deviceProps().get(device_id_)->maxTexture1DLinear;
+    #if CUDART_VERSION >= 13000
+    #ifdef cudaDevAttrMaxTexture1DLinear
+        int v = 0; cudaDeviceGetAttribute(&v, cudaDevAttrMaxTexture1DLinear, device_id_);
+        return v;
+    #else
+        return 0; // atributo removido: no disponible
+    #endif
+    #else
+        return deviceProps().get(device_id_)->maxTexture1DLinear;
+    #endif
     }
***************
*** 789,800 ****
     int cv::cuda::DeviceInfo::memoryClockRate() const
     {
-        return deviceProps().get(device_id_)->memoryClockRate;
+    #if CUDART_VERSION >= 13000
+        int v = 0; cudaDeviceGetAttribute(&v, cudaDevAttrMemoryClockRate, device_id_);
+        return v; // kHz
+    #else
+        return deviceProps().get(device_id_)->memoryClockRate;
+    #endif
     }
***************
*** 930,977 ****
     void cv::cuda::printCudaDeviceInfo(int device)
     {
         const cudaDeviceProp& prop = *deviceProps().get(device);
-        printf("  GPU Clock Speed:                               %.2f GHz\n", prop.clockRate * 1e-6f);
-        printf("  Concurrent copy and execution:                 %s with %d copy engine(s)\n", (prop.deviceOverlap ? "Yes" : "No"), prop.asyncEngineCount);
-        printf("  Run time limit on kernels:                     %s\n", prop.kernelExecTimeoutEnabled ? "Yes" : "No");
-        printf("      %s \n", computeMode[prop.computeMode]);
+    #if CUDART_VERSION >= 13000
+        int clockKHz=0, timeout=0, cm=0, concurrent=0, asyncEngines=0;
+        cudaDeviceGetAttribute(&clockKHz,  cudaDevAttrClockRate,           device);
+        cudaDeviceGetAttribute(&timeout,   cudaDevAttrKernelExecTimeout,   device);
+        cudaDeviceGetAttribute(&cm,        cudaDevAttrComputeMode,         device);
+        cudaDeviceGetAttribute(&concurrent,cudaDevAttrConcurrentKernels,   device);
+    #ifdef cudaDevAttrAsyncEngineCount
+        cudaDeviceGetAttribute(&asyncEngines, cudaDevAttrAsyncEngineCount, device);
+    #endif
+        printf("  GPU Clock Speed:                               %.2f GHz\n", clockKHz * 1e-6f);
+        printf("  Concurrent copy and execution:                 %s with %d copy engine(s)\n",
+               (concurrent ? "Yes" : "No"), asyncEngines);
+        printf("  Run time limit on kernels:                     %s\n", timeout ? "Yes" : "No");
+        static const char* computeMode[] = {"Default","Exclusive","Prohibited","ExclusiveProcess"};
+        printf("      %s \n", computeMode[cm]);
+    #else
+        printf("  GPU Clock Speed:                               %.2f GHz\n", prop.clockRate * 1e-6f);
+        printf("  Concurrent copy and execution:                 %s with %d copy engine(s)\n", (prop.deviceOverlap ? "Yes" : "No"), prop.asyncEngineCount);
+        printf("  Run time limit on kernels:                     %s\n", prop.kernelExecTimeoutEnabled ? "Yes" : "No");
+        static const char* computeMode[] = {"Default","Exclusive","Prohibited","ExclusiveProcess"};
+        printf("      %s \n", computeMode[prop.computeMode]);
+    #endif
     }
PATCH

# Aplica el parche en el árbol opencv de opencv-python
pushd /opt/opencv-python/opencv
git apply /tmp/opencv-cuda13.patch
popd

# Aplica el mismo parche en el repo opencv de /opt (para el build C++)
pushd /opt/opencv
git apply /tmp/opencv-cuda13.patch
popd

# ------------------------------------------------------------------------------
# 6) Parches para setup.py del wrapper (si tienes alguno)
# ------------------------------------------------------------------------------
cd /opt/opencv-python
git apply $TMP/patches.diff || echo "failed to apply git patches"
git diff || true

# ------------------------------------------------------------------------------
# 7) Flags de compilación (tus originales)
# ------------------------------------------------------------------------------
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
   -DBUILD_TESTS=OFF \
   -DBUILD_OPENCV_VIDEOSTAB=OFF \
   -DBUILD_opencv_rgbd=OFF"

# aarch64: NEON
if [ "$(uname -m)" == "aarch64" ]; then
    OPENCV_BUILD_ARGS="${OPENCV_BUILD_ARGS} -DENABLE_NEON=ON"
fi

# ------------------------------------------------------------------------------
# 8) Entorno y build del wheel (opencv-python)
# ------------------------------------------------------------------------------
export CMAKE_BUILD_PARALLEL_LEVEL=$(nproc)
export CMAKE_POLICY_VERSION_MINIMUM="3.5"
export CMAKE_LIBRARY_PATH=/usr/local/cuda/lib64/stubs
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export ENABLE_CONTRIB=1

cat <<EOF > /opt/opencv-python/cv2/version.py
opencv_version = "${OPENCV_VERSION}"
contrib = True
headless = False
rolling = False
EOF

CMAKE_ARGS="${OPENCV_BUILD_ARGS} -DOPENCV_EXTRA_MODULES_PATH=/opt/opencv-python/opencv_contrib/modules" \
pip3 wheel --wheel-dir=/opt --verbose .

ls /opt
cd /
rm -rf /opt/opencv-python

# ------------------------------------------------------------------------------
# 9) Instalar, probar y (opcional) subir wheel
# ------------------------------------------------------------------------------
pip3 install /opt/opencv*.whl
python3 -c "import cv2; print('OpenCV version:', str(cv2.__version__)); print(cv2.getBuildInformation())"
twine upload --verbose /opt/opencv*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"

# ------------------------------------------------------------------------------
# 10) Build C++ DEBs clásicos (usa /opt/opencv + opencv_contrib)
# ------------------------------------------------------------------------------
mkdir -p /opt/opencv/build
cd /opt/opencv/build

cmake \
    ${OPENCV_BUILD_ARGS} \
    -DOPENCV_EXTRA_MODULES_PATH=/opt/opencv_contrib/modules \
    ../

make -j$(nproc)
make install
make package

# ------------------------------------------------------------------------------
# 11) Subida de paquetes a tu APT (si procede)
# ------------------------------------------------------------------------------
mkdir -p /tmp/debs/
cp *.deb /tmp/debs/

tarpack upload OpenCV-${OPENCV_VERSION} /tmp/debs/ || echo "failed to upload tarball"
echo "installed" > "$TMP/.opencv"
