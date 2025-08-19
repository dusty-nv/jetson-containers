#!/usr/bin/env bash
echo "Building opencv-python ${OPENCV_VERSION}"
set -ex
cd /opt

# ------------------------------------------------------------------------------
# 1) Dependencias
# ------------------------------------------------------------------------------
bash $TMP/install_deps.sh

# ------------------------------------------------------------------------------
# 2) Clonar repos
# ------------------------------------------------------------------------------
git clone --branch "${OPENCV_VERSION}" --recursive https://github.com/opencv/opencv \
  || git clone --recursive https://github.com/opencv/opencv

git clone --branch "${OPENCV_VERSION}" --recursive https://github.com/opencv/opencv_contrib \
  || git clone --recursive https://github.com/opencv/opencv_contrib

git clone --branch "${OPENCV_PYTHON}" --recursive https://github.com/opencv/opencv-python \
  || git clone --recursive https://github.com/opencv/opencv-python

# opencv_extra ayuda con tests/datasets (opcional pero útil)
git clone --branch "${OPENCV_VERSION}" --recursive https://github.com/opencv/opencv_extra \
  || git clone --recursive https://github.com/opencv/opencv_extra

# Forzar las ramas solicitadas (si el tag no existe, caer a 4.x)
cd /opt/opencv-python/opencv || git checkout --recurse-submodules origin/4.x
git checkout --recurse-submodules ${OPENCV_VERSION} || git checkout --recurse-submodules origin/4.x || true
cat modules/core/include/opencv2/core/version.hpp || true

cd /opt/opencv-python/opencv_contrib
git checkout --recurse-submodules ${OPENCV_VERSION} || git checkout --recurse-submodules origin/4.x || true

cd /opt/opencv-python/opencv_extra
git checkout --recurse-submodules ${OPENCV_VERSION} || git checkout --recurse-submodules origin/4.x || true

cd /opt/opencv
git checkout --recurse-submodules ${OPENCV_VERSION} || git checkout --recurse-submodules origin/4.x || true

cd /opt

# ------------------------------------------------------------------------------
# 3) Parches FP16 DNN (tus cambios)
# ------------------------------------------------------------------------------
function patch_opencv_dnn_fp16() {
    sed -i 's|weight != 1.0|(float)weight != 1.0f|' opencv/modules/dnn/src/cuda4dnn/primitives/normalize_bbox.hpp || true
    sed -i 's|nms_iou_threshold > 0|(float)nms_iou_threshold > 0.0f|' opencv/modules/dnn/src/cuda4dnn/primitives/region.hpp || true
    grep 'weight' opencv/modules/dnn/src/cuda4dnn/primitives/normalize_bbox.hpp || true
    grep 'nms_iou_threshold' opencv/modules/dnn/src/cuda4dnn/primitives/region.hpp || true
}
pushd /opt/opencv-python; patch_opencv_dnn_fp16; popd
pushd /opt;             patch_opencv_dnn_fp16; popd

# ------------------------------------------------------------------------------
# 4) Symlink cuDNN header (workaround)
# ------------------------------------------------------------------------------
ln -sfnv /usr/include/$(uname -i)-linux-gnu/cudnn_version_v*.h /usr/include/$(uname -i)-linux-gnu/cudnn_version.h || true

# ------------------------------------------------------------------------------
# 5) Parche CUDA 13 por REGEX (robusto a 4.12/4.13-dev)
# ------------------------------------------------------------------------------
cat > /tmp/patch_cuda13.py <<'PY'
import io, os, sys, re, pathlib

FILES = []
for root in ("/opt/opencv-python/opencv", "/opt/opencv"):
    p = pathlib.Path(root) / "modules/core/src/cuda_info.cpp"
    if p.exists():
        FILES.append(str(p))

def patch_text(s: str) -> str:
    # Helpers
    def rep(pattern, repl, flags=re.S):
        nonlocal s
        s2 = re.sub(pattern, repl, s, flags=flags)
        return s2

    # 1) clockRate()
    s = rep(
        r'(int\s+cv::cuda::DeviceInfo::clockRate\(\)\s*const\s*\{[^{}]*?)return\s+deviceProps\(\)\.get\(device_id_\)->clockRate\s*;',
        r'\1#if CUDART_VERSION >= 13000\n        int v = 0; cudaDeviceGetAttribute(&v, cudaDevAttrClockRate, device_id_);\n        return v;\n#else\n        return deviceProps().get(device_id_)->clockRate;\n#endif',
    )

    # 2) kernelExecTimeoutEnabled()
    s = rep(
        r'(bool\s+cv::cuda::DeviceInfo::kernelExecTimeoutEnabled\(\)\s*const\s*\{[^{}]*?)return\s+deviceProps\(\)\.get\(device_id_\)->kernelExecTimeoutEnabled\s*!=\s*0\s*;',
        r'\1#if CUDART_VERSION >= 13000\n        int v = 0; cudaDeviceGetAttribute(&v, cudaDevAttrKernelExecTimeout, device_id_);\n        return v != 0;\n#else\n        return deviceProps().get(device_id_)->kernelExecTimeoutEnabled != 0;\n#endif',
    )

    # 3) computeMode()
    s = rep(
        r'(cv::cuda::DeviceInfo::ComputeMode\s+cv::cuda::DeviceInfo::computeMode\(\)\s*const\s*\{\s*)static\s+ComputeMode\s+tbl\[\]\s*=\s*\{[^}]*\}\s*;\s*return\s+tbl\[deviceProps\(\)\.get\(device_id_\)->computeMode\]\s*;',
        r'\1#if CUDART_VERSION >= 13000\n        int v = 0; cudaDeviceGetAttribute(&v, cudaDevAttrComputeMode, device_id_);\n        static ComputeMode tbl[] = { Default, Exclusive, Prohibited, ExclusiveProcess };\n        return tbl[v];\n#else\n        static ComputeMode tbl[] = { Default, Exclusive, Prohibited, ExclusiveProcess };\n        return tbl[deviceProps().get(device_id_)->computeMode];\n#endif',
    )

    # 4) maxTexture1DLinear()
    s = rep(
        r'(int\s+cv::cuda::DeviceInfo::maxTexture1DLinear\(\)\s*const\s*\{[^{}]*?)return\s+deviceProps\(\)\.get\(device_id_\)->maxTexture1DLinear\s*;',
        r'\1#if CUDART_VERSION >= 13000\n#ifdef cudaDevAttrMaxTexture1DLinear\n        int v = 0; cudaDeviceGetAttribute(&v, cudaDevAttrMaxTexture1DLinear, device_id_);\n        return v;\n#else\n        return 0;\n#endif\n#else\n        return deviceProps().get(device_id_)->maxTexture1DLinear;\n#endif',
    )

    # 5) memoryClockRate()
    s = rep(
        r'(int\s+cv::cuda::DeviceInfo::memoryClockRate\(\)\s*const\s*\{[^{}]*?)return\s+deviceProps\(\)\.get\(device_id_\)->memoryClockRate\s*;',
        r'\1#if CUDART_VERSION >= 13000\n        int v = 0; cudaDeviceGetAttribute(&v, cudaDevAttrMemoryClockRate, device_id_);\n        return v;\n#else\n        return deviceProps().get(device_id_)->memoryClockRate;\n#endif',
    )

    # 6) printCudaDeviceInfo() — sustituir líneas de printf clásicas por bloque CUDA13
    # En lugar de borrar lo original, lo envolvemos con #if/#else
    s = rep(
        r'(void\s+cv::cuda::printCudaDeviceInfo\s*\(\s*int\s+device\s*\)\s*\{\s*[^{}]*?const\s+cudaDeviceProp&\s+prop\s*=\s*\*deviceProps\(\)\.get\(device\);\s*)'
        r'(\s*printf\([^;]*GPU Clock Speed[^;]*;[\s\S]*?printf\([^;]*Run time limit on kernels[^;]*;[\s\S]*?printf\([^;]*\[[^;]*computeMode[^;]*;)',
        r'\1'
        r'\n#if CUDART_VERSION >= 13000\n'
        r'    int clockKHz=0, timeout=0, cm=0, concurrent=0, asyncEngines=0;\n'
        r'    cudaDeviceGetAttribute(&clockKHz,  cudaDevAttrClockRate,           device);\n'
        r'    cudaDeviceGetAttribute(&timeout,   cudaDevAttrKernelExecTimeout,   device);\n'
        r'    cudaDeviceGetAttribute(&cm,        cudaDevAttrComputeMode,         device);\n'
        r'    cudaDeviceGetAttribute(&concurrent,cudaDevAttrConcurrentKernels,   device);\n'
        r'#ifdef cudaDevAttrAsyncEngineCount\n'
        r'    cudaDeviceGetAttribute(&asyncEngines, cudaDevAttrAsyncEngineCount, device);\n'
        r'#endif\n'
        r'    printf("  GPU Clock Speed:                               %.2f GHz\\n", clockKHz * 1e-6f);\n'
        r'    printf("  Concurrent copy and execution:                 %s with %d copy engine(s)\\n", (concurrent ? "Yes" : "No"), asyncEngines);\n'
        r'    printf("  Run time limit on kernels:                     %s\\n", timeout ? "Yes" : "No");\n'
        r'    static const char* computeMode[] = {"Default","Exclusive","Prohibited","ExclusiveProcess"};\n'
        r'    printf("      %s \\n", computeMode[cm]);\n'
        r'#else\n'
        r'\2'
        r'\n#endif',
    )

    return s

def patch_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        src = f.read()
    out = patch_text(src)
    if out != src:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(out)
        print(f"[patched] {path}")
        return True
    else:
        print(f"[noop]    {path} (patterns not found or already patched)")
        return False

ok = False
for f in FILES:
    ok |= patch_file(f)

sys.exit(0 if ok else 0)  # no error si ya estaba aplicado
PY

python3 /tmp/patch_cuda13.py

# ------------------------------------------------------------------------------
# 6) Parches adicionales del wrapper (si tienes alguno)
# ------------------------------------------------------------------------------
cd /opt/opencv-python
git apply $TMP/patches.diff || echo "failed to apply git patches"
git diff || true

# ------------------------------------------------------------------------------
# 7) Flags de compilación
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
# 10) Build C++ .deb (repo /opt/opencv + contrib)
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
