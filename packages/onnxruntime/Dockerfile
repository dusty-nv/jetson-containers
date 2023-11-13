#---
# name: onnxruntime
# group: ml
# config: config.py
# depends: [cmake, python, numpy, onnx]
# test: test.py
# notes: the onnxruntime-gpu wheel that's built is saved in the container under /opt
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG ONNXRUNTIME_VERSION=v1.16.1
ARG ONNXRUNTIME_FLAGS="--allow_running_as_root"
ARG CUDA_ARCHITECTURES=53;62;72;87

#ADD https://api.github.com/repos/microsoft/onnxruntime/git/refs/heads/${ONNXRUNTIME_VERSION} /tmp/onnxruntime_version.json

# https://onnxruntime.ai/docs/build/eps.html#nvidia-jetson-tx1tx2nanoxavier
# https://github.com/microsoft/onnxruntime/commit/b7b8b5b2ce80edb33990c7ae0fedac6ae3c623f4
RUN pip3 uninstall -y onnxruntime && \
    git clone https://github.com/microsoft/onnxruntime /tmp/onnxruntime && \
    cd /tmp/onnxruntime && \
    git checkout ${ONNXRUNTIME_VERSION} && \
    git submodule update --init --recursive && \
    sed -i 's|archive/3.4/eigen-3.4.zip;ee201b07085203ea7bd8eb97cbcb31b07cfa3efb|archive/3.4.0/eigen-3.4.0.zip;ef24286b7ece8737c99fa831b02941843546c081|' cmake/deps.txt || echo "cmake/deps.txt not found" && \
    ./build.sh --config Release --update --parallel --build --build_wheel \
        --skip_tests --skip_submodule_sync ${ONNXRUNTIME_FLAGS} \
        --cmake_extra_defines CMAKE_CXX_FLAGS="-Wno-unused-variable -I/usr/local/cuda/include" CMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" onnxruntime_BUILD_UNIT_TESTS=OFF \
        --cuda_home /usr/local/cuda --cudnn_home /usr/lib/$(uname -m)-linux-gnu \
        --use_tensorrt --tensorrt_home /usr/lib/$(uname -m)-linux-gnu && \
    cd build/Linux/Release && \
    make install && \
    cp dist/onnxruntime_gpu-*.whl /opt && \
    pip3 install --no-cache-dir --verbose /opt/onnxruntime_gpu-*.whl && \
    rm -rf /tmp/onnxruntime

# test import and print build info
RUN python3 -c 'import onnxruntime; print(onnxruntime.__version__);'