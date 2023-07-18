#
# name: onnxruntime
# category: ml
# config: config.py
# depends: [cmake, python]
# test: test.py
# notes: the onnxruntime-gpu wheel that's built is saved in the container under /opt
#
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG ONNXRUNTIME_VERSION=main
ARG CUDA_ARCHITECTURES=53;62;72;87

# needed for building older versions of onnxruntime (~1.11) on JetPack 4 (Target "Python::NumPy" not found)
#RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1 && \
#    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1
    
# https://onnxruntime.ai/docs/build/eps.html#nvidia-jetson-tx1tx2nanoxavier
RUN pip3 uninstall -y onnxruntime && \
    git clone --branch ${ONNXRUNTIME_VERSION} --depth 1 --recursive https://github.com/microsoft/onnxruntime /tmp/onnxruntime && \
    cd /tmp/onnxruntime && \
    ./build.sh --config Release --update --build --parallel --allow_running_as_root --skip_tests --skip_submodule_sync --build_wheel \
        --cmake_extra_defines CMAKE_CXX_FLAGS="-Wno-unused-variable" CMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" onnxruntime_BUILD_UNIT_TESTS=OFF \
        --cuda_home /usr/local/cuda --cudnn_home /usr/lib/aarch64-linux-gnu \
        --use_tensorrt --tensorrt_home /usr/lib/aarch64-linux-gnu && \
    cd build/Linux/Release && \
    make install && \
    cp dist/onnxruntime_gpu-*.whl /opt && \
    pip3 install --no-cache-dir --verbose /opt/onnxruntime_gpu-*.whl && \
    rm -rf /tmp/onnxruntime

# onnx module isn't a dependency but is commonly used
RUN pip3 install --no-cache-dir --verbose onnx

# test import and print build info
RUN python3 -c 'import onnxruntime; print(onnxruntime.get_build_info());'