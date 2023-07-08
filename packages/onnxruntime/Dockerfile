#
# onnxruntime (https://onnxruntime.ai/docs/build/eps.html#nvidia-jetson-tx1tx2nanoxavier)
#
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG ONNXRUNTIME_VERSION=main

RUN pip3 uninstall -y onnxruntime && \
    git clone --branch ${ONNXRUNTIME_VERSION} --depth 1 --recursive https://github.com/microsoft/onnxruntime /tmp/onnxruntime && \
    cd /tmp/onnxruntime && \
    ./build.sh --config Release --update --build --parallel --build_wheel --allow_running_as_root \
        --cmake_extra_defines CMAKE_CXX_FLAGS="-Wno-unused-variable" \
        --cuda_home /usr/local/cuda --cudnn_home /usr/lib/aarch64-linux-gnu \
        --use_tensorrt --tensorrt_home /usr/lib/aarch64-linux-gnu && \
    cd build/Linux/Release && \
    make install && \
    pip3 install --no-cache-dir --verbose dist/onnxruntime_gpu-*.whl && \
    rm -rf /tmp/onnxruntime

# ImportError: `onnxruntime-gpu` is installed, but GPU dependencies are not loaded.
RUN sed -i 's/if "ORT_CUDA" not in file_string or "ORT_TENSORRT" not in file_string:/if False:/g' /usr/local/lib/python${PYTHON3_VERSION}/dist-packages/optimum/onnxruntime/utils.py

# test import and print build info
RUN python3 -c 'import onnxruntime; print(onnxruntime.get_build_info());'
