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

# https://onnxruntime.ai/docs/build/eps.html#nvidia-jetson-tx1tx2nanoxavier
RUN pip3 uninstall -y onnxruntime && \
    git clone --branch ${ONNXRUNTIME_VERSION} --depth 1 --recursive https://github.com/microsoft/onnxruntime /tmp/onnxruntime && \
    cd /tmp/onnxruntime && \
    ./build.sh --config Release --update --build --parallel --build_wheel --allow_running_as_root \
        --cmake_extra_defines CMAKE_CXX_FLAGS="-Wno-unused-variable" \
        --cuda_home /usr/local/cuda --cudnn_home /usr/lib/aarch64-linux-gnu \
        --use_tensorrt --tensorrt_home /usr/lib/aarch64-linux-gnu && \
    cd build/Linux/Release && \
    make install && \
    cp dist/onnxruntime_gpu-*.whl /opt && \
    pip3 install --no-cache-dir --verbose /opt/onnxruntime_gpu-*.whl && \
    rm -rf /tmp/onnxruntime

# ImportError: `onnxruntime-gpu` is installed, but GPU dependencies are not loaded.
#RUN sed -i 's/if "ORT_CUDA" not in file_string or "ORT_TENSORRT" not in file_string:/if False:/g' /usr/local/lib/python${PYTHON3_VERSION}/dist-packages/optimum/onnxruntime/utils.py

# onnx module isn't a dependency but is commonly used
RUN pip3 install --no-cache-dir --verbose onnx

# test import and print build info
RUN python3 -c 'import onnxruntime; print(onnxruntime.get_build_info());'