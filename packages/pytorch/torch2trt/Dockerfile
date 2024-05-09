#---
# name: torch2trt
# group: pytorch
# depends: [pytorch, torchvision, tensorrt]
# test: test.py
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ADD https://api.github.com/repos/NVIDIA-AI-IOT/torch2trt/git/refs/heads/master /tmp/torch2trt_version.json

RUN cd /opt && \
    git clone --depth=1 https://github.com/NVIDIA-AI-IOT/torch2trt && \
    cd torch2trt && \
    #sed 's|collections.Sequence|collections.abc.Sequence|g' -i torch2trt/converters/interpolate.py && \
    #cat torch2trt/converters/interpolate.py | grep Sequence && \
    pip3 install --verbose . && \
    sed 's|^set(CUDA_ARCHITECTURES.*|#|g' -i CMakeLists.txt && \
    cmake -B build -DCUDA_ARCHITECTURES=${CUDA_ARCHITECTURES} . && \
    cmake --build build --target install && \
    ldconfig && \
    pip3 install --no-cache-dir --verbose nvidia-pyindex && \
    pip3 install --no-cache-dir --verbose onnx-graphsurgeon
    
RUN pip3 show torch2trt && python3 -c 'import torch2trt'
