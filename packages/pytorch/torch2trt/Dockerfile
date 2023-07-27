#---
# name: torch2trt
# group: ml
# config: config.py
# depends: [pytorch, torchvision]
# test: test.py
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG CUDA_ARCHITECTURES

RUN cd /opt && \
    git clone --depth=1 https://github.com/NVIDIA-AI-IOT/torch2trt && \
    cd torch2trt && \
    pip3 install --verbose . && \
    sed 's|^set(CUDA_ARCHITECTURES.*|#|g' -i CMakeLists.txt && \
    cmake -B build -DCUDA_ARCHITECTURES=${CUDA_ARCHITECTURES} . && \
    cmake --build build --target install && \
    ldconfig

RUN pip3 show torch2trt && python3 -c 'import torch2trt'
