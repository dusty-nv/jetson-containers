#
# name: torch2trt
# category: ml
# depends: [pytorch, torchvision]
# test: test.py
#
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

RUN cd /opt && \
    git clone --depth=1 https://github.com/NVIDIA-AI-IOT/torch2trt && \
    cd torch2trt && \
    pip3 install --verbose . && \
    cmake -B build . && \
    cmake --build build --target install && \
    ldconfig

RUN pip3 show torch2trt && python3 -c 'import torch2trt'
