#---
# name: torch2trt
# group: pytorch
# depends: [pytorch, torchvision, tensorrt]
# test: test.py
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ADD https://api.github.com/repos/NVIDIA-AI-IOT/torch2trt/git/refs/heads/master /tmp/torch2trt_version.json
COPY install.sh patches/ /tmp/torch2trt/
RUN bash /tmp/torch2trt/install.sh
