#---
# name: jetson-inference
# group: cv
# config: config.py
# depends: [cudastack:standard, pytorch, torchvision, cmake, gstreamer]
# test: [test_utils.sh, test_infer.sh]
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ADD https://api.github.com/repos/dusty-nv/jetson-inference/git/refs/heads/master /tmp/jetson_inference_version.json

COPY build.sh /tmp/build_jetson_inference.sh

RUN /tmp/build_jetson_inference.sh

ENV LD_PRELOAD=/lib/aarch64-linux-gnu/libGLdispatch.so.0:${LD_PRELOAD}
