#---
# name: onnxruntime
# group: ml
# config: config.py
# depends: [cuda, cudnn, tensorrt, cmake, python, numpy, onnx]
# test: test.py
# notes: the `onnxruntime-gpu` wheel that's built is saved in the container under `/opt`
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG ONNXRUNTIME_VERSION \
    ONNXRUNTIME_BRANCH \
    ONNXRUNTIME_FLAGS \
    FORCE_BUILD=off

COPY install.sh build.sh /tmp/onnxruntime/

RUN /tmp/onnxruntime/install.sh || /tmp/onnxruntime/build.sh
