#---
# name: onnxruntime
# group: ml
# config: config.py
# depends: [cuda, cudastack:standard, cmake, python, numpy, onnx]
# test: test.py
# notes: https://github.com/microsoft/onnxruntime
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG ONNXRUNTIME_VERSION \
    ONNXRUNTIME_BRANCH \
    ONNXRUNTIME_FLAGS \
    FORCE_BUILD=off

COPY install.sh build.sh /tmp/onnxruntime/

RUN /tmp/onnxruntime/install.sh || /tmp/onnxruntime/build.sh
