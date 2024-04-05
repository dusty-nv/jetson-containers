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
    ONNXRUNTIME_FLAGS
    
COPY build.sh /tmp/build_onnxruntime.sh

RUN pip3 install --no-cache-dir --verbose onnxruntime-gpu==${ONNXRUNTIME_VERSION} || \
    /tmp/build_onnxruntime.sh
