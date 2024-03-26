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

ARG ONNXRUNTIME_URL \
    ONNXRUNTIME_WHL

RUN set -ex \
    && wget --quiet --show-progress --progress=bar:force:noscroll --no-check-certificate ${ONNXRUNTIME_URL} -O /opt/${ONNXRUNTIME_WHL} \
    && pip3 install --verbose /opt/${ONNXRUNTIME_WHL} \
    \
    && python3 -c 'import onnxruntime; print(onnxruntime.__version__);'
