#---
# name: onnxruntime
# group: ml
# config: config.py
# depends: [cuda, cudnn, tensorrt, cmake, python, numpy, onnx, openai-triton]
# test: test.py
# notes: the onnxruntime-gpu wheel that's built is saved in the container under /opt
#---
ARG BASE_IMAGE

FROM ${BASE_IMAGE}

ARG ONNXRUNTIME_URL
ARG ONNXRUNTIME_WHL

# https://elinux.org/Jetson_Zoo#ONNX_Runtime
RUN cd /opt && \
    wget --quiet --show-progress --progress=bar:force:noscroll --no-check-certificate ${ONNXRUNTIME_URL} -O ${ONNXRUNTIME_WHL} && \
    pip3 install --verbose ${ONNXRUNTIME_WHL}

# test import and print build info
RUN python3 -c 'import onnxruntime; print(onnxruntime.__version__);'
