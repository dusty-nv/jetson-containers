#---
# name: cuda-python
# group: cuda
# config: config.py
# requires: '>=34.1.0'
# depends: [cuda, numpy]
# test: [test_driver.py, test_runtime.py]
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG CUDA_PYTHON_VERSION

COPY build.sh /tmp/build_cuda_python.sh

RUN pip3 install --no-cache-dir cuda-python==${CUDA_PYTHON_VERSION} || /tmp/build_cuda_python.sh