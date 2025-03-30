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

ARG CUDA_PYTHON_VERSION \
    FORCE_BUILD=off

COPY build.sh install.sh /tmp/cuda-python/

RUN /tmp/cuda-python/install.sh || /tmp/cuda-python/build.sh
