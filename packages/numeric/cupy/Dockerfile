#---
# name: cupy
# group: cuda
# config: config.py
# depends: [cuda, python, numpy]
# test: test.py
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG CUPY_VERSION \
    CUPY_NVCC_GENERATE_CODE \
    FORCE_BUILD=off

COPY build.sh install.sh /tmp/cupy/

RUN  /tmp/cupy/install.sh || /tmp/cupy/build.sh
