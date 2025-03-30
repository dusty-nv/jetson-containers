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
    CUPY_NVCC_GENERATE_CODE

COPY build.sh /tmp/build_cupy.sh

RUN pip3 install cupy || /tmp/build_cupy.sh
