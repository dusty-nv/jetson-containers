#---
# name: pycuda
# group: cuda
# config: config.py
# depends: [cuda, build-essential, python, numpy]
# test: test.py
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG PYCUDA_VERSION

COPY build.sh /tmp/build_pycuda.sh

RUN pip3 install pycuda || /tmp/build_pycuda.sh
