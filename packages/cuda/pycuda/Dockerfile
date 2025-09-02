#---
# name: pycuda
# group: cuda
# config: config.py
# depends: [cuda, build-essential, python, numpy]
# test: test.py
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG PYCUDA_VERSION \
    FORCE_BUILD=off

COPY build.sh install.sh /tmp/pycuda/

RUN /tmp/pycuda/install.sh || /tmp/pycuda/build.sh
