#---
# name: tilelang
# group: attention
# config: config.py
# depends: [pytorch, triton, cutlass, tvm]
# requires: '>=35'
# test: test.py
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG TILELANG_VERSION \
    IS_SBSA \
    FORCE_BUILD=off

COPY build.sh install.sh /tmp/tilelang/

RUN /tmp/tilelang/install.sh || /tmp/tilelang/build.sh

