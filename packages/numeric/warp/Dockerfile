#---
# name: warp
# group: numerical
# depends: [cuda, cupy, numba, llvm]
# test: test.sh
# notes: https://github.com/NVIDIA/warp
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG WARP_VERSION \
    WARP_INSTALL \
    TMP=/tmp/warp \
    SRC=/opt/nvidia/warp

COPY install.sh $TMP/

RUN $TMP/install.sh

