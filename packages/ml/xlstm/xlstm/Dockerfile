#---
# name: xlstm
# group: xlstm
# config: config.py
# depends: [cuda, pytorch, triton, mlstm_kernels]
# requires: '>=34.1.0'
# test: test.py
# notes: https://github.com/NX-AI/xlstm
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG XLSTM_VERSION \
    XLSTM_VERSION_SPEC \
    FORCE_BUILD=off

COPY build.sh install.sh /tmp/xlstm/
RUN /tmp/xlstm/install.sh || /tmp/xlstm/build.sh
