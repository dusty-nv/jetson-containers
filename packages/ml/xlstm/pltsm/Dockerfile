#---
# name: plstm
# group: xltsm
# config: config.py
# depends: [cuda, pytorch, triton, mlstm_kernels]
# requires: '>=34.1.0'
# test: test.py
# notes: https://github.com/ml-jku/plstm
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG PLSTM_VERSION \
    PLSTM_VERSION_SPEC \
    FORCE_BUILD=off

COPY build.sh install.sh /tmp/plstm/
RUN /tmp/plstm/install.sh || /tmp/plstm/build.sh
