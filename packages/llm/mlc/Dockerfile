#---
# name: mlc
# group: llm
# config: config.py
# depends: [transformers]
# requires: '>=35'
# test: [test.py, test.sh]
# docs: docs.md
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG TVM_VERSION \
    MLC_VERSION \
    MLC_COMMIT \
    MLC_PATCH \
    LLVM_VERSION=17 \
    FORCE_BUILD=off

ENV LD_LIBRARY_PATH="/usr/local/lib/python${PYTHON_VERSION}/dist-packages/tvm:${LD_LIBRARY_PATH}" \
    TVM_HOME=/opt/mlc-llm/3rdparty/tvm
    
COPY build.sh install.sh benchmark.py /tmp/mlc/
COPY ${MLC_PATCH} /tmp/mlc/patch.diff

RUN /tmp/mlc/install.sh || /tmp/mlc/build.sh
