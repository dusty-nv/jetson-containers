#---
# name: mlc
# group: llm
# config: config.py
# depends: [transformers, triton, llvm, sudonim]
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
    FORCE_BUILD=off \
    SOURCE_DIR=/opt/mlc-llm \
    TMP_DIR=/tmp/mlc

ENV LD_LIBRARY_PATH="/usr/local/lib/python${PYTHON_VERSION}/dist-packages/tvm:/opt/venv/lib/python${PYTHON_VERSION}/site-packages/tvm:${LD_LIBRARY_PATH}" \
    TVM_HOME=${SOURCE_DIR}/3rdparty/tvm \
    TVM_SOURCE_DIR=${SOURCE_DIR}/3rdparty/tvm \
    MLC_LLM_SOURCE_DIR=${SOURCE_DIR}
    
COPY build.sh install.sh benchmark.py wheels/ ${TMP_DIR}/
COPY ${MLC_PATCH} ${TMP_DIR}/patch.diff

RUN ${TMP_DIR}/install.sh || ${TMP_DIR}/build.sh || echo "BUILD FAILED (mlc ${MLC_VERSION} ${MLC_COMMIT})"
