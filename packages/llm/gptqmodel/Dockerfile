#---
# name: gptqmodel
# alias: gptqmodel
# group: llm
# config: config.py
# depends: [transformers]
# requires: '>=34.1.0'
# test: test.py
# docs: docs.md
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG GPTQMODEL_VERSION \
    GPTQMODEL_BRANCH \
    FORCE_BUILD="off"

COPY build.sh build.sh /tmp/gptqmodel/

RUN pip3 install gekko && \
    pip3 install gptqmodel==${GPTQMODEL_VERSION} || \
    /tmp/build.sh
