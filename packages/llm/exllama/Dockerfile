#---
# name: exllama
# group: llm
# config: config.py
# depends: [pytorch, huggingface_hub]
# requires: '>=34.1.0'
# test: test.sh
# docs: docs.md
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG EXLLAMA_VERSION \
    EXLLAMA_BRANCH

COPY build.sh /tmp/build_exllama.sh

RUN git clone --branch=v${EXLLAMA_BRANCH} --depth=1 --recursive https://github.com/turboderp/exllamav2 /opt/exllamav2 && \
    pip3 install --no-cache-dir --verbose exllamav2==${EXLLAMA_VERSION} || \
    /tmp/build_exllama.sh
