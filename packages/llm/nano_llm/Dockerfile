#---
# name: nano_llm
# group: llm
# config: config.py
# depends: [mlc, riva-client:python, jetson-utils, torch2trt, torchaudio, piper1-tts, nanodb, mimicgen]
# requires: '>=35'
# docs: docs.md
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG NANO_LLM_BRANCH=main \
    NANO_LLM_PATH=/opt/NanoLLM

ENV PYTHONPATH=${PYTHONPATH}:${NANO_LLM_PATH} \
    SSL_KEY=/etc/ssl/private/localhost.key.pem \
    SSL_CERT=/etc/ssl/private/localhost.cert.pem
    
ADD https://api.github.com/repos/dusty-nv/NanoLLM/git/refs/heads/${NANO_LLM_BRANCH} /tmp/nano_llm_version.json

COPY install.sh /tmp/nano_llm/install.sh

RUN /tmp/nano_llm/install.sh