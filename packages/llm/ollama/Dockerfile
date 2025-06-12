#---
# name: ollama
# group: llm
# config: config.py
# depends: [cuda, python]
# requires: '>=34.1.0'
# docs: docs.md
# test: test.sh
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG OLLAMA_VERSION \
    JETPACK_VERSION_MAJOR \
    TMP=/tmp/ollama
    
ENV OLLAMA_VERSION=${OLLAMA_VERSION} \
    OLLAMA_HOST=0.0.0.0 \
    OLLAMA_LOGS=/data/logs/ollama.log \
    OLLAMA_MODELS=/data/models/ollama/models
    
COPY nv_tegra_release /etc/nv_tegra_release

COPY install.sh $TMP/
RUN $TMP/install.sh

COPY start_ollama /
CMD /start_ollama && /bin/bash

EXPOSE 11434