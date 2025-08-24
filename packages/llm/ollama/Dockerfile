#---
# name: ollama
# group: llm
# config: config.py
# depends: [cuda, cudnn, cmake, python, numpy, huggingface_hub, sudonim, go]
# requires: '>=34.1.0'
# docs: docs.md
# test: test.sh
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG OLLAMA_VERSION \
    JETPACK_VERSION_MAJOR \
    IS_SBSA \
    CUDA_VERSION_MAJOR
    
ENV OLLAMA_VERSION=${OLLAMA_VERSION} \
    OLLAMA_HOST=0.0.0.0 \
    OLLAMA_LOGS=/data/logs/ollama.log \
    OLLAMA_MODELS=/data/models/ollama/models \
    OLLAMA_HOME=/opt/ollama \
    IS_SBSA=${IS_SBSA} \
    CUDA_VERSION_MAJOR=${CUDA_VERSION_MAJOR}
    
COPY nv_tegra_release /etc/nv_tegra_release

COPY build.sh install.sh /tmp/OLLAMA/

RUN chmod +x /tmp/OLLAMA/build.sh /tmp/OLLAMA/install.sh \
 && if [ "${IS_SBSA}" = "True" ]; then \
      /tmp/OLLAMA/build.sh; \
    else \
      /tmp/OLLAMA/install.sh || /tmp/OLLAMA/build.sh; \
    fi

COPY start_ollama /start_ollama
RUN chmod +x /start_ollama

CMD /start_ollama && /bin/bash

EXPOSE 11434