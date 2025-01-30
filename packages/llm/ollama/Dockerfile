#---
# name: ollama 
# group: llm
# requires: '>=34.1.0'
# docs: docs.md
#---
ARG BASE_IMAGE \
    CMAKE_CUDA_ARCHITECTURES \
    CUDA_VERSION_MAJOR \
    JETPACK_VERSION \
    OLLAMA_REPO \
    OLLAMA_BRANCH \
    GOLANG_VERSION \
    CMAKE_VERSION

# build the runtime container
FROM ${BASE_IMAGE}

ARG JETPACK_VERSION \
    CUDA_VERSION_MAJOR

EXPOSE 11434
ENV OLLAMA_HOST=0.0.0.0 \
    OLLAMA_MODELS=/data/models/ollama/models \
    PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    LD_LIBRARY_PATH=/usr/local/cuda/lib:/usr/local/cuda/lib64:/usr/local/cuda/include:${LD_LIBRARY_PATH} \
    JETSON_JETPACK=${JETPACK_VERSION} \
    CUDA_VERSION_MAJOR=${CUDA_VERSION_MAJOR}

RUN apt-get update && \
    apt install -y curl  && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean

RUN curl -fsSL https://ollama.com/install.sh | sh 

COPY start_ollama /

CMD /start_ollama && /bin/bash
