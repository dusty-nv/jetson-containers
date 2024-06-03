#---
# name: ollama 
# group: llm
# config: config.py
# depends: [build-essential, cuda]
# requires: '>=34.1.0'
# test: test.sh
# docs: docs.md
#---
ARG BASE_IMAGE \
    CMAKE_CUDA_ARCHITECTURES \
    JETPACK_VERSION \
    OLLAMA_REPO \
    OLLAMA_BRANCH \
    GOLANG_VERSION \
    CMAKE_VERSION

FROM ${BASE_IMAGE} AS ollama-l4t-build

ARG OLLAMA_REPO \
    OLLAMA_BRANCH \
    GOLANG_VERSION \
    CMAKE_VERSION \
    CMAKE_CUDA_ARCHITECTURES

WORKDIR /opt

ADD https://api.github.com/repos/${OLLAMA_REPO}/git/refs/heads/${OLLAMA_BRANCH} /tmp/ollama_version.json
RUN git clone --branch=${OLLAMA_BRANCH} --depth=1 --recursive https://github.com/${OLLAMA_REPO}

COPY ollama_deps.sh /opt/ollama_deps.sh
RUN CMAKE_VERSION=${CMAKE_VERSION} GOLANG_VERSION=${GOLANG_VERSION} sh /opt/ollama_deps.sh

# generate llama.cpp backend to bundle with Ollama
WORKDIR ollama/llm/generate
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/nvidia/compat:${LD_LIBRARY_PATH} \
    CMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}
RUN sed 's|-j8|-j$(nproc)|' -i gen_common.sh
RUN bash gen_linux.sh

WORKDIR /opt/ollama
RUN go build -trimpath .

# build the runtime container
FROM ${BASE_IMAGE}
COPY --from=ollama-l4t-build /opt/ollama/ollama /bin/ollama

ARG JETPACK_VERSION

EXPOSE 11434
ENV OLLAMA_HOST=0.0.0.0 \
    OLLAMA_MODELS=/data/models/ollama/models \
    PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    LD_LIBRARY_PATH=/usr/local/cuda/lib:/usr/local/cuda/lib64:/usr/local/cuda/include:${LD_LIBRARY_PATH} \
    JETSON_JETPACK=${JETPACK_VERSION}

COPY benchmark.py benchmark.sh test.sh /opt/ollama/
COPY start_ollama /

CMD /start_ollama && /bin/bash
