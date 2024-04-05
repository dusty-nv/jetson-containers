#---
# name: ollama 
# group: llm
# config: config.py
# depends: [build-essential, cuda]
# requires: '>=34.1.0'
# docs: docs.md
#---
ARG BASE_IMAGE
ARG CMAKE_CUDA_ARCHITECTURES
ARG JETPACK_VERSION
ARG OLLAMA_REPO
ARG OLLAMA_BRANCH
ARG GOLANG_VERSION
ARG CMAKE_VERSION

FROM ${BASE_IMAGE} AS ollama-l4t-build

ARG OLLAMA_REPO
ARG OLLAMA_BRANCH
ARG GOLANG_VERSION
ARG CMAKE_VERSION
ARG CMAKE_CUDA_ARCHITECTURES

WORKDIR /opt

ADD https://api.github.com/repos/${OLLAMA_REPO}/git/refs/heads/${OLLAMA_BRANCH} /tmp/ollama_version.json
RUN git clone --branch=${OLLAMA_BRANCH} --depth=1 --recursive https://github.com/${OLLAMA_REPO}

COPY ollama_deps.sh /opt/ollama_deps.sh
RUN CMAKE_VERSION=${CMAKE_VERSION} GOLANG_VERSION=${GOLANG_VERSION} sh /opt/ollama_deps.sh

# generate llama.cpp backend to bundle with Ollama
WORKDIR ollama/llm/generate
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/nvidia/compat
ENV CMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}
RUN bash gen_linux.sh

WORKDIR /opt/ollama
RUN go build -trimpath .

# build the runtime container
FROM ${BASE_IMAGE}
COPY --from=ollama-l4t-build /opt/ollama/ollama /bin/ollama

ARG JETPACK_VERSION

EXPOSE 11434
ENV OLLAMA_HOST 0.0.0.0
ENV PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib:/usr/local/cuda/lib64:/usr/local/cuda/include
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV JETSON_JETPACK=${JETPACK_VERSION}

COPY test.sh /test.sh
COPY benchmark.py /benchmark.py
COPY benchmark.sh /benchmark.sh

ENTRYPOINT ["/bin/bash", "-c"]

CMD ["/bin/ollama serve"]
