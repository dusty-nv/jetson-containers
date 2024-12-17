#---
# name: ollama 
# group: llm
# config: config.py
# depends: [cuda, python]
# requires: '>=34.1.0'
# test: test.sh
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

FROM ${BASE_IMAGE} AS ollama-l4t-build

ARG OLLAMA_REPO \
    OLLAMA_BRANCH \
    GOLANG_VERSION \
    CMAKE_VERSION \
    CUDA_VERSION_MAJOR \
    CMAKE_CUDA_ARCHITECTURES
    
#ADD https://api.github.com/repos/${OLLAMA_REPO}/git/refs/heads/${OLLAMA_BRANCH} /tmp/ollama_version.json
ADD https://api.github.com/repos/${OLLAMA_REPO}/git/refs/tags/${OLLAMA_BRANCH} /tmp/ollama_version.json
RUN git clone --branch=${OLLAMA_BRANCH} --depth=1 --recursive https://github.com/${OLLAMA_REPO} /opt/ollama

COPY ollama_deps.sh /opt/ollama_deps.sh
RUN CMAKE_VERSION=${CMAKE_VERSION} GOLANG_VERSION=${GOLANG_VERSION} sh /opt/ollama_deps.sh

ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/nvidia/compat:${LD_LIBRARY_PATH} \
    CMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}
    
# https://github.com/dusty-nv/jetson-containers/issues/592#issuecomment-2323177164
# export VERSION=$(git describe --tags --first-parent --abbrev=7 --long --dirty --always | sed -e "s/^v//g") && \
RUN cd /opt/ollama && \
    export VERSION="0.0.0" && \
    export GOFLAGS="'-ldflags=-w -s \"-X=github.com/ollama/ollama/version.Version=$VERSION\" \"-X=github.com/ollama/ollama/server.mode=release\"'" && \
    export GOARCH=arm64 && \
    echo "OLLAMA_VERSION=${VERSION}  GOFLAGS=${GOFLAGS}" && \
    make -j$(nproc) && \
    go build -trimpath .

# build the runtime container
FROM ${BASE_IMAGE}

COPY --from=ollama-l4t-build /opt/ollama/ollama /bin/ollama
COPY --from=ollama-l4t-build /opt/ollama/dist/linux-arm64/lib/ollama /usr/lib/ollama

ARG JETPACK_VERSION \
    CUDA_VERSION_MAJOR

EXPOSE 11434
ENV OLLAMA_HOST=0.0.0.0 \
    OLLAMA_MODELS=/data/models/ollama/models \
    PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    LD_LIBRARY_PATH=/usr/local/cuda/lib:/usr/local/cuda/lib64:/usr/local/cuda/include:${LD_LIBRARY_PATH} \
    JETSON_JETPACK=${JETPACK_VERSION} \
    CUDA_VERSION_MAJOR=${CUDA_VERSION_MAJOR}

RUN python3 -m pip install ollama

COPY benchmark.py benchmark.sh test.sh test.py /opt/ollama/
COPY start_ollama /

CMD /start_ollama && /bin/bash
