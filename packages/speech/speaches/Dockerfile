#---
# name: speaches
# group: audio
# depends: [build-essential, cuda:12.8, cudnn:9.8, python:3.12]
# requires: '>=36.0'
# docs: docs.md
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libopenblas0          libopenblas-dev  \
        wget                  gnupg2           \
        libsndfile1                            \
        build-essential       cmake            \
        ninja-build && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/arm64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    apt-get update && \
    apt-get install -y --no-install-recommends cudnn9-cuda-12 && \
    rm -rf /var/lib/apt/lists/*

# Build CTranslate2
RUN git clone --recursive https://github.com/OpenNMT/CTranslate2.git && \
    cd CTranslate2 && mkdir build && cd build && \
    cmake .. -GNinja -DCMAKE_BUILD_TYPE=Release \
              -DWITH_CUDA=ON  -DWITH_CUDNN=ON \
              -DWITH_OPENBLAS=ON -DWITH_MKL=OFF \
              -DOPENMP_RUNTIME=COMP && \
    ninja && ninja install && ldconfig && \
    cd /workspace

    RUN cd CTranslate2/python && \
    python3 -m pip install --upgrade pip setuptools wheel && \
    python3 -m pip install -r install_requirements.txt && \
    python3 setup.py bdist_wheel && \
    python3 -m pip install dist/*.whl && \
    cd /workspace

# Install speaches
RUN python3 -m pip install uv && \
    git clone https://github.com/speaches-ai/speaches.git && \
    cd speaches && \
    uv venv && \
    . .venv/bin/activate && \
    uv sync --all-extras && \
    uv pip install /workspace/CTranslate2/python/dist/*.whl && \
    deactivate && \
    cd /workspace

COPY model_aliases.json /workspace/speaches/ 

COPY start-server.sh /workspace/speaches/
COPY download-models.sh /workspace/speaches/
RUN chmod +x /workspace/speaches/start-server.sh /workspace/speaches/download-models.sh

ARG PORT=8000
ENV PORT=${PORT}
EXPOSE ${PORT}

WORKDIR /workspace/speaches

ENTRYPOINT ["/workspace/speaches/start-server.sh"]
