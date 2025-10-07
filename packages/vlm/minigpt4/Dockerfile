#---
# name: minigpt4
# group: vlm
# config: config.py
# depends: [pytorch, torchvision, huggingface_hub]
# requires: '>=34.1.0'
# test: test.sh
# docs: docs.md
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /opt

# upgrade toolchain to g++-10 for --std=c++2a
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
		  gcc-10 \
		  g++-10 \
		  time \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

ARG CC=/usr/bin/gcc-10
ARG CXX=/usr/bin/g++-10

ARG CUDA_ARCHITECTURES

ARG MINIGPT4_CPP_REPO=Maknee/minigpt4.cpp
ARG MINIGPT4_CPP_BRANCH=master

COPY patch.*.diff /tmp/
ADD https://api.github.com/repos/${MINIGPT4_CPP_REPO}/git/refs/heads/${MINIGPT4_CPP_BRANCH} /tmp/minigpt4_cpp_version.json

RUN git clone --branch=${MINIGPT4_CPP_BRANCH} --depth=1 --recursive https://github.com/${MINIGPT4_CPP_REPO} && \
    cd minigpt4.cpp && \
    git apply /tmp/patch.minigpt4.diff && \
    git status && \
    mkdir build && \
    cd build && \
    cmake -DMINIGPT4_CUBLAS=ON -DMINIGPT4_CUDA_FP16=ON -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES} ../

#RUN cd minigpt4.cpp/build/_deps/llama_cpp-src && \
#    git apply /tmp/patch.llamacpp.diff && \
#    git status

RUN cd minigpt4.cpp/build && \
    cmake --build . --config Release --parallel $(nproc)

RUN uv pip install -r minigpt4.cpp/minigpt4/requirements.txt
RUN uv pip install gradio>=3.38.0

COPY benchmark.py minigpt4.cpp/minigpt4/

WORKDIR /
