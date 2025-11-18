#---
# name: kai_scheduler
# group: dynamo
# config: config.py
# depends: [cuda, rust, cmake, ninja, helm, llvm:21]
# requires: '>=34.1.0'
# notes: https://github.com/ai-kai_scheduler/kai_scheduler
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG KAI_SCHEDULER_VERSION \
    KAI_SCHEDULER_VERSION_SPEC \
    SYSTEM_ARM


RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        libnuma-dev \
        libsndfile1 \
        libsndfile1-dev \
        libprotobuf-dev \
        libsm6 \
        libxext6 \
        libgl1 \
        protobuf-compiler \
        libucx0 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY build.sh install.sh /tmp/kai_scheduler/

RUN /tmp/kai_scheduler/install.sh || /tmp/kai_scheduler/build.sh
