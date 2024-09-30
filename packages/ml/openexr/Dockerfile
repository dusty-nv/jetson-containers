#---
# name: openexr
# group: ml
# config: config.py
# depends: [python, numpy]
# requires: '>=34.1.0'
# test: test.py
# notes: https://github.com/state-spaces/openexr
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG OPENEXR_VERSION \
    CUDAARCHS

RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenexr-dev \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get clean

COPY build.sh install.sh /tmp/openexr/

RUN /tmp/openexr/install.sh || /tmp/openexr/build.sh
