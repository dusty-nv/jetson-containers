#---
# name: flax
# group: jax
# config: config.py
# depends: [jax]
# requires: '>=34.1.0'
# test: test.py
# notes: https://github.com/google/flax.git
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG FLAX_VERSION \
    CUDAARCHS

COPY build.sh install.sh /tmp/flax/

RUN /tmp/flax/install.sh || /tmp/flax/build.sh
