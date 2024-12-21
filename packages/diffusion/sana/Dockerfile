#---
# name: sana
# group: sana
# config: config.py
# depends: [transformers, xformers, diffusers, triton]
# requires: '>=34.1.0'
# test: test.py
# notes: https://github.com/NVlabs/Sana
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG SANA_VERSION

COPY build.sh install.sh /tmp/SANA/

RUN /tmp/SANA/install.sh || /tmp/SANA/build.sh
