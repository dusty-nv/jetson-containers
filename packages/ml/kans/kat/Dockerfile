#---
# name: kat
# group: vit
# config: config.py
# depends: [pytorch, torchvision, triton, transformers]
# requires: '>=34.1.0'
# test: test.py
# notes: https://github.com/Adamdad/kat
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG KAT_VERSION \
    CUDAARCHS

COPY build.sh test.py /tmp/KAT/

RUN /tmp/KAT/build.sh
