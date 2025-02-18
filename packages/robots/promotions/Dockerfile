#---
# name: protomotions
# group: robots
# config: config.py
# depends: [ninja, pytorch, torchvision, torchaudio, triton, genesis]
# requires: '>=34.1.0'
# test: test.py
# notes: https://github.com/NVlabs/ProtoMotions
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG PROTOMOTIONS_VERSION \
    CUDAARCHS

COPY build.sh install.sh /tmp/PROTOMOTIONS_VERSION/

RUN /tmp/PROTOMOTIONS/install.sh || /tmp/PROTOMOTIONS/build.sh
