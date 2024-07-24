#---
# name: openvla
# group: llm
# depends: [transformers, flash-attention, bitsandbytes, tensorboard, h5py]
# test: [test.sh, test.py]
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ADD https://api.github.com/repos/dusty-nv/openvla/git/refs/heads/main /tmp/openvla_version.json

RUN git clone --depth=1 --recursive https://github.com/dusty-nv/openvla /opt/openvla && \
    cd /opt/openvla && \
    pip3 install --verbose -e .
