#---
# name: radial-attention
# group: attention
# config: config.py
# depends: [pytorch, triton, xformers, flashinfer]
# requires: '>=35'
# test: test.sh
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG RADIAL_ATTENTION_VERSION \
    IS_SBSA \
    FORCE_BUILD=off

COPY build.sh install.sh /tmp/radial-attention/

RUN /tmp/radial-attention/install.sh || /tmp/radial-attention/build.sh

WORKDIR /opt/radial-attention
CMD ["/bin/bash"]

