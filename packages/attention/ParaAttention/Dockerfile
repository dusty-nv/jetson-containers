#---
# name: paraattention
# group: attention
# config: config.py
# depends: [pytorch, torchvision, torchaudio, torchao, flash-attention, transformers, diffusers, opencv]
# requires: '>=35'
# test: test.py
# docs: https://github.com/chengzeyi/ParaAttention
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG PARAATENTTION_VERSION \
    FORCE_BUILD=off

COPY build.sh install.sh /tmp/paraattention/

RUN /tmp/paraattention/install.sh || /tmp/paraattention/build.sh
