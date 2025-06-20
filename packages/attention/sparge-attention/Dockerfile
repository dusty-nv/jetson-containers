#---
# name: sparge-attention
# group: attention
# config: config.py
# depends: [pytorch, torchvision, torchaudio, torchao, flash-attention, transformers, diffusers, opencv]
# requires: '>=35'
# test: test.py
# notes: https://github.com/thu-ml/SpargeAttn.git
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG SPARGE_ATTENTION_VERSION \
    FORCE_BUILD=off

COPY build.sh install.sh /tmp/sparge-attention/

RUN /tmp/sparge-attention/install.sh || /tmp/sparge-attention/build.sh
