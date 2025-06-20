#---
# name: flexprefill
# group: attention
# config: config.py
# depends: [pytorch, torchvision, torchaudio, torchao, flash-attention, transformers]
# requires: '>=35'
# test: test.py
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG FLEXPREFILL_VERSION \
    FORCE_BUILD=off

COPY build.sh install.sh /tmp/flexprefill/

RUN /tmp/flexprefill/install.sh || /tmp/flexprefill/build.sh
