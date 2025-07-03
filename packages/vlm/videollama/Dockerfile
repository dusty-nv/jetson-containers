#---
# name: videollama
# group: vlm
# config: config.py
# depends: [pytorch, torchvision, torchaudio, torchao, bitsandbytes, flash-attention, transformers, diffusers, opencv, ffmpeg]
# requires: '>=35'
# test: test.py
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG VIDEOLLAMA_VERSION \
    FORCE_BUILD=off

COPY build.sh install.sh /tmp/videollama/

RUN /tmp/videollama/install.sh || /tmp/videollama/build.sh

WORKDIR /opt/videollama
