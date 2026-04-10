#---
# name: fastvideo
# group: diffusion
# config: config.py
# depends: [cuda, pytorch, torchvision, torchaudio, transformers, diffusers, flash-attention, opencv, huggingface_hub]
# requires: '>=36'
# test: test.py
# notes: https://github.com/hao-ai-lab/FastVideo
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG FASTVIDEO_VERSION \
    FORCE_BUILD=off

COPY build.sh install.sh /tmp/FASTVIDEO/

RUN /tmp/FASTVIDEO/install.sh || /tmp/FASTVIDEO/build.sh
