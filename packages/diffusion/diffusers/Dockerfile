#---
# name: diffusers
# group: diffusion
# config: config.py
# depends: [pytorch, torchvision, transformers, onnx]
# requires: '>=34.1.0'
# test: test.py
# notes: https://github.com/huggingface/diffusers
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG DIFFUSERS_VERSION \
    FORCE_BUILD=off

ENV DIFFUSERS_FORCE_DISABLE_TRITON=1

COPY build.sh install.sh /tmp/DIFFUSERS/

RUN /tmp/DIFFUSERS/install.sh || /tmp/DIFFUSERS/build.sh
    
    
