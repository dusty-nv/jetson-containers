#---
# name: partpacker
# group: objects
# config: config.py
# depends: [cuda, pytorch, torchvision, triton, onxxruntime, opencv, transformers, diffusers, accelerate, flash-attention, xformers, pymeshlab]
# requires: '>=34.1.0'
# notes: https://github.com/NVlabs/PartPacker
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG PARTPACKER_VERSION

COPY install.sh /tmp/partpacker/
RUN /tmp/partpacker/install.sh
