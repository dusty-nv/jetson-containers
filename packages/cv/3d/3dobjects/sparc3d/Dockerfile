#---
# name: sparc3d
# group: objects
# config: config.py
# depends: [cuda, pytorch, torchvision, triton, onnxruntime, opencv, transformers, diffusers, flash-attention, xformers, pymeshlab]
# requires: '>=34.1.0'
# notes: https://github.com/lizhihao6/Sparc3D
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG SPARC3D_VERSION

COPY install.sh /tmp/sparc3d/
RUN /tmp/sparc3d/install.sh
