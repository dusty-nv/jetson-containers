#---
# name: torch-memory-saver
# group: pytorch
# config: config.py
# depends: [cuda, pytorch, triton, cmake]
# buildkit_device: nvidia.com/gpu=all
# test: test.py
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG TORCH_MEMORY_SAVER_VERSION \
    FORCE_BUILD=off

COPY build.sh install.sh /tmp/torch_memory_saver/

RUN /tmp/torch_memory_saver/install.sh || /tmp/torch_memory_saver/build.sh
