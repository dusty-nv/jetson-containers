#---
# name: cuda-repo
# group: cuda
# config: config.py
# depends: [build-essential]
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG DISTRO \
    CUDA_VERSION_MAJOR \
    L4T_VERSION_MAJOR \
    IS_SBSA

COPY install.sh /tmp/cuda-repo/install.sh
RUN /tmp/cuda-repo/install.sh && rm -rf /tmp/cuda-repo
