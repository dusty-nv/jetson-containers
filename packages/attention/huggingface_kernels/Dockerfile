#---
# name: huggingface_kernels
# group: attention
# config: config.py
# depends: [pytorch, triton, huggingface_hub, flash-attention]
# requires: '>=35'
# test: test.py
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG KERNELS_VERSION \
    IS_SBSA \
    TORCH_CUDA_ARCH_LIST \
    FORCE_BUILD=off

COPY build.sh install.sh /tmp/huggingface_kernels/

RUN /tmp/huggingface_kernels/install.sh || /tmp/huggingface_kernels/build.sh
