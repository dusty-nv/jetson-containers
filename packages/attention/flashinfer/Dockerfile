#---
# name: flashinfer
# group: attention
# config: config.py
# depends: [cuda, cudastack:standard, pytorch, triton, cuda-python, cudnn_frontend, llvm:21]
# requires: '>=35'
# test: test.py
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG FLASHINFER_VERSION \
    FLASHINFER_VERSION_SPEC \
    FLASHINFER_ENABLE_AOT=1 \
    TORCH_CUDA_ARCH_LIST \
    FLASHINFER_CUDA_ARCH_LIST \
    MAX_JOBS="$(nproc)" \
    IS_SBSA \
    FORCE_BUILD=off \
    TMP=/tmp/flashinfer

COPY build.sh install.sh $TMP/

RUN apt-get update && \
    apt-get install -y --no-install-recommends libopenmpi-dev openmpi-bin && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean && \
    \
    $TMP/install.sh || $TMP/build.sh || \
    (echo "FlashInfer ${FLASHINFER_VERSION} build failed!" && exit 1)
