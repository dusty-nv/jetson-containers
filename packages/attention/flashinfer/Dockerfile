#---
# name: flashinfer
# group: attention
# config: config.py
# depends: [cuda, pytorch, triton, cuda-python, cudnn_frontend]
# requires: '>=35'
# test: test.py
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG FLASHINFER_VERSION \
    FLASHINFER_VERSION_SPEC \
    FLASHINFER_ENABLE_AOT=1 \
    TORCH_CUDA_ARCH_LIST \
    MAX_JOBS="$(nproc)" \
    IS_SBSA \
    FORCE_BUILD=off \
    TMP=/tmp/flashinfer

RUN apt-get update -y || true ; apt-get install --no-install-recommends libopenmpi-dev openmpi-bin -y && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean

COPY build.sh install.sh $TMP/

ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}

RUN $TMP/install.sh || $TMP/build.sh || touch $TMP/.build.failed

# this retains the stage above for debugging on build failure
RUN if [ -f $TMP/.build.failed ]; then \
      echo "FlashInfer ${FLASHINFER_VERSION} build failed!"; \
      exit 1; \
    fi
