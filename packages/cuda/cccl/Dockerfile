#---
# name: cuda_cccl
# group: cuda
# config: config.py
# depends: [cuda, cudnn, python, cmake, ninja, cuda-python, cupy]
# requires: '>=35'
# test: test.py
#---

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG CCCL_VERSION \
    CCCL_VERSION_SPEC \
    TORCH_CUDA_ARCH_LIST \
    MAX_JOBS="$(nproc)" \
    IS_SBSA \
    FORCE_BUILD=off \
    TMP=/tmp/cuda_cccl

RUN apt-get update -y || true ; apt-get install --no-install-recommends cuda-cccl libopenmpi-dev openmpi-bin -y && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean


COPY build.sh install.sh $TMP/

RUN $TMP/install.sh || $TMP/build.sh || touch $TMP/.build.failed

# this retains the stage above for debugging on build failure
RUN if [ -f $TMP/.build.failed ]; then \
      echo "cuda_cccl ${cuda_cccl_VERSION} build failed!"; \
      exit 1; \
    fi
