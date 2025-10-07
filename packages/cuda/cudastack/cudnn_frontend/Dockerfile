#---
# name: cudnn_frontend
# group: cuda
# config: config.py
# depends: [cuda, cudastack:standard, python, cmake, ninja, cuda-python]
# requires: '>=35'
# test: test.py
#---

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG CUDNN_FRONTEND_VERSION \
    CUDNN_FRONTEND_VERSION_SPEC \
    TORCH_CUDA_ARCH_LIST \
    MAX_JOBS="$(nproc)" \
    IS_SBSA \
    FORCE_BUILD=off \
    TMP=/tmp/cudnn_frontend

RUN apt-get update -y || true ; apt-get install --no-install-recommends libopenmpi-dev openmpi-bin -y && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean

COPY build.sh install.sh $TMP/

RUN $TMP/install.sh || $TMP/build.sh || touch $TMP/.build.failed

# this retains the stage above for debugging on build failure
RUN if [ -f $TMP/.build.failed ]; then \
      echo "cudnn_frontend ${CUDNN_FRONTEND_VERSION} build failed!"; \
      exit 1; \
    fi
