#---
# name: genai-bench
# group: sglang
# config: config.py
# depends: [cuda, cmake, pytorch, torchvision, torchaudio, torchao, triton, xformers, flashinfer, xgrammar]
# requires: '>=36'
# test: test.py
# notes: https://github.com/sgl-project/genai-bench
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG GENAI_BENCH_VERSION \
    GENAI_BENCH_BRANCH \
    IS_SBSA \
    TORCH_CUDA_ARCH_LIST \
    FORCE_BUILD=off \
    TMP=/tmp/genai-bench

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        libnuma-dev \
        libsndfile1 \
        libsndfile1-dev \
        libprotobuf-dev \
        libsm6 \
        libxext6 \
        libgl1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY build.sh install.sh $TMP/
RUN $TMP/install.sh || $TMP/build.sh || touch $TMP/.build.failed

# this retains the stage above for debugging on build failure
RUN if [ -f $TMP/.build.failed ]; then \
      echo "GENAI BENCH ${GENAI_BENCH_VERSION} build failed!"; \
      exit 1; \
    fi

RUN /tmp/transformers/install.sh
