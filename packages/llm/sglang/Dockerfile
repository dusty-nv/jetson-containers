#---
# name: sglang
# group: llm
# config: config.py
# depends: [sgl-kernel, torch-memory-saver]
# requires: '>=36'
# test: test.py
# notes: https://github.com/sgl-project/sglang
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG SGLANG_VERSION \
    SGLANG_VERSION_SPEC \
    IS_SBSA \
    FORCE_BUILD=off \
    TMP=/tmp/sglang

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
      echo "SGLANG ${SGLANG_VERSION} build failed!"; \
      exit 1; \
    fi

RUN /tmp/transformers/install.sh
