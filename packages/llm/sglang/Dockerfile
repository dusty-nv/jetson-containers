#---
# name: sglang
# group: llm
# config: config.py
# depends: [pytorch, torchvision, torchaudio, transformers, triton:3.1.0, vllm]
# requires: '>=34.1.0'
# test: test.sh
# notes: https://github.com/sgl-project/sglang
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG SGLANG_VERSION \
    FLASHINFER_VERSION \
    MAX_JOBS="$(nproc)" \
    FORCE_BUILD=off

RUN apt-get update -y && apt-get install -y libnuma-dev \
    libsndfile1 libsndfile1-dev libprotobuf-dev libsm6 libxext6 libgl1 && \
    rm -rf /var/lib/apt/lists/*

COPY build.sh install.sh /tmp/SGLANG/
RUN /tmp/SGLANG/install.sh || /tmp/SGLANG/build.sh || touch /tmp/SGLANG/.build.failed

# this retains the stage above for debugging on build failure
RUN if [ -f /tmp/SGLANG/.build.failed ]; then \
      echo "SGLANG ${SGLANG_VERSION} build failed!"; \
      exit 1; \
    fi

# Default command
CMD ["/bin/bash"]
