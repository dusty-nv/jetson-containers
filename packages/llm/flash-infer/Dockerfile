#---
# name: flashinfer
# group: llm
# config: config.py
# depends: [pytorch, triton]
# requires: '>=36'
# test: test.py
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG FLASHINFER_VERSION \
    FLASHINFER_VERSION_SPEC \
    FLASHINFER_ENABLE_AOT=1 \
    MAX_JOBS="$(nproc)" \
    FORCE_BUILD=off \
    TMP=/tmp/flashinfer

COPY build.sh install.sh $TMP/

RUN $TMP/install.sh || $TMP/build.sh || touch $TMP/.build.failed

# this retains the stage above for debugging on build failure
RUN if [ -f $TMP/.build.failed ]; then \
      echo "FlashInfer ${FLASHINFER_VERSION} build failed!"; \
      exit 1; \
    fi
    
