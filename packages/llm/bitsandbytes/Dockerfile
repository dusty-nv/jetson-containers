#---
# name: bitsandbytes
# group: llm
# config: config.py
# requires: '>=35'
# depends: [transformers]
# test: test.py
# notes: fork of https://github.com/TimDettmers/bitsandbytes for Jetson
#---

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG BITSANDBYTES_VERSION \
    FORCE_BUILD=off \
    BITSANDBYTES_REPO \
    BITSANDBYTES_BRANCH \
    CUDA_INSTALLED_VERSION \
    CUDA_MAKE_LIB 

COPY build.sh install.sh /tmp/bitsandbytes/

RUN /tmp/bitsandbytes/install.sh || /tmp/bitsandbytes/build.sh

# ARG BASE_IMAGE
# ARG BUILD_IMAGE

# FROM ${BUILD_IMAGE} as builder
# FROM ${BASE_IMAGE} as runtime

# COPY --from=builder /opt/bitsandbytes*.whl /opt/

# RUN set -ex \
#     && pip3 install --no-cache-dir --verbose \
#         scipy \
#         /opt/bitsandbytes*.whl \
#     \
#     && pip3 show bitsandbytes \
#     && python3 -c 'import bitsandbytes'
