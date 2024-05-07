#---
# name: bitsandbytes
# group: llm
# requires: '==35.*'
# config: config.py
# depends: [transformers]
# test: test.py
# notes: fork of https://github.com/TimDettmers/bitsandbytes for Jetson
#---
ARG BASE_IMAGE
ARG BUILD_IMAGE

FROM ${BUILD_IMAGE} as builder
FROM ${BASE_IMAGE} as runtime

COPY --from=builder /opt/bitsandbytes*.whl /opt/

RUN set -ex \
    && pip3 install --no-cache-dir --verbose \
        scipy \
        /opt/bitsandbytes*.whl \
    \
    && pip3 show bitsandbytes \
    && python3 -c 'import bitsandbytes'
