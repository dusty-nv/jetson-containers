#---
# name: exllama:v1
# alias: exllama
# group: llm
# config: config.py
# depends: [pytorch, huggingface_hub]
# requires: '>=34.1.0'
# test: test.sh
# docs: docs.md
#---
ARG BASE_IMAGE
ARG BUILD_IMAGE

FROM ${BUILD_IMAGE} as builder
FROM ${BASE_IMAGE} as runtime

COPY --from=builder /opt/exllama*.whl /opt/

RUN set -ex \
    && pip3 install --no-cache-dir --verbose /opt/exllama*.whl \
    \
    && pip3 show exllama \
    && python3 -c 'import exllama'
