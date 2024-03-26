#---
# name: auto_gptq
# group: llm
# config: config.py
# depends: [transformers]
# requires: '>=34.1.0'
# test: test.py
# docs: docs.md
#---
ARG BASE_IMAGE
ARG BUILD_IMAGE

FROM ${BUILD_IMAGE} as builder
FROM ${BASE_IMAGE} as runtime

COPY --from=builder /opt/auto_gptq*.whl /opt/

RUN set -ex \
    && pip3 install --no-cache-dir --verbose /opt/auto_gptq*.whl \
    \
    && pip3 show autoawq \
    && python3 -c 'import awq'
