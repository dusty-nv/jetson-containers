#---
# name: auto_awq
# group: llm
# config: config.py
# depends: [transformers]
# requires: '>=36'
# test: test.py
#---
ARG BASE_IMAGE
ARG BUILD_IMAGE

FROM ${BUILD_IMAGE} as builder
FROM ${BASE_IMAGE} as runtime

COPY --from=builder /opt/autoawq_kernels*.whl /opt/
COPY --from=builder /opt/autoawq*.whl /opt/

RUN set -ex \
    && pip3 install --no-cache-dir --verbose \
        /opt/autoawq_kernels*.whl \
        /opt/autoawq*.whl \
    && pip3 show autoawq \
    && python3 -c 'import awq'
