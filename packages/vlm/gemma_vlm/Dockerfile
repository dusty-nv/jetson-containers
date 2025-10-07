#---
# name: gemma_vlm
# alias: gemma_vlm
# group: vlm
# depends: [transformers, bitsandbytes]
# requires: '>=35'
# test: test.py
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

COPY *.whl /tmp/
COPY *.py /opt/

RUN uv pip install /tmp/*.whl && \
    chmod +x /opt/*.py
