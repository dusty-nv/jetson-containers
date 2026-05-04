#---
# name: llama-vision
# alias: llama_vision
# group: vlm
# depends: [transformers, bitsandbytes]
# buildkit_device: nvidia.com/gpu=all
# requires: '>=35'
# test: test.py
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

COPY *.whl /tmp/
COPY *.py /opt/

RUN uv pip install /tmp/*.whl && \
    chmod +x /opt/*.py

