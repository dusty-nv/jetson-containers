#---
# name: transformers
# config: config.py
# group: llm
# depends: [pytorch, torchvision, huggingface_hub, rust]
# test: [test_version.py, huggingface-benchmark.py]
# docs: docs.md
# notes: https://github.com/huggingface/transformers
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG TRANSFORMERS_PACKAGE \
    TRANSFORMERS_VERSION

ENV TRANSFORMERS_PACKAGE=${TRANSFORMERS_PACKAGE} \
    TRANSFORMERS_VERSION=${TRANSFORMERS_VERSION}

COPY install.sh /tmp/transformers/install.sh
RUN /tmp/transformers/install.sh

COPY huggingface-benchmark.py /usr/local/bin

RUN uv pip show transformers \
    && python3 -c 'import transformers; print(transformers.__version__)'
