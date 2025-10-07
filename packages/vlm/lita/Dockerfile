#---
# name: lita
# group: vlm
# docs: docs.md
# depends: [pytorch, transformers, bitsandbytes, flash-attention, decord2]
# requires: '>=34.1.0'
# test: test.py
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG LITA_ROOT=/opt/LITA \
    LITA_REPO=NVlabs/LITA \
    LITA_BRANCH=main

ADD https://api.github.com/repos/${LITA_REPO}/git/refs/heads/${LITA_BRANCH} /tmp/lita_version.json

RUN git clone --branch=${LITA_BRANCH} --depth=1 https://github.com/${LITA_REPO} ${LITA_ROOT} && \
    cd ${LITA_ROOT} && \
    #sed 's|"transformers.*"|"transformers<=4.35.2"|' -i pyproject.toml && \
    #sed 's|"accelerate.*",||' -i pyproject.toml && \
    #sed 's|"deepspeed.*",||' -i pyproject.toml && \
    sed 's|"bitsandbytes.*",||' -i pyproject.toml && \
    #sed 's|"peft.*",||' -i pyproject.toml && \
    #sed 's|"timm.*",||' -i pyproject.toml && \
    sed -i 's/==/>=/g' pyproject.toml && \
    sed -i 's/~=/>=/g' pyproject.toml && \
    sed -i 's/decord/decord2/g' pyproject.toml && \
    cat pyproject.toml

COPY benchmark.py ${LITA_ROOT}/lita/serve/

RUN uv pip install -e ${LITA_ROOT}
