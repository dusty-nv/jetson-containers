#---
# name: lita
# group: vlm
# docs: docs.md
# depends: [decord, pytorch, transformers, flash-attention]
# requires: '>=34.1.0'
# test: test.py
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /opt

ARG LITA_REPO=NVlabs/LITA
ARG LITA_BRANCH=main

ADD https://api.github.com/repos/${LITA_REPO}/git/refs/heads/${LITA_BRANCH} /tmp/lita_version.json

RUN git clone --branch=${LITA_BRANCH} --depth=1 https://github.com/${LITA_REPO} lita && \
    cd lita && \
#    sed 's|"deepspeed==0.9.5",.*||' -i pyproject.toml && \
#    sed 's|"transformers.*"|"transformers"|' -i pyproject.toml && \
#    sed 's|"accelerate.*"|"accelerate"|' -i pyproject.toml && \
    sed 's|"bitsandbytes.*",||' -i pyproject.toml && \
    cat pyproject.toml

COPY benchmark.py lita/lita/serve/

RUN cd lita && \
    pip3 wheel --wheel-dir=dist --no-deps --verbose . && \
    cp dist/lita*.whl /opt
    
RUN pip3 install --no-cache-dir --verbose lita*.whl

RUN pip3 install --verbose /opt/torch*.whl
RUN pip3 install --verbose /opt/torchvision*.whl

WORKDIR /
