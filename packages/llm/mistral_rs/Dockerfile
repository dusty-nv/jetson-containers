#---
# name: mistral_rs
# alias: mistralrs
# group: llm
# config: config.py
# depends: [cuda, rust]
# requires: '>=36.3.0'
# test: test.sh
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE} as builder

ARG MISTRALRS_REPO \
    MISTRALRS_BRANCH

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
              libssl-dev

WORKDIR /mistralrs

RUN git clone --branch=${MISTRALRS_BRANCH} --depth=1 https://github.com/${MISTRALRS_REPO} /mistralrs

ARG FEATURES="cuda flash-attn"
RUN cargo build --release --features "${FEATURES}"

FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
              libssl-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

RUN mkdir -p ~/.cache && \
    ln -s /data/models/huggingface ~/.cache/huggingface

COPY --from=builder /mistralrs/target/release/mistralrs-bench /mistralrs/target/release/mistralrs-server /usr/local/bin/
RUN chmod +x /usr/local/bin/mistralrs-bench /usr/local/bin/mistralrs-server
