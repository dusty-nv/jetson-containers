#---
# name: mistral_rs
# alias: mistralrs
# group: llm
# config: config.py
# depends: [cuda, cudnn, rust]
# requires: '>=36.3.0'
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE} as builder

ARG MISTRALRS_REPO \
    MISTRALRS_BRANCH

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /mistralrs

RUN git clone --branch=${MISTRALRS_BRANCH} --depth=1 https://github.com/${MISTRALRS_REPO} /mistralrs

ARG FEATURES="cuda flash-attn"
RUN cargo build --release --features "${FEATURES}"

FROM ${BASE_IMAGE}

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p ~/.cache && \
    ln -s /data/models/huggingface ~/.cache/huggingface

COPY --from=builder /mistralrs/target/release/mistralrs-bench /usr/local/bin/mistralrs-bench
RUN chmod +x /usr/local/bin/mistralrs-bench
COPY --from=builder /mistralrs/target/release/mistralrs-server /usr/local/bin/mistralrs-server
RUN chmod +x /usr/local/bin/mistralrs-server
