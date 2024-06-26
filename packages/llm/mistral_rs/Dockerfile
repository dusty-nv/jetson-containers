#---
# name: mistral_rs
# alias: mistralrs
# group: llm
# config: config.py
# depends: [cuda, cudnn]
# requires: '>=36.3.0'
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE} as builder

ARG MISTRALRS_REPO \
    MISTRALRS_BRANCH

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    libssl-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

RUN curl https://sh.rustup.rs -sSf | bash -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"
RUN rustup update nightly
RUN rustup default nightly
WORKDIR /mistralrs

RUN git clone --branch=${MISTRALRS_BRANCH} --depth=1 https://github.com/${MISTRALRS_REPO} /mistralrs

ARG FEATURES="cuda flash-attn cudnn"
ENV RAYON_NUM_THREADS=4
RUN RUSTFLAGS="-Z threads=4" cargo build --release --workspace --exclude mistralrs-pyo3 --features "${FEATURES}"

FROM ${BASE_IMAGE}

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    libomp-dev \
    ca-certificates \
    libssl-dev \
    curl \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /mistralrs/target/release/mistralrs-bench /usr/local/bin/mistralrs-bench
RUN chmod +x /usr/local/bin/mistralrs-bench
COPY --from=builder /mistralrs/target/release/mistralrs-server /usr/local/bin/mistralrs-server
RUN chmod +x /usr/local/bin/mistralrs-server
CMD mistralrs-server
