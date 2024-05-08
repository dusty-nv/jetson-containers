#---
# name: homeassistant-supervisor
# group: smart-home
# config: config.py
# requires: '>=34.1.0'
# depends: [homeassistant-base, ciso8601, eudev]
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG COSIGN_VERSION \
    SUPERVISOR_VERSION

SHELL ["/bin/bash", "-xo", "pipefail", "-c"]

WORKDIR /usr/src

ENV CRYPTOGRAPHY_OPENSSL_NO_LEGACY=1 \
    S6_SERVICES_GRACETIME=10000 \
    SUPERVISOR_API=http://localhost \
    LD_PRELOAD="${LD_PRELOAD:-}:/usr/local/lib/libjemalloc.so.2" \
    MALLOC_CONF="background_thread:true,metadata_thp:auto"

COPY *.sh /tmp/homeassistant-supervisor/

RUN /tmp/homeassistant-supervisor/build.sh

WORKDIR /
