#---
# name: homeassistant-core
# group: smart-home
# config: config.py
# requires: '>=34.1.0'
# docs: docs.md
# depends: [homeassistant-base, ciso8601, psutil-home-assistant, ffmpeg]
# test: test_homeassistant.py
# notes: The `homeassistant-core` wheel that's build is saved in `/usr/src/homeassistant`
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG HA_VERSION

SHELL ["/bin/bash", "-exo", "pipefail", "-c"]

WORKDIR /usr/src

ENV S6_SERVICES_READYTIME=50 \
    S6_SERVICES_GRACETIME=240000 \
    UV_SYSTEM_PYTHON=true \
    LD_PRELOAD="${LD_PRELOAD:-}:/usr/local/lib/libjemalloc.so.2" \
    MALLOC_CONF="background_thread:true,metadata_thp:auto,dirty_decay_ms:20000,muzzy_decay_ms:20000"

COPY *.sh /tmp/homeassistant/

RUN /tmp/homeassistant/build.sh \
    && ln -s /data/homeassistant /config

WORKDIR /config
