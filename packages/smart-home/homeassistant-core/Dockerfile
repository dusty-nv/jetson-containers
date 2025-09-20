#---
# name: homeassistant-core
# group: smart-home
# config: config.py
# requires: '>=34.1.0'
# docs: docs.md
# depends: [homeassistant-base, psutil-home-assistant]
# test: [go2rtc_test.sh, sqlite3_test.py, test.py]
# notes: The `homeassistant-core` wheel that's build is saved in `/usr/src/homeassistant`
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG HA_VERSION \
    SQLITE_VERSION

SHELL ["/bin/bash", "-exo", "pipefail", "-c"]

WORKDIR /usr/src

ENV S6_SERVICES_READYTIME=50 \
    S6_SERVICES_GRACETIME=240000 \
    UV_SYSTEM_PYTHON=true \
    UV_NO_CACHE=true \
    LD_PRELOAD="${LD_PRELOAD:-}:/usr/local/lib/libjemalloc.so.2" \
    MALLOC_CONF="background_thread:true,metadata_thp:auto,dirty_decay_ms:20000,muzzy_decay_ms:20000" \
    SQLITE_VERSION="${SQLITE_VERSION}"

COPY *.sh /tmp/homeassistant/

RUN /tmp/homeassistant/build.sh

ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:/usr/local/lib

WORKDIR /config
