#---
# name: homeassistant-core
# group: smart-home
# config: config.py
# requires: '>=34.1.0'
# docs: docs.md
# depends: [homeassistant-base, ciso8601, psutil-home-assistant, ffmpeg]
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

RUN /tmp/homeassistant/build.sh \
    && ln -s /data/homeassistant /config \
    && curl -L https://github.com/AlexxIT/go2rtc/releases/download/v1.9.7/go2rtc_linux_arm64 --output /bin/go2rtc \
    && chmod +x /bin/go2rtc \
    && go2rtc --version

ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:/usr/local/lib

WORKDIR /config
