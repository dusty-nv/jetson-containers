#---
# name: homeassistant-core
# group: smart-home
# config: config.py
# requires: '>=34.1.0'
# docs: docs.md
# depends: [homeassistant-base, ffmpeg]
# test: [test_ciso8601.py, test_psutil_home_assistant.py, test_homeassistant.py]
# notes: The `homeassistant-core` wheel that's build is saved in `/usr/src/homeassistant`
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG HA_BRANCH

ENV S6_SERVICES_READYTIME=50 \
    S6_SERVICES_GRACETIME=240000 \
    UV_SYSTEM_PYTHON=true

WORKDIR /usr/src

COPY *.diff build_*.sh /tmp/

RUN /tmp/build_ciso8601.sh && /tmp/build_homeassistant.sh
RUN ln -s /data/homeassistant /config

WORKDIR /config

ENTRYPOINT ["/init"]
