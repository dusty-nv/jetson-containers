#---
# name: wyoming-openwakeword
# group: wyoming
# requires: '>=34.1.0'
# docs: docs.md
# config: config.py
# depends: [homeassistant-base, python:3.11]
# notes: The `openWakeWord` using the `wyoming` protocol for usage with Home Assistant. Based on `https://github.com/home-assistant/addons/blob/master/openwakeword/Dockerfile`
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG OPENWAKEWORD_VERSION

ENV PIP_BREAK_SYSTEM_PACKAGES=1 \
    OPENWAKEWORD_PORT=10400 \
    OPENWAKEWORD_THRESHOLD=0.5 \
    OPENWAKEWORD_TRIGGER_LEVEL=1 \
    OPENWAKEWORD_PRELOAD_MODEL="ok_nabu" \
    OPENWAKEWORD_DEBUG=true

COPY build.sh /tmp/wyoming/build_openwakeword.sh
RUN cd /usr/src && /tmp/wyoming/build_openwakeword.sh

# FIXME: For some reason below HEALTHCHECK is not playing nicelly with my CI/CD env
# instead of just declaring itself it executes the CMD...
# HEALTHCHECK --start-period=10m CMD echo '{ "type": "describe" }' | nc -w 1 localhost ${OPENWAKEWORD_PORT} | grep -iq "openWakeWord" || exit 1

EXPOSE ${OPENWAKEWORD_PORT}/tcp

ENTRYPOINT ["/init"]
