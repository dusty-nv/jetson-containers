#---
# name: wyoming-piper
# group: wyoming
# config: config.py
# depends: [homeassistant-base, cuda, cudnn, python:3.11, piper-tts]
# test: test.sh
# requires: '>=34.1.0'
# docs: docs.md
# notes: The `piper-tts` using the `wyoming` protocol for usage with Home Assistant. Based on `https://github.com/home-assistant/addons/tree/master/piper` and `https://github.com/rhasspy/wyoming-piper`
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG WYOMING_PIPER_VERSION

# Specify the shell with desired options
SHELL ["/bin/bash", "-euxo", "pipefail"]

ENV PIP_BREAK_SYSTEM_PACKAGES=1 \
    WYOMING_PIPER_VERSION="$WYOMING_PIPER_VERSION" \
    PIPER_CACHE=/data/models/piper \
    PIPER_PORT=10200 \
    PIPER_UPDATE_VOICES="true" \
    PIPER_LENGTH_SCALE="1.0" \
    PIPER_NOISE_SCALE="0.667" \
    PIPER_NOISE_W="0.333" \
    PIPER_SPEAKER="0" \
    PIPER_VOICE="en_US-lessac-high" \
    PIPER_MAX_PROC="1" \
    PIPER_DEBUG="true"

COPY rootfs/ /
COPY build.sh /tmp/wyoming/build_wyoming-piper.sh
COPY *.diff /tmp/wyoming/

WORKDIR /usr/src

RUN /tmp/wyoming/build_wyoming-piper.sh

EXPOSE ${PIPER_PORT}/tcp

HEALTHCHECK --start-period=10m \
    CMD echo '{ "type": "describe" }' \
        | nc -w 1 localhost ${PIPER_PORT} \
        | grep -q "piper" \
        || exit 1

LABEL \
    io.hass.type="addon" \
    io.hass.name="piper-tts" \
    io.hass.version="${WYOMING_PIPER_VERSION}" \
    io.hass.arch="${BUILD_ARCH}" \
    io.hass.machine="qemuarm-64"