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

ARG FORCE_BUILD=off \
    WYOMING_PIPER_VERSION \
    WYOMING_PIPER_BRANCH

ENV PIP_BREAK_SYSTEM_PACKAGES=1 \
    WYOMING_PIPER_VERSION="$WYOMING_PIPER_VERSION" \
    WYOMING_PIPER_BRANCH="$WYOMING_PIPER_BRANCH" \
    PIPER_CACHE=/data/models/piper \
    PIPER_PORT=10200 \
    PIPER_UPDATE_VOICES="true" \
    PIPER_LENGTH_SCALE="1.0" \
    PIPER_NOISE_SCALE="0.667" \
    PIPER_NOISE_W="0.333" \
    PIPER_SPEAKER="777" \
    PIPER_VOICE="en_US-libritts_r-medium" \
    PIPER_MAX_PROC="1" \
    PIPER_DEBUG="true"

COPY rootfs/ /
COPY *.sh *.diff /tmp/wyoming/piper/

WORKDIR /usr/src

RUN /tmp/wyoming/piper/install.sh || /tmp/wyoming/piper/build.sh

EXPOSE ${PIPER_PORT}/tcp

HEALTHCHECK --start-period=10m \
    CMD echo '{ "type": "describe" }' \
        | nc -w 1 localhost ${PIPER_PORT} \
        | grep -q "piper" \
        || exit 1
