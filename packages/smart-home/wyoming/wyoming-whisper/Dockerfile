#---
# name: wyoming-whisper
# group: wyoming
# config: config.py
# depends: [cuda, cudastack:standard, python, homeassistant-base, faster-whisper]
# test: test.sh
# requires: '>=34.1.0'
# docs: docs.md
# notes: The `faster-whisper` using the `wyoming` protocol for usage with Home Assistant. Based on `https://github.com/home-assistant/addons/blob/master/whisper/Dockerfile`
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG FORCE_BUILD=off \
    WYOMING_WHISPER_VERSION \
    WYOMING_WHISPER_BRANCH

ENV PIP_BREAK_SYSTEM_PACKAGES=1 \
    WYOMING_WHISPER_VERSION="$WYOMING_WHISPER_VERSION" \
    WYOMING_WHISPER_BRANCH="$WYOMING_WHISPER_BRANCH" \
    WHISPER_PORT=10300 \
    WHISPER_MODEL="tiny-int8" \
    WHISPER_BEAM_SIZE="1" \
    WHISPER_LANGUAGE="en" \
    WHISPER_DEBUG="true" \
    WHISPER_COMPUTE_TYPE="default" \
    WHISPER_INITIAL_PROMPT="" \
    WHISPER_OFFLINE="false"

COPY rootfs/ /
COPY *.sh /tmp/wyoming/whisper/

WORKDIR /

RUN /tmp/wyoming/whisper/install.sh || /tmp/wyoming/whisper/build.sh

EXPOSE ${WHISPER_PORT}/tcp

HEALTHCHECK --start-period=10m \
    CMD echo '{ "type": "describe" }' \
        | nc -w 1 localhost ${WHISPER_PORT} \
        | grep -q "faster-whisper" \
        || exit 1
