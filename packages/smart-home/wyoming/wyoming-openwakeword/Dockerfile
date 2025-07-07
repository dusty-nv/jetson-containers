#---
# name: wyoming-openwakeword
# group: wyoming
# requires: '>=34.1.0'
# docs: docs.md
# config: config.py
# depends: [python, numpy, homeassistant-base]
# test: test.sh
# notes: The `openWakeWord` using the `wyoming` protocol for usage with Home Assistant. Based on `https://github.com/home-assistant/addons/blob/master/openwakeword/Dockerfile` and `https://github.com/rhasspy/wyoming-openwakeword`
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG FORCE_BUILD=off \
    WYOMING_OPENWAKEWORD_VERSION \
    WYOMING_OPENWAKEWORD_BRANCH

ENV PIP_BREAK_SYSTEM_PACKAGES=1 \
    WYOMING_OPENWAKEWORD_VERSION="$WYOMING_OPENWAKEWORD_VERSION" \
    WYOMING_OPENWAKEWORD_BRANCH="$WYOMING_OPENWAKEWORD_BRANCH" \
    OPENWAKEWORD_PORT=10400 \
    OPENWAKEWORD_THRESHOLD=0.5 \
    OPENWAKEWORD_TRIGGER_LEVEL=1 \
    OPENWAKEWORD_PRELOAD_MODEL="ok_nabu" \
    OPENWAKEWORD_CUSTOM_MODEL_DIR="/share/openwakeword" \
    OPENWAKEWORD_DEBUG=true

WORKDIR /usr/src

COPY rootfs/ /
COPY models/ ${OPENWAKEWORD_CUSTOM_MODEL_DIR}/
COPY *.sh /tmp/wyoming/openwakeword/

RUN /tmp/wyoming/openwakeword/install.sh || /tmp/wyoming/openwakeword/build.sh

HEALTHCHECK --start-period=10m \
    CMD echo '{ "type": "describe" }' \
        | nc -w 1 localhost ${OPENWAKEWORD_PORT} \
        | grep -iq "openWakeWord" \
        || exit 1

EXPOSE ${OPENWAKEWORD_PORT}/tcp
