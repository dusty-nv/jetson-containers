#---
# name: wyoming-piper
# group: wyoming
# requires: '>=34.1.0'
# depends: [homeassistant-base, cuda, cudnn, python:3.11, piper-tts]
# notes: The `piper-tts` using the `wyoming` protocol for usage with Home Assistant. Based on `https://github.com/home-assistant/addons/tree/master/piper`
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ENV PIP_BREAK_SYSTEM_PACKAGES=1 \
    PIPER_PORT=10200 \
    PIPER_UPDATE_VOICES="true" \
    PIPER_LENGTH_SCALE="1.0" \
    PIPER_NOISE_SCALE="0.667" \
    PIPER_NOISE_W="0.333" \
    PIPER_SPEAKER="0" \
    PIPER_VOICE="en_US-lessac-medium" \
    PIPER_MAX_PROC="1" \
    PIPER_DEBUG="true"

COPY build.sh /tmp/wyoming/build_wyoming-piper.sh

WORKDIR /usr/src

RUN /tmp/wyoming/build_wyoming-piper.sh

EXPOSE ${PIPER_PORT}/tcp

ENTRYPOINT ["/init"]
