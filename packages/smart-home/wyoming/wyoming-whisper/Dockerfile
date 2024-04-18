#---
# name: wyoming-whisper
# group: wyoming
# requires: '>=34.1.0'
# depends: [homeassistant-base, cuda, cudnn, python:3.11, faster-whisper]
# notes: The `faster-whisper` using the `wyoming` protocol for usage with Home Assistant. Based on `https://github.com/home-assistant/addons/blob/master/whisper/Dockerfile`
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ENV PIP_BREAK_SYSTEM_PACKAGES=1 \
    HF_HUB_CACHE=/share/whisper \
    WHISPER_PORT=10300 \
    WHISPER_MODEL="tiny-int8" \
    WHISPER_BEAM_SIZE="1" \
    WHISPER_LANGUAGE="en" \
    WHISPER_INITIAL_PROMPT="" \
    WHISPER_DEBUG="true"

COPY build.sh /tmp/wyoming/build_wyoming-faster-whisper.sh

WORKDIR /usr/src

RUN /tmp/wyoming/build_wyoming-faster-whisper.sh

EXPOSE ${WHISPER_PORT}/tcp

ENTRYPOINT ["/init"]
