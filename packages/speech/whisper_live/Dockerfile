#---
# name: whisper_live
# group: audio
# depends: [faster-whisper]
# requires: '>=34.1.0'
# docs: docs.md
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /opt

RUN git clone https://github.com/collabora/WhisperLive