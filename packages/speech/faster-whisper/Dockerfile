#---
# name: faster-whisper
# group: audio
# depends: [ctranslate2, onnxruntime, huggingface_hub, numpy]
# requires: '>=34.1.0'
# test: test.py
# notes: Faster Whisper transcription with CTranslate2 (https://github.com/SYSTRAN/faster-whisper)
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG FORCE_BUILD=off \
    FASTER_WHISPER_VERSION \
    FASTER_WHISPER_BRANCH

COPY *.sh /tmp/faster-whisper/

RUN /tmp/faster-whisper/install.sh || /tmp/faster-whisper/build.sh
