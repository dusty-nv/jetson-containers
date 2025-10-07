#---
# name: piper1-tts
# group: audio
# depends: [onnxruntime]
# test: test.py
# notes: A self-contained version of Piper, the fast and local neural text-to-speech engine (https://github.com/OHF-Voice/piper1-gpl). This version directly embeds espeak-ng and therefore has a GPL license.
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG PIPER_VERSION \
    PIPER_BRANCH=main \
    FORCE_BUILD=off \
    TMP_DIR=/tmp/piper

ENV PIPER_CACHE=/data/models/piper

COPY *.sh ${TMP_DIR}/

RUN uv pip install --no-cache-dir --verbose 'flask>=3,<4' \
    && ${TMP_DIR}/install.sh || ${TMP_DIR}/build.sh
