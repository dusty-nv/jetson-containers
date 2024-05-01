#---
# name: torchaudio
# group: pytorch
# config: config.py
# depends: [pytorch, cmake]
# test: test.py
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG TORCHAUDIO_VERSION

COPY build.sh /tmp/build_torchaudio.sh

RUN pip3 install --no-cache-dir --verbose pysoundfile && \
    pip3 install --no-cache-dir torchaudio==${TORCHAUDIO_VERSION} || /tmp/build_torchaudio.sh
