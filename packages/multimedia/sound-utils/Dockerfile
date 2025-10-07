#---
# name: sound-utils
# group: multimedia
# depends: [python, torch, torchaudio]
# requires: '>=34.1.0'
# test: [test.py]
# docs: docs.md
# notes: Sound utilities for working with audio
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# Install required dependencies
RUN apt-get update &&  \
    apt-get install -y portaudio*-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python dependencies in one step
RUN uv pip install soundfile sounddevice

