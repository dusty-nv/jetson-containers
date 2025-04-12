#---
# name: sound-utils
# group: multimedia
# description: Sound utilities for working with audio 
# depends: [python, torch, torchaudio]
# requires: '>=34.1.0'
# docs: docs.md
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# Install required dependencies
RUN apt-get update &&  \
    apt-get install -y portaudio*-dev python3-pip && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python dependencies in one step
RUN pip install soundfile sounddevice

    