#---
# name: silerovad
# group: audio
# depends: [torch, torchaudio, onnxruntime]
# requires: '>=34.1.0'
# docs: docs.md
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# Install required dependencies
RUN apt-get update &&  \
    apt-get install -y portaudio19-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python dependencies in one step
RUN pip install soundfile sounddevice silero-vad

    