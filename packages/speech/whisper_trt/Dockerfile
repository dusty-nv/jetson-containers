#---
# name: whisper_trt
# group: audio
# depends: [pytorch, torch2trt, onnxruntime]
# requires: '>=36'
# test: test.py
# notes: TensorRT optimized Whisper ASR from https://github.com/NVIDIA-AI-IOT/whisper_trt
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ADD https://api.github.com/repos/NVIDIA-AI-IOT/whisper_trt/git/refs/heads/main /tmp/whisper_trt_version.json

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

RUN uv pip install openai-whisper && \
    git clone https://github.com/NVIDIA-AI-IOT/whisper_trt /opt/whisper_trt && \
    cd /opt/whisper_trt && \
    uv pip install --no-build-isolation -e . && \
    uv pip install onnxruntime && \
    mkdir -p ~/.cache && \
    ln -s /data/models/whisper ~/.cache/whisper && \
    ln -s /data/models/whisper ~/.cache/whisper_trt

