#---
# name: whisper_streaming
# group: audio
# depends: [pytorch, torchaudio, faster-whisper]
# requires: '>=34.1.0'
# docs: docs.md
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /opt

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
		  ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

RUN git clone https://github.com/ufal/whisper_streaming.git && \
    cd whisper_streaming && \
    pip3 install --no-cache-dir --verbose librosa soundfile