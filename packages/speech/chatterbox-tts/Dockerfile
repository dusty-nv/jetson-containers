#---
# name: chatterbox-tts
# group: audio
# depends: [torch, torchaudio, transformers, numpy, diffusers, sound-utils]
# requires: ['>=36.1.0']
# test: [test.py]
# docs: docs.md
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG CHATTERBOX_TTS_VERSION=master

#pip install librosa pytorch Transformers
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libsndfile1 \
        libsndfile1-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    \
    git clone --branch=${CHATTERBOX_TTS_VERSION} https://github.com/resemble-ai/chatterbox /opt/chatterbox-tts && \
    cd /opt/chatterbox-tts && \
    sed -i 's/==/>=/g' pyproject.toml && \
    sed -i 's/~=/>=/g' pyproject.toml && \
    cat pyproject.toml && \
    pip install -e .

COPY inference.py /opt/chatterbox-tts

RUN cd /opt/chatterbox-tts

WORKDIR /opt/chatterbox-tts

ENV PYTHONPATH="/opt/chatterbox-tts:${PYTHONPATH}" \
    TRANSFORMERS_CACHE=/data/models/huggingface \
    HUGGINGFACE_HUB_CACHE=/data/models/huggingface \
    HF_HOME=/data/models/huggingface

# Use bash as the default command
CMD ["bash"]
