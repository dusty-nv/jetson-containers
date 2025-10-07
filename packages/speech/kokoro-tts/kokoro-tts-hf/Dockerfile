#---
# name: kokoro-tts:hf
# group: audio
# depends: [transformers, torchaudio]
# test: [test.py]
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

RUN apt-get update && \
    apt-get install -y libespeak-ng-dev portaudio19-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# pin these versions because of an infinite build loop with blis
#  https://github.com/explosion/cython-blis/issues/117#issuecomment-2660947800
#  https://github.com/explosion/spaCy/issues/13293
#  https://github.com/explosion/spaCy/issues/13649

ENV BLIS_ARCH="generic"

RUN uv pip install loguru IPython
RUN git clone https://github.com/hexgrad/misaki /opt/misaki && \
    cd /opt/misaki && \
    uv pip install -e .

RUN git clone https://github.com/hexgrad/kokoro /opt/kokoro && \
    cd /opt/kokoro && \
    uv pip install -e .
