#---
# name: xtts
# group: audio
# depends: [transformers, torchaudio, torch2trt]
# requires: '>=34.1.0'
# test: [test.py, test_stream.py]
# notes: 'Fork of coqui-ai/TTS with support for quantization and TensorRT'
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG TTS_REPO=dusty-nv/TTS
ARG TTS_PATH=/opt/TTS

RUN apt-get update && \
    apt-get install -y --no-install-recommends libsndfile1-dev && \
    rm -rf /var/lib/apt/lists/*  && \
    apt-get clean

ADD https://api.github.com/repos/${TTS_REPO}/git/refs/heads/dev /tmp/tts_version.json

RUN git clone https://github.com/${TTS_REPO} ${TTS_PATH} && \
    cd ${TTS_PATH} && \
    sed -i 's/==/>=/g' requirements.txt && \
    sed -i 's/~=/>=/g' requirements.txt && \
    sed 's|answer[[:space:]]*=.*|answer=\"y\"|' -i TTS/utils/manage.py && \
    cat requirements.txt && \
    \
    uv pip install --reinstall blinker && \
    uv pip install -r requirements.txt && \
    \
    python3 setup.py --verbose bdist_wheel && \
    cp dist/TTS*.whl /opt && \
    \
    mkdir -p /root/.local/share && \
    ln -s /data/models/tts /root/.local/share/tts

# https://github.com/huggingface/transformers/issues/31040
RUN uv pip install --force-reinstall 'transformers<4.41'

RUN uv pip install /opt/TTS*.whl && \
    uv pip show TTS && python3 -c 'import TTS'
