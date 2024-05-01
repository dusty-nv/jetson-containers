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

ADD https://api.github.com/repos/${TTS_REPO}/git/refs/heads/dev /tmp/tts_version.json

RUN git clone https://github.com/${TTS_REPO} ${TTS_PATH} && \
    cd ${TTS_PATH} && \
    sed 's|^torch.*||' -i requirements.txt && \
    sed 's|^transformers.*||' -i requirements.txt && \
    sed 's|answer[[:space:]]*=.*|answer=\"y\"|' -i TTS/utils/manage.py && \
    cat requirements.txt && \
    \
    pip3 install --ignore-installed --no-cache-dir blinker && \
    pip3 install --no-cache-dir --verbose -r requirements.txt && \
    \
    python3 setup.py --verbose bdist_wheel && \
    cp dist/TTS*.whl /opt && \
    \
    mkdir -p /root/.local/share && \
    ln -s /data/models/tts /root/.local/share/tts

RUN pip3 install --no-cache-dir --verbose /opt/TTS*.whl && \
    pip3 show TTS && python3 -c 'import TTS'