#---
# name: speaches
# group: audio
# depends: [faster-whisper, piper-tts, kokoro-tts:onnx]
# requires: '>=34.1.0'
# docs: docs.md
# notes: run with cuda12.6
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

RUN apt-get update && \
    apt-get install -y libsndfile1-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/speaches-ai/speaches /opt/speaches && \
    cd /opt/speaches && \
    sed -i 's|requires-python.*|requires-python = ">=3.10"|g' pyproject.toml && \
    sed -i 's|"faster-whisper.*",|"faster-whisper",|g' pyproject.toml && \
    sed -i 's|"ctranslate2.*",|"ctranslate2",|g' pyproject.toml && \
    sed -i 's|"kokoro-onnx.*",|"kokoro-onnx",|g' pyproject.toml && \
    sed -i 's|"numpy.*",|"numpy",|g' pyproject.toml && \
    cat pyproject.toml && \
    pip3 install --verbose -e '.[ui]'

ARG PORT=8000
ENV PORT=${PORT}
EXPOSE ${PORT}

CMD ["python3 -m uvicorn speaches.main:create_app --host 0.0.0.0 --port $PORT --factory"]
