#---
# name: speaches
# group: audio
# depends: [faster-whisper, piper-tts, kokoro-tts:onnx]
# requires: '>=34.1.0'
# docs: docs.md
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

RUN apt-get update && \
    apt-get install -y libsndfile1-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY patches/patch_py312_features.py /tmp
COPY patches/patch_py311_features.py /tmp
COPY patches/src/ /tmp/speaches/src/

RUN git clone https://github.com/speaches-ai/speaches /opt/speaches && \
    cd /opt/speaches && \
    sed -i 's|requires-python.*|requires-python = ">=3.10"|g' pyproject.toml && \
    sed -i 's|"faster-whisper.*",|"faster-whisper",|g' pyproject.toml && \
    sed -i 's|"ctranslate2.*",|"ctranslate2",|g' pyproject.toml && \
    sed -i 's|"kokoro-onnx.*",|"kokoro-onnx",|g' pyproject.toml && \
    sed -i 's|"numpy.*",|"numpy",|g' pyproject.toml && \
    cat pyproject.toml
RUN cd /opt/speaches && \
    python3 /tmp/patch_py312_features.py && \
    python3 /tmp/patch_py311_features.py && \
    cp -r /tmp/speaches/src/* /opt/speaches/src && \
    nl /opt/speaches/src/speaches/realtime/response_event_router.py && \
    pip3 install -e '.[ui]'
    #sed -i 's|enable_ui: bool = True|enable_ui: bool = False|g' src/speaches/config.py
    #pip3 install gradio==5.13.0 gradio-client==1.6.0

ARG PORT=8000
ENV PORT=${PORT}
EXPOSE ${PORT}

WORKDIR /opt/speaches

COPY start-server.sh /opt/speaches
RUN chmod +x /opt/speaches/start-server.sh

ENTRYPOINT ["/opt/speaches/start-server.sh"]
