#---
# name: speaches
# group: audio
# depends: [faster-whisper, piper1-tts, kokoro-tts:onnx, python, nodejs, onnxruntime]
# requires: '>=34.1.0'
# docs: docs.md
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

RUN set -ex && \
    git clone https://github.com/speaches-ai/speaches /opt/speaches && \
    cd /opt/speaches && \
    sed -i 's|requires-python.*|requires-python = ">=3.10"|g' pyproject.toml && \
    sed -i 's|"faster-whisper.*",|"faster-whisper",|g' pyproject.toml && \
    sed -i 's|"ctranslate2.*",|"ctranslate2",|g' pyproject.toml && \
    sed -i 's|"kokoro-onnx.*",|"kokoro-onnx",|g' pyproject.toml && \
    sed -i 's|"numpy.*",|"numpy",|g' pyproject.toml && \
    sed -i 's/==/>=/g' requirements.txt && \
    sed -i 's/~=/>=/g' requirements.txt && \
    echo "👁️######## speaches - pyproject.toml ########👁️" && \
    cat pyproject.toml && \
    echo "👁️^^^^^^^^ speaches - pyproject.toml ^^^^^^^^👁️" && \
    uv pip install '.[ui]' && \
    cp /opt/speaches/model_aliases.json / && \
    uv pip uninstall onnxruntime onnxruntime-gpu || true

# Patch to skip UI mount if dist/ missing
COPY 0001-Conditionally-mount-realtime-console-dist.patch /tmp/
RUN cd /opt/speaches && git apply /tmp/0001-Conditionally-mount-realtime-console-dist.patch && \
    cat /opt/speaches/src/speaches/main.py

RUN python3 -c "from importlib.metadata import version; \
print('🌃 faster-whisper:', version('faster-whisper'), \
'\n🌃 kokoro-onnx:', version('kokoro-onnx'), \
'\n🌃 numpy:', version('numpy'))"

ENV PYTHONPATH="/opt/speaches/src:${PYTHONPATH}"


RUN uv pip install openai aiostream && \
    python3 -c "import speaches; print('✅ all modules import OK')"

# 4. Confirm ONNX Runtime is correct
RUN python3 -c "import onnxruntime; print('☑️✅ ONNX Runtime version:', onnxruntime.__version__); print('✅ EPs:', onnxruntime.get_available_providers())"

# Set app config
ARG PORT=8000
ENV PORT=${PORT}
EXPOSE ${PORT}

# 5. Add launcher
COPY start-server.sh /opt/speaches/
RUN chmod +x /opt/speaches/start-server.sh

# 6. Set default working directory
WORKDIR /opt/speaches

ENTRYPOINT ["/opt/speaches/start-server.sh"]
