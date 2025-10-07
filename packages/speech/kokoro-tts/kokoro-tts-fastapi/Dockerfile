#---
# name: kokoro-tts:fastapi
# group: audio
# depends: [kokoro-tts:hf, ffmpeg]
# test: test.py
# docs: docs.md
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG KOKORO_ROOT="/opt/kokoro-fastapi"
ARG KORORO_MODELS="${KOKORO_ROOT}/api/src/models/v1_0"

# Install astral-uv
#RUN curl -LsSf https://astral.sh/uv/install.sh | sh
#ENV PATH="/root/.local/bin:$PATH"

# Clone repository
RUN git clone https://github.com/remsky/Kokoro-FastAPI ${KOKORO_ROOT} && \
    cd ${KOKORO_ROOT} && \
    uv pip install -e .

# Create models directory, download model, and restructure directories
RUN mkdir -p ${KORORO_MODELS} && \
    cd ${KOKORO_ROOT}/docker/scripts && \
    python3 ./download_model.py --output /tmp/models && \
    mv /tmp/models/kokoro-v1_0.pth ${KORORO_MODELS} && \
    mv /tmp/models/config.json ${KORORO_MODELS} && \
    rm -rf /tmp/models

# Expose port
EXPOSE 8880

ENV USE_GPU=true \
    USE_ONNX=false \
    PYTHONPATH="$KOKORO_ROOT:$KOKORO_ROOT/api:$PYTHONPATH" \
    MODEL_DIR="$KOKORO_ROOT/api/src/models" \
    VOICES_DIR="$KOKORO_ROOT/api/src/voices/v1_0" \
    WEB_PLAYER_PATH="$KOKORO_ROOT/web" \
    KOKORO_ROOT="$KOKORO_ROOT"

COPY start-gpu.sh ${KOKORO_ROOT}
RUN chmod +x ${KOKORO_ROOT}/start-gpu.sh

ENTRYPOINT ["/opt/kokoro-fastapi/start-gpu.sh"]
