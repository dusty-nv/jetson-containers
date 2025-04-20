#---
# name: speaches
# group: audio
# depends: [faster-whisper, piper-tts, ffmpeg, numba]
# requires: '>=34.1.0'
# docs: docs.md
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# Create non-root user
RUN useradd -m -u 1000 ubuntu && \
    # Give ubuntu user write permissions to necessary directories
    mkdir -p /home/ubuntu/.cache && \
    chown -R ubuntu:ubuntu /home/ubuntu/.cache && \
    chmod -R 755 /home/ubuntu/.cache

WORKDIR /opt/speaches

# Clone Speaches repository
RUN git clone -b renovate/jetson-arm64 https://github.com/JonnyTran/speaches . && \
    chown -R ubuntu:ubuntu .

# # Modify pyproject.toml for ARM64 compatibility
# RUN sed -i 's/ctranslate2>=4.5.0/ctranslate2>=4.4.0/' pyproject.toml && \
#     sed -i 's/kokoro-onnx>=0.2.2/kokoro-onnx[gpu]>=0.2.2/' pyproject.toml && \
#     sed -i 's/override-dependencies = \["onnxruntime-gpu>=1.20.1 ; platform_machine == '\''x86_64'\''\"/override-dependencies = \["onnxruntime-gpu>=1.20.1 ; platform_machine == '\''x86_64'\''\", "onnxruntime ; platform_machine != '\''aarch64'\''", "onnxruntime-gpu ; platform_machine != '\''aarch64'\''", "piper-phonemize ; platform_machine != '\''aarch64'\''", "piper-tts ; platform_machine != '\''aarch64'\''\", "numpy>=1.24.0,<2.0.0"/' pyproject.toml

# RUN sed -i 's/type ResponseFormat = Literal\["mp3", "flac", "wav", "pcm"\]/type ResponseFormat = Literal\["mp3", "flac", "wav", "pcm", "aac"\]/' src/speaches/routers/speech.py && \
#     sed -i 's/SUPPORTED_RESPONSE_FORMATS = ("mp3", "flac", "wav", "pcm")/SUPPORTED_RESPONSE_FORMATS = ("mp3", "flac", "wav", "pcm", "aac")/' src/speaches/routers/speech.py && \
#     sed -i 's/UNSUPORTED_RESPONSE_FORMATS = ("opus", "aac")/UNSUPORTED_RESPONSE_FORMATS = ("opus")/' src/speaches/routers/speech.py

RUN uv pip install --compile-bytecode --editable ".[ui]"
RUN sed -i 's/np.load(voices_path)/np.load(voices_path.replace(".json", ".bin"))/' /opt/conda/lib/python3.12/site-packages/kokoro_onnx/__init__.py

USER ubuntu

# Create and add entrypoint script
ENV HOME=/home/ubuntu
RUN mkdir -p $HOME/.cache/huggingface/hub
ENV WHISPER__MODEL=Systran/faster-whisper-large-v3
ENV UVICORN_HOST=0.0.0.0
ENV UVICORN_PORT=8000
# https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables#hfhubenablehftransfer
# NOTE: I've disabled this because it doesn't inside of Docker container. I couldn't pinpoint the exact reason. This doesn't happen when running the server locally.
# RuntimeError: An error occurred while downloading using `hf_transfer`. Consider disabling HF_HUB_ENABLE_HF_TRANSFER for better error handling.
ENV HF_HUB_ENABLE_HF_TRANSFER=0
# https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables#donottrack
# https://www.reddit.com/r/StableDiffusion/comments/1f6asvd/gradio_sends_ip_address_telemetry_by_default/
ENV DO_NOT_TRACK=1
EXPOSE 8000
ENV LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1
CMD ["uvicorn", "--factory", "speaches.main:create_app"]
