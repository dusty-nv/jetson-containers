#---
# name: spark-tts
# group: audio
# depends: [torchaudio, transformers]
# requires: ['>=36.1.0']
# test: [test.sh]
# notes: 'Spark-TTS: An Efficient LLM-Based Text-to-Speech Model with Single-Stream Decoupled Speech Tokens – https://github.com/SparkAudio/Spark-TTS'
# docs: docs.md
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG SPARK_TTS_VERSION=main

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libsndfile1 \
        libsndfile1-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    \
    git clone --branch=${SPARK_TTS_VERSION} https://github.com/SparkAudio/Spark-TTS /opt/spark-tts && \
    cd /opt/spark-tts && \
    sed -i 's/==/>=/g' requirements.txt && \
    sed -i 's/~=/>=/g' requirements.txt && \
    uv pip install -U -r requirements.txt

COPY inference.py /opt/spark-tts

ENV PYTHONPATH="/opt/spark-tts:${PYTHONPATH}" \
    TRANSFORMERS_CACHE=/data/models/huggingface \
    HUGGINGFACE_HUB_CACHE=/data/models/huggingface \
    HF_HOME=/data/models/huggingface

CMD ["python3", "/opt/spark-tts/inference.py"]
