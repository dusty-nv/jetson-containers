#---
# name: spark-tts
# group: audio
# depends: [transformers]
# requires: ['>=36.1.0', '>=cu126']
# test: [test.py]
# notes: 'Spark-TTS: An Efficient LLM-Based Text-to-Speech Model with Single-Stream Decoupled Speech Tokens – https://github.com/SparkAudio/Spark-TTS'
# docs: docs.md
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG SPARK_TTS_VERSION=main

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        libsndfile1 \
        libsndfile1-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    \
    git clone --branch=${SPARK_TTS_VERSION} https://github.com/SparkAudio/Spark-TTS /opt/spark-tts && \
    cd /opt/spark-tts && \
    sed -i \
        -e 's|^numpy=.*|numpy|g' \
        -e 's|^torchaudio=.*|torchaudio|g' \
        -e 's|^torch=.*|torch|g' \
        -e 's|^transformers=.*|transformers|g' \
        requirements.txt && \
    cat requirements.txt && \
    pip3 install -r requirements.txt
    
COPY inference.py /opt/spark-tts

ENV PYTHONPATH="/opt/spark-tts:${PYTHONPATH}"
