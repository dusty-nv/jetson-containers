#---
# name: spark-tts
# group: audio
# depends: [numpy, torch, torchaudio, transformers]
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
        -e 's|^numpy=.*||g' \
        -e 's|^torchaudio=.*||g' \
        -e 's|^torch=.*||g' \
        -e 's|^transformers=.*||g' \
        requirements.txt && \
    cat requirements.txt && \
    pip3 install -r requirements.txt
    
COPY inference.py /opt/spark-tts

ENV PYTHONPATH="/opt/spark-tts:${PYTHONPATH}"


# make sure it starts downloading the model
RUN timeout 10s python3 /opt/spark-tts/inference.py \
    --text "The Quick brown fox jumps over the lazy dog" \
    --gender "male" --pitch "moderate" --speed "moderate" 2>&1 | tee /tmp/output.log && \
    grep -q "Fetching.*files" /tmp/output.log || \
    (echo "Model download did not start properly" && exit 1)
