#---
# name: faster-whisper:speaches
# group: audio
# depends: [ctranslate2:master,faster-whisper, python:3.12, numpy]
# requires: '>=34.1.0'
# docs: docs.md
# notes: run with cuda12.6
#---

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# Define port argument with default value
ARG PORT=8000

WORKDIR /opt
COPY start-speaches-server.sh ./start-speaches-server.sh

RUN pip3 install faster-whisper \
    && apt-get clean \
    && apt-get update \
    && apt-get install -y libsndfile1-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/* \
    && git clone https://github.com/speaches-ai/speaches

# Expose port using the ARG value
EXPOSE ${PORT}

# Pass the port to the entrypoint script
ENV SERVER_PORT=${PORT}

ENTRYPOINT ["/opt/start-speaches-server.sh"]
