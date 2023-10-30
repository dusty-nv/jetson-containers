#---
# name: local_llm
# group: llm
# depends: [mlc, jetson-utils, riva-client:python]
# requires: '>=34.1.0'
#---
# depends: [mlc:dev, awq:dev]
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /opt/local_llm/local_llm

COPY requirements.txt .
RUN pip3 install --no-cache-dir --verbose -r requirements.txt

COPY *.py ./

COPY agents agents
COPY models models
COPY plugins plugins
COPY utils utils
COPY vision vision
COPY web web

ENV PYTHONPATH=${PYTHONPATH}:/opt/local_llm

WORKDIR /
