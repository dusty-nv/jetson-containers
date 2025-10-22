#---
# name: local_llm
# group: llm
# config: config.py
# depends: [nanodb, mlc, riva-client:python, jetson-inference, torch2trt, onnxruntime]
# requires: '>=34.1.0'
# docs: docs.md
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /opt/local_llm/local_llm

ENV PYTHONPATH=${PYTHONPATH}:/opt/local_llm \
    SSL_KEY=/etc/ssl/private/localhost.key.pem \
    SSL_CERT=/etc/ssl/private/localhost.cert.pem

COPY requirements.txt .

RUN uv pip install --reinstall blinker && \
    uv pip install -r requirements.txt && \
    openssl req \
	-new \
	-newkey rsa:4096 \
	-days 3650 \
	-nodes \
	-x509 \
	-keyout ${SSL_KEY} \
	-out ${SSL_CERT} \
	-subj '/CN=localhost'

COPY *.py ./

COPY agents agents
COPY chat chat
COPY models models
COPY plugins plugins
COPY test test
COPY utils utils
COPY vision vision
COPY web web

WORKDIR /
