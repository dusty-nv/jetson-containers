#---
# name: local_llm
# group: llm
# depends: [nanodb, mlc, riva-client:python, jetson-inference, torch2trt]
# requires: '>=34.1.0'
# docs: docs.md
#---
# depends: [mlc:dev, awq:dev]
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /opt/local_llm/local_llm

COPY requirements.txt .

RUN pip3 install --ignore-installed --no-cache-dir blinker
RUN pip3 install --no-cache-dir --verbose -r requirements.txt

COPY *.py ./

COPY agents agents
COPY chat chat
COPY models models
COPY plugins plugins
COPY utils utils
COPY vision vision
COPY web web

ENV PYTHONPATH=${PYTHONPATH}:/opt/local_llm

ENV SSL_KEY=/etc/ssl/private/localhost.key.pem
ENV SSL_CERT=/etc/ssl/private/localhost.cert.pem

RUN openssl req \
	-new \
	-newkey rsa:4096 \
	-days 3650 \
	-nodes \
	-x509 \
	-keyout ${SSL_KEY} \
	-out ${SSL_CERT} \
	-subj '/CN=localhost'
  
WORKDIR /
