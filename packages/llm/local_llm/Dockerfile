#---
# name: local_llm
# group: llm
# depends: [awq, torch2trt, mlc:dev]
# requires: '>=34.1.0'
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /opt/local_llm/local_llm

COPY requirements.txt .
RUN pip3 install --no-cache-dir --verbose -r requirements.txt

COPY *.py ./

ENV PYTHONPATH=${PYTHONPATH}:/opt/local_llm

WORKDIR /
