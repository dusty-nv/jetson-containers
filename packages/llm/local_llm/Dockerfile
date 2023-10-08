#---
# name: local_llm
# group: llm
# depends: [mlc:dev]
# requires: '>=34.1.0'
#---
# depends: [mlc:dev, awq:dev]
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /opt/local_llm/local_llm

COPY requirements.txt .
RUN pip3 install --no-cache-dir --verbose -r requirements.txt

COPY *.py ./

ENV PYTHONPATH=${PYTHONPATH}:/opt/local_llm

WORKDIR /
