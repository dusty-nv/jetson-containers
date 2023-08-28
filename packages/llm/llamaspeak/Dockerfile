#---
# name: llamaspeak
# group: llm
# depends: [riva-client:python, numpy]
# requires: '>=34.1.0'
# docs: docs.md
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

COPY requirements.txt /opt/llamaspeak/
RUN pip3 install --no-cache-dir --verbose -r /opt/llamaspeak/requirements.txt

COPY *.py /opt/llamaspeak/
COPY static/ /opt/llamaspeak/static
COPY templates/ /opt/llamaspeak/templates