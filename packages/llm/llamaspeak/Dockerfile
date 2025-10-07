#---
# name: llamaspeak
# group: llm
# depends: [riva-client:python, numpy]
# requires: '>=34.1.0'
# docs: docs.md
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /opt/llamaspeak

COPY requirements.txt .
RUN uv pip install -r requirements.txt

COPY *.py ./
COPY static/ static
COPY templates/ templates

WORKDIR /

CMD /bin/bash -c "cd /opt/llamaspeak && python3 chat.py"
