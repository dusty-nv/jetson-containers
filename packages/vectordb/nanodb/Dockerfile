#---
# name: nanodb
# group: vectordb
# depends: [pytorch, transformers, faiss:lite, torch2trt]
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /opt/nanodb/nanodb

COPY requirements.txt .
RUN pip3 install --no-cache-dir --verbose -r requirements.txt

COPY *.py ./

ENV PYTHONPATH=${PYTHONPATH}:/opt/nanodb

WORKDIR /
