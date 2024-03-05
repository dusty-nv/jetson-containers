#---
# name: faiss_lite
# group: vectordb
# config: config.py
# depends: [pytorch, cuda-python, faiss]
# test: test.sh
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /opt

ARG CUDA_ARCHITECTURES

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
		  gdb \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean
    
COPY CMakeLists.txt faiss_lite/
COPY faiss_lite.cu faiss_lite/
COPY test.cu faiss_lite/

RUN mkdir faiss_lite/build && \
    cd faiss_lite/build && \
    cmake \
	-DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES} \
	../ && \
    make

COPY faiss_lite.py faiss_lite/
COPY benchmark.py faiss_lite/

ENV PYTHONPATH=${PYTHONPATH}:/opt/faiss_lite
    
WORKDIR /
