#---
# name: faiss_lite
# group: vectordb
# depends: [pytorch, cuda-python, faiss]
# test: test.sh
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

COPY CMakeLists.txt faiss_lite.cu test.cu faiss_lite.py benchmark.py /opt/faiss_lite/

RUN mkdir /opt/faiss_lite/build && \
    cd /opt/faiss_lite/build && \
    cmake \
	-DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES} \
	../ && \
    make

ENV PYTHONPATH=${PYTHONPATH}:/opt/faiss_lite

