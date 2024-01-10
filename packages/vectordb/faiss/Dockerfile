#---
# name: faiss
# alias: faiss:main
# group: vectordb
# config: config.py
# depends: [cuda, numpy, cmake]
# test: [test.py]
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /opt

ARG CUDA_ARCHITECTURES

ARG FAISS_REPO=facebookresearch/faiss
ARG FAISS_BRANCH=main

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
		  libopenblas-dev \
		  swig \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean
    
# # workaround for 'Could NOT find Python3 (missing: Python3_NumPy_INCLUDE_DIRS Development'
# # update-alternatives --install /usr/bin/python python /usr/bin/python3 1 && \
# RUN apt purge -y python3.9 libpython3.9* || echo "python3.9 not found, skipping removal" && \
#     ls -ll /usr/bin/python*

ADD https://api.github.com/repos/${FAISS_REPO}/git/refs/heads/${FAISS_BRANCH} /tmp/faiss_version.json

RUN git clone --branch=${FAISS_BRANCH} --depth=1 https://github.com/${FAISS_REPO}

RUN mkdir faiss/build && \
    cd faiss/build && \
    cmake \
     -DFAISS_ENABLE_GPU=ON \
	-DFAISS_ENABLE_PYTHON=ON \
	-DFAISS_ENABLE_RAFT=OFF \
	-DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES} \
	-DPYTHON_EXECUTABLE=/usr/bin/python3 \
	../ && \
    make -j$(nproc) faiss && \
    make install
    
RUN cd faiss/build && \
    make demo_ivfpq_indexing && \
    make demo_ivfpq_indexing_gpu 
    
RUN cd faiss/build && \
    make -j$(nproc) swigfaiss
    
RUN cd faiss/build/faiss/python && \
    python3 setup.py --verbose bdist_wheel && \
    cp dist/faiss*.whl /opt
    
RUN pip3 install --no-cache-dir --verbose /opt/faiss*.whl

WORKDIR /

RUN pip3 show faiss && python3 -c 'import faiss'