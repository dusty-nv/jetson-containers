#---
# name: cuda-python
# category: cuda
# depends: [build-essential, python, numpy]
# test: test.py
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ENV CUDA_HOME=/usr/local/cuda

ENV LANG=en_US.UTF-8 
ENV PYTHONIOENCODING=utf-8
RUN locale-gen en_US en_US.UTF-8 && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8

RUN cd /opt && \
    git clone --depth=1 https://github.com/NVIDIA/cuda-python && \
    cd cuda-python && \
    sed 's|^numpy>=.*|numpy|g' -i requirements.txt && \
    pip3 install --no-cache-dir --verbose -r requirements.txt && \
    python3 setup.py bdist_wheel --verbose && \
    ls dist
