#---
# name: cupy
# group: cuda
# config: config.py
# depends: [cuda, python, numpy]
# test: test.py
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG CUPY_VERSION=main
ARG CUPY_NVCC_GENERATE_CODE="arch=compute_53,code=sm_53;arch=compute_62,code=sm_62;arch=compute_72,code=sm_72;arch=compute_87,code=sm_87"

RUN git clone --branch ${CUPY_VERSION} --depth 1 --recursive https://github.com/cupy/cupy cupy && \
    cd cupy && \
    pip3 install --no-cache-dir fastrlock && \
    python3 setup.py bdist_wheel --verbose && \
    cp dist/cupy*.whl /opt && \
    pip3 install --no-cache-dir --verbose /opt/cupy*.whl && \
    cd ../ && \
    rm -rf cupy
    
# the pip3 way used above catches dependency problems earlier
# MAKEFLAGS="-j$(nproc)" python3 setup.py install --verbose && \