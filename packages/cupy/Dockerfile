#
# name: cupy
# category: cuda
# config: config.py
# depends: [python, numpy]
# test: test.py
#
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG CUPY_VERSION=main
ARG CUPY_NVCC_GENERATE_CODE="arch=compute_53,code=sm_53;arch=compute_62,code=sm_62;arch=compute_72,code=sm_72;arch=compute_87,code=sm_87"

RUN git clone --branch ${CUPY_VERSION} --depth 1 --recursive https://github.com/cupy/cupy cupy && \
    cd cupy && \
    pip3 install --no-cache-dir fastrlock && \
    python3 setup.py install --verbose && \
    cd ../ && \
    rm -rf cupy
