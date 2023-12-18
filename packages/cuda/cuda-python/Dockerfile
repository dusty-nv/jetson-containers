#---
# name: cuda-python
# group: cuda
# config: config.py
# requires: '>=34.1.0'
# depends: [cuda, build-essential, python, numpy]
# test: [test_driver.py, test_runtime.py]
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG CUDA_PYTHON_VERSION
ENV CUDA_HOME=/usr/local/cuda

ENV LANG=en_US.UTF-8 
ENV PYTHONIOENCODING=utf-8

RUN apt-get update && \
    apt-get install -y --no-install-recommends locales \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean
    
RUN locale-gen en_US en_US.UTF-8 && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8

RUN git clone --branch ${CUDA_PYTHON_VERSION} --depth=1 https://github.com/NVIDIA/cuda-python && \
    cd cuda-python && \
    sed 's|^numpy.=.*|numpy|g' -i requirements.txt && \
    sed 's|^numba.=.*|numba|g' -i requirements.txt && \
    pip3 install --no-cache-dir --verbose -r requirements.txt && \
    python3 setup.py bdist_wheel --verbose && \
    cp dist/cuda*.whl /opt && \
    pip3 install --no-cache-dir --verbose /opt/cuda*.whl && \
    cd ../ && \
    rm -rf cuda-python
    
RUN pip3 show cuda-python && python3 -c 'import cuda; print(cuda.__version__)'