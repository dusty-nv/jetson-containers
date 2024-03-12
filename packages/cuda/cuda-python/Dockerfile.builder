#---
# name: cuda-python:builder
# group: cuda
# config: config.py
# requires: '>=34.1.0'
# depends: [cuda, numpy]
# test: [test_driver.py, test_runtime.py]
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG CUDA_PYTHON_VERSION

RUN git clone --branch ${CUDA_PYTHON_VERSION} --depth=1 https://github.com/NVIDIA/cuda-python && \
    cd cuda-python && \
    sed 's|^numpy.=.*|numpy|g' -i requirements.txt && \
    sed 's|^numba.=.*|numba|g' -i requirements.txt && \
    pip3 install --no-cache-dir --verbose -r requirements.txt && \
    python3 setup.py bdist_wheel --verbose && \
    cp dist/cuda*.whl /opt && \
    cd ../ && \
    rm -rf cuda-python
    
RUN pip3 install --no-cache-dir --verbose /opt/cuda*.whl && \
    pip3 show cuda-python && python3 -c 'import cuda; print(cuda.__version__)'