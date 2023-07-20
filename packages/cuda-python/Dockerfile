#---
# name: cuda-python
# category: cuda
# config: config.py
# depends: [build-essential, python, numpy]
# test: [test_driver.py, test_runtime.py]
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG CUDA_PYTHON_VERSION
ENV CUDA_HOME=/usr/local/cuda

ENV LANG=en_US.UTF-8 
ENV PYTHONIOENCODING=utf-8
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