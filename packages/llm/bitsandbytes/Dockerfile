#
# name: bitsandbytes
# category: llm
# depends: [build-essential, pytorch]
# test: test.py
#
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

RUN pip3 uninstall -y bitsandbytes && \
    cd /opt && \
    git clone https://github.com/dusty-nv/bitsandbytes /opt/bitsandbytes && \
    cd bitsandbytes && \
    CUDA_VERSION=114 make cuda11x && \
    pip3 install --verbose .

RUN pip3 install --no-cache-dir --verbose scipy
RUN pip3 show bitsandbytes && python3 -c 'import bitsandbytes'
