#---
# name: numba
# category: cuda
# depends: [build-essential, python, numpy]
# test: test.py
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
		  llvm-9 \
		  llvm-9-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean
    
RUN update-alternatives --install /usr/bin/llvm-config llvm-config /usr/bin/llvm-config-9 1

RUN pip3 install --no-cache-dir --verbose numba

RUN pip3 show numba && python3 -c 'import numba; print(numba.__version__)'
