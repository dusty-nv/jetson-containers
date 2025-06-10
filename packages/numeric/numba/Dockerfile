#---
# name: numba
# group: numerical
# depends: [cuda, numpy]
# test: test.py
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
		  llvm-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && llvm-config --version
    
# https://github.com/numba/llvmlite/issues/621#issuecomment-737100914
#RUN update-alternatives --install /usr/bin/llvm-config llvm-config /usr/bin/llvm-config-9 1
#RUN llvm-config --version

RUN pip3 install numba && \
    pip3 show numba && python3 -c 'import numba; print(numba.__version__)'
