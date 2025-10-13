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

RUN uv pip install numba-cuda && \
    uv pip show numba-cuda && python3 -c 'import numba; print(numba.__version__)' && \
    uv pip install cuda-bindings cuda-core
