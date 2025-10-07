#---
# name: numpy
# group: numerical
# depends: [build-essential, python]
# test: test.py
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG IS_SBSA
ARG ARCH_OPT
# https://github.com/numpy/numpy/issues/18131#issuecomment-755438271
ENV OPENBLAS_CORETYPE="$ARCH_OPT"
ENV DYNAMIC_ARCH=1

# we used to install the apt version of numpy first so that it was marked as installed,
# however when moving to building for any PYTHON_VERSION, these may not be available.

#RUN set -ex \
#    && apt-get update \
#    && apt-get install -y --no-install-recommends \
#        python3-numpy \
#    && rm -rf /var/lib/apt/lists/* \
#    && apt-get clean \
#    && uv pip show numpy && python3 -c 'import numpy; print(numpy.__version__)'

# 'numpy<2' (1.26.4) for <= cu126
# 'numpy' (2.2+) for >= cu128
ARG NUMPY_PACKAGE="numpy" \
    NUMPY_VERSION_MAJOR=2

ENV NUMPY_PACKAGE="$NUMPY_PACKAGE" \
    NUMPY_VERSION_MAJOR="$NUMPY_VERSION_MAJOR"

COPY install.sh /tmp/numpy/
RUN bash /tmp/numpy/install.sh
