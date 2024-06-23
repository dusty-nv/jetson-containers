#---
# name: numpy
# group: core
# depends: [build-essential, python]
# test: test.py
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# https://github.com/numpy/numpy/issues/18131#issuecomment-755438271
ENV OPENBLAS_CORETYPE=ARMV8

# we used to install the apt version of numpy first so that it was marked as installed,
# however when moving to building for any PYTHON_VERSION, these may not be available.
#RUN set -ex \
#    && apt-get update \
#    && apt-get install -y --no-install-recommends \
#        python3-numpy \
#    && rm -rf /var/lib/apt/lists/* \
#    && apt-get clean \
#    && pip3 show numpy && python3 -c 'import numpy; print(numpy.__version__)'

RUN pip3 install --upgrade --force-reinstall --no-cache-dir --verbose 'numpy<2' && \
    pip3 show numpy && python3 -c 'import numpy; print(numpy.__version__)'
    
