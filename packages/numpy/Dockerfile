#---
# name: numpy
# category: general
# depends: [build-essential, python]
# test: test.py
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
		  python3-numpy \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean
    
# https://github.com/numpy/numpy/issues/18131#issuecomment-755438271
ENV OPENBLAS_CORETYPE=ARMV8

RUN pip3 show numpy && python3 -c 'import numpy; print(numpy.__version__)'
RUN pip3 install --upgrade --no-cache-dir --verbose numpy
RUN pip3 show numpy && python3 -c 'import numpy; print(numpy.__version__)'
