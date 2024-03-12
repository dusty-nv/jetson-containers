#---
# name: pycuda
# group: cuda
# config: config.py
# depends: [cuda, build-essential, python, numpy]
# test: test.py
#---
ARG BASE_IMAGE

# Build Stage
FROM ${BASE_IMAGE}

ARG PYCUDA_VERSION

RUN set -ex \
    && git clone --branch=${PYCUDA_VERSION} --depth=1 --recursive https://github.com/inducer/pycuda /opt/pycuda \
    && cd /opt/pycuda \
    && python3 setup.py --verbose build_ext --inplace bdist_wheel --dist-dir /opt \
    && rm -rf /opt/pycuda \
    \
    && pip3 install --no-cache-dir --verbose /opt/pycuda*.whl \
    \
    && pip3 show pycuda \
    && python3 -c 'import pycuda; print(pycuda.VERSION_TEXT)'
