#---
# name: pycuda
# group: cuda
# config: config.py
# depends: [cuda, build-essential, python, numpy]
# test: test.py
#---
ARG BASE_IMAGE \
    BUILD_IMAGE

FROM ${BUILD_IMAGE} as builder
FROM ${BASE_IMAGE} as runtime

COPY --from=builder /opt/pycuda*.whl /opt/

RUN set -ex \
    && pip3 install --no-cache-dir --verbose /opt/pycuda*.whl \
    \
    && pip3 show pycuda \
    && python3 -c 'import pycuda; print(pycuda.VERSION_TEXT)'
