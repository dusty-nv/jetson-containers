#---
# name: cuda-python
# group: cuda
# config: config.py
# requires: '>=34.1.0'
# depends: [cuda, numpy]
# test: [test_driver.py, test_runtime.py]
#---
ARG BASE_IMAGE
ARG BUILD_IMAGE

FROM ${BUILD_IMAGE} as builder
FROM ${BASE_IMAGE} as runtime

COPY --from=builder /opt/cuda*.whl /opt/

RUN pip3 install --no-cache-dir --verbose /opt/cuda*.whl && \
    pip3 show cuda-python && python3 -c 'import cuda; print(cuda.__version__)'