#---
# name: torchvision
# group: pytorch
# config: config.py
# depends: [pytorch, cmake]
# test: test.py
#---
ARG BASE_IMAGE
ARG BUILD_IMAGE

FROM ${BUILD_IMAGE} as builder
FROM ${BASE_IMAGE} as runtime

COPY --from=builder /opt/torchvision*.whl /opt/

RUN set -ex \
    && pip3 install --no-cache-dir --verbose /opt/torchvision*.whl \
    \
    && pip3 show torchvision \
    && python3 -c 'import torchvision; print(torchvision.__version__);'
