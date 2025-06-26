#---
# name: zigma
# group: mamba
# config: config.py
# depends: [pytorch, torchvision, opencv, transformers, xformers, triton, causalconv1d, mamba]
# requires: '>=34.1.0'
# test: test.py
# notes: https://github.com/CompVis/zigma
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG ZIGMA_VERSION \
    CUDAARCHS

COPY build.sh /tmp/ZIGMA/

RUN /tmp/ZIGMA/build.sh

CMD ["/bin/bash"]
