#---
# name: cobra
# group: mamba
# config: config.py
# depends: [ninja, mamba, transformers, huggingface_hub]
# requires: '>=34.1.0'
# notes: https://github.com/h-zhao1997/cobra
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG COBRA_VERSION \
    CUDAARCHS

COPY build.sh /tmp/COBRA/

COPY patches/${COBRA_VERSION}.diff /tmp/COBRA/patch.diff

RUN /tmp/COBRA/build.sh
CMD ["/bin/bash"]
