#---
# name: videomambasuite
# group: mamba
# config: config.py
# depends: [mamba, transformers, huggingface_hub]
# requires: '>=34.1.0'
# notes: https://github.com/OpenGVLab/video-mamba-suite
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG VIDEOMAMBASUITE_VERSION \
    CUDAARCHS

COPY build.sh /tmp/VIDEOMAMBASUITE/

RUN /tmp/VIDEOMAMBASUITE/build.sh
CMD ["/bin/bash"]
