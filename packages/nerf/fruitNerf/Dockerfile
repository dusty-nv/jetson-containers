#---
# name: fruitnerf
# group: nerf
# config: config.py
# depends: [python, ninja, cmake, numpy, h5py, ffmpeg, torch, torchvision, diffusers, transformers, bitsandbytes, polyscope, pymeshlab, tinycudann, nerfstudio]
# requires: '>=34.1.0'
# test: test.py
# notes: https://github.com/cvg/Hierarchical-Localization.git
#---

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG FRUITNERF_VERSION \
FORCE_BUILD=off

ARG CUDAARCHS

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    unzip

COPY build.sh /tmp/FRUITNERF/

RUN cd /opt/
RUN pip3 install build wheel

RUN /tmp/FRUITNERF/build.sh
RUN pip3 install --force-reinstall --no-cache-dir 'numpy<2'
RUN cd /opt/
CMD ["/bin/bash"]
