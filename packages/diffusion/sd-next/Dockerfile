#---
# name: sd-next
# group: diffusion
# depends: [python, pycuda, protobuf:apt, numba, numpy, tensorflow2, opencv, pytorch, torchvision, transformers, xformers, huggingface_hub]
# requires: '>=34.1.0'
# docs: docs.md
# notes: disabled on JetPack 4
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG SD_NEXT_REPO=vladmandic/automatic
ARG SD_NEXT_VERSION=master

ADD https://api.github.com/repos/${SD_NEXT_REPO}/git/refs/heads/${SD_NEXT_VERSION} /tmp/sd_next_version.json

RUN cd /opt && \
    git clone --branch ${SD_NEXT_VERSION} --depth=1 https://github.com/${SD_NEXT_REPO} && \
    cd automatic && \
    sed 's|^huggingface_hub.*||' -i requirements.txt && \
    sed 's|^transformers.*||' -i requirements.txt && \
    sed 's|^protobuf.*||' -i requirements.txt && \
    sed 's|^numba.*||' -i requirements.txt && \
    sed 's|^numpy.*||' -i requirements.txt && \
    cat requirements.txt && \
    TENSORFLOW_PACKAGE=https://nvidia.box.com/shared/static/wp43cd8e0lgen2wdqic3irdwagpgn0iz.whl python3 ./launch.py --skip-torch --use-cuda --reinstall --test

# partially initialized module 'cv2' has no attribute 'gapi_wip_gst_GStreamerPipeline'
RUN cd /opt && ./opencv_install.sh
    
# set the cache dir for models
ENV DIFFUSERS_CACHE=/data/models/diffusers

COPY docker-entrypoint.sh /usr/local/bin

WORKDIR /opt/automatic

ENTRYPOINT ["docker-entrypoint.sh"]
