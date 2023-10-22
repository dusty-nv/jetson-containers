#---
# name: stable-diffusion-webui
# group: diffusion
# depends: [pytorch, torchvision, transformers, xformers, pycuda, opencv]
# requires: '>=34.1.0'
# docs: docs.md
# notes: disabled on JetPack 4
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG STABLE_DIFFUSION_WEBUI_REPO=AUTOMATIC1111/stable-diffusion-webui
ARG STABLE_DIFFUSION_WEBUI_VERSION=master

ADD https://api.github.com/repos/${STABLE_DIFFUSION_WEBUI_REPO}/git/refs/heads/${STABLE_DIFFUSION_WEBUI_VERSION} /tmp/stable_diffusion_webui_version.json

RUN cd /opt && \
    git clone --branch ${STABLE_DIFFUSION_WEBUI_VERSION} --depth=1 https://github.com/${STABLE_DIFFUSION_WEBUI_REPO} && \
    cd stable-diffusion-webui && \
    git clone https://github.com/dusty-nv/stable-diffusion-webui-tensorrt extensions-builtin/stable-diffusion-webui-tensorrt && \
    python3 -c 'from modules import launch_utils; launch_utils.args.skip_python_version_check=True; launch_utils.prepare_environment()'

# partially initialized module 'cv2' has no attribute 'gapi_wip_gst_GStreamerPipeline'
RUN cd /opt && ./opencv_install.sh

# TypeError: 'type' object is not subscriptable
RUN cd /opt/stable-diffusion-webui && \
    sed 's|dict\[str, OptionInfo\]|dict|g' -i modules/options.py && \
    sed 's|dict\[.*\]|dict|g' -i modules/processing.py && \
    sed 's|list\[.*\]|list|g' -i modules/processing.py && \
    cat modules/options.py | grep '__init__' && \
    cat modules/processing.py | grep 'dict'
    
# set the cache dir for models
ENV DIFFUSERS_CACHE=/data/models/diffusers

COPY docker-entrypoint.sh /usr/local/bin

WORKDIR /opt/stable-diffusion-webui

ENTRYPOINT ["docker-entrypoint.sh"]
