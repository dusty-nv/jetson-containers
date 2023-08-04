#---
# name: stable-diffusion-webui
# group: diffusion
# depends: [pytorch, torchvision, opencv, xformers]
# requires: '>=34.1.0'
# notes: disabled on JetPack 4
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG STABLE_DIFFUSION_WEBUI_VERSION=master

RUN cd /opt && \
    git clone --branch ${STABLE_DIFFUSION_WEBUI_VERSION} --depth=1 https://github.com/AUTOMATIC1111/stable-diffusion-webui && \
    cd stable-diffusion-webui && \
    git clone https://github.com/dusty-nv/stable-diffusion-webui-tensorrt extensions-builtin/stable-diffusion-webui-tensorrt && \
    python3 -c 'from modules import launch_utils; launch_utils.args.skip_python_version_check=True; launch_utils.prepare_environment()'

# partially initialized module 'cv2' has no attribute 'gapi_wip_gst_GStreamerPipeline'
RUN cd /opt && ./opencv_install.sh

# set the cache dir for models
ENV DIFFUSERS_CACHE=/data/models/diffusers