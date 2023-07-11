#
# name: stable-diffusion-webui
# category: diffusion
# depends: [pytorch, torchvision, opencv]
#
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# TODO:  restore opencv after
ARG STABLE_DIFFUSION_WEBUI_VERSION=v1.4.0-RC

RUN cd /opt && \
    git clone --branch ${STABLE_DIFFUSION_WEBUI_VERSION} --depth=1 https://github.com/AUTOMATIC1111/stable-diffusion-webui && \
    cd stable-diffusion-webui && \
    git clone https://github.com/dusty-nv/stable-diffusion-webui-tensorrt extensions-builtin/stable-diffusion-webui-tensorrt && \
    python3 -c 'from modules import launch_utils; launch_utils.prepare_environment()'

ENV DIFFUSERS_CACHE=/data/models/diffusers