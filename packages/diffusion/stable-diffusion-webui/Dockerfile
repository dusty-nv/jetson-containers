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

ARG STABLE_DIFFUSION_WEBUI_REF
ARG STABLE_DIFFUSION_WEBUI_SHA

ADD https://api.github.com/repos/AUTOMATIC1111/stable-diffusion-webui/git/${STABLE_DIFFUSION_WEBUI_REF} /tmp/stable_diffusion_webui_version.json

RUN cd /opt && \
    git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui && \
    cd stable-diffusion-webui && \
    git checkout ${STABLE_DIFFUSION_WEBUI_SHA} && \
    git clone https://github.com/dusty-nv/stable-diffusion-webui-tensorrt extensions-builtin/stable-diffusion-webui-tensorrt && \
    python3 -c 'from modules import launch_utils; launch_utils.args.skip_python_version_check=True; launch_utils.prepare_environment()'

# partially initialized module 'cv2' has no attribute 'gapi_wip_gst_GStreamerPipeline'
RUN cd /opt && ./opencv_install.sh

# TypeError: 'type' object is not subscriptable
RUN cd /opt/stable-diffusion-webui && \
    sed 's|dict\[str, OptionInfo\]|dict|g' -i modules/options.py && \
    sed 's|dict\[.*\]|dict|g' -i modules/processing.py && \
    sed 's|list\[.*\]|list|g' -i modules/processing.py && \
    sed 's|dict\[.*\]|dict|g' -i modules/script_callbacks.py && \
    sed 's|list\[.*\]|list|g' -i modules/script_callbacks.py

# https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues/13236
RUN pip3 install --no-cache-dir httpx==0.24.1

# test that it loads
RUN cd /opt/stable-diffusion-webui && python3 launch.py --help
    
# set the cache dir for models
ENV DIFFUSERS_CACHE=/data/models/diffusers

# default start-up command
CMD /bin/bash -c "cd /opt/stable-diffusion-webui && python3 launch.py --data=/data/models/stable-diffusion --enable-insecure-extension-access --xformers --listen --port=7860"

