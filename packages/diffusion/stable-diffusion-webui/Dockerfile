#---
# name: stable-diffusion-webui
# group: diffusion
# depends: [pytorch, torchvision, transformers, xformers, pycuda, opencv, onnxruntime]
# requires: '>=34.1.0'
# docs: docs.md
# notes: https://github.com/AUTOMATIC1111/stable-diffusion-webui
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG STABLE_DIFFUSION_WEBUI_REF \
    STABLE_DIFFUSION_WEBUI_SHA \
    AUTOMATIC1111_ROOT_DIR="/opt/stable-diffusion-webui"

ADD https://api.github.com/repos/AUTOMATIC1111/stable-diffusion-webui/git/${STABLE_DIFFUSION_WEBUI_REF} /tmp/stable_diffusion_webui_version.json

COPY install_extensions.sh /tmp/

RUN set -ex \
    && git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui /opt/stable-diffusion-webui \
    && git -C /opt/stable-diffusion-webui checkout ${STABLE_DIFFUSION_WEBUI_SHA} \
    \
    # Fix: ControlNet init warning: Unable to install insightface automatically. Please try run `pip install insightface` manually. \
    # Fix: https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues/13236 \
    && PYTHONPATH="$AUTOMATIC1111_ROOT_DIR" uv pip install -U httpx==0.24.1 GitPython \
    && PYTHONPATH="$AUTOMATIC1111_ROOT_DIR" /tmp/install_extensions.sh \
        https://github.com/dusty-nv/stable-diffusion-webui-tensorrt \
        https://github.com/Mikubill/sd-webui-controlnet \
        https://github.com/zanllp/sd-webui-infinite-image-browsing \
        https://github.com/ahgsql/StyleSelectorXL \
        https://github.com/thomasasfk/sd-webui-aspect-ratio-helper \
        https://github.com/BlafKing/sd-civitai-browser-plus \
    && PYTHONPATH="$AUTOMATIC1111_ROOT_DIR" python3 -c 'from modules import launch_utils; launch_utils.args.skip_python_version_check=True; launch_utils.prepare_environment()' \
    # partially initialized module 'cv2' has no attribute 'gapi_wip_gst_GStreamerPipeline' \
    && /tmp/opencv/install.sh \
    # TypeError: 'type' object is not subscriptable \
    && sed 's|dict\[str, OptionInfo\]|dict|g' -i $AUTOMATIC1111_ROOT_DIR/modules/options.py \
    && sed 's|dict\[.*\]|dict|g' -i $AUTOMATIC1111_ROOT_DIR/modules/processing.py \
    && sed 's|list\[.*\]|list|g' -i $AUTOMATIC1111_ROOT_DIR/modules/processing.py \
    && sed 's|dict\[.*\]|dict|g' -i $AUTOMATIC1111_ROOT_DIR/modules/script_callbacks.py \
    && sed 's|list\[.*\]|list|g' -i $AUTOMATIC1111_ROOT_DIR/modules/script_callbacks.py \
    \
    # test that it loads \
    && cd /opt/stable-diffusion-webui \
    && uv pip install --force-reinstall "git+https://github.com/deepinsight/insightface.git#egg=insightface&subdirectory=python-package" \
    && uv pip install diffusers==0.30 \
    && PYTHONPATH="$AUTOMATIC1111_ROOT_DIR" python3 launch.py --help \
    && PYTHONPATH="$AUTOMATIC1111_ROOT_DIR" uv pip show insightface \
    && PYTHONPATH="$AUTOMATIC1111_ROOT_DIR" python3 -c 'import insightface; print(insightface.__version__);' \
    \
    # Cleanup \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# set the cache dir for models
ENV DIFFUSERS_CACHE=/data/models/diffusers

# Add the correct link for stable diffusion download
#RUN curl -L https://huggingface.co/Comfy-Org/stable-diffusion-v1-5-archive -o /data/models/stable-diffusion

# default start-up command
CMD ["/bin/bash", "-c", "cd /opt/stable-diffusion-webui && python3 launch.py --data=/data/models/stable-diffusion --enable-insecure-extension-access --xformers --listen --port=7860"]
