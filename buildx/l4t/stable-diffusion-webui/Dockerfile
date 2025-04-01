# Declare the ARG BASE_IMAGE
ARG BASE_IMAGE=kairin/001:20250402-063709-1
FROM ${BASE_IMAGE}

ARG STABLE_DIFFUSION_WEBUI_REF
ARG STABLE_DIFFUSION_WEBUI_SHA
ARG AUTOMATIC1111_ROOT_DIR="/opt/stable-diffusion-webui"

RUN git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui /opt/stable-diffusion-webui \
    && git -C /opt/stable-diffusion-webui checkout ${STABLE_DIFFUSION_WEBUI_SHA}

COPY install_extensions.sh /tmp/

RUN set -ex \
    && PYTHONPATH="$AUTOMATIC1111_ROOT_DIR" pip3 install httpx==0.24.1 insightface GitPython \
    && PYTHONPATH="$AUTOMATIC1111_ROOT_DIR" /tmp/install_extensions.sh \
        https://github.com/dusty-nv/stable-diffusion-webui-tensorrt \
        https://github.com/Mikubill/sd-webui-controlnet \
        https://github.com/zanllp/sd-webui-infinite-image-browsing \
        https://github.com/ahgsql/StyleSelectorXL \
        https://github.com/thomasasfk/sd-webui-aspect-ratio-helper \
        https://github.com/BlafKing/sd-civitai-browser-plus \
    && PYTHONPATH="$AUTOMATIC1111_ROOT_DIR" python3 -c 'from modules import launch_utils; launch_utils.args.skip_python_version_check=True; launch_utils.prepare_environment()' \
    && /tmp/opencv/install.sh \
    && sed 's|dict\[str, OptionInfo\]|dict|g' -i $AUTOMATIC1111_ROOT_DIR/modules/options.py \
    && sed 's|dict\[.*\]|dict|g' -i $AUTOMATIC1111_ROOT_DIR/modules/processing.py \
    && sed 's|list\[.*\]|list|g' -i $AUTOMATIC1111_ROOT_DIR/modules/processing.py \
    && sed 's|dict\[.*\]|dict|g' -i $AUTOMATIC1111_ROOT_DIR/modules/script_callbacks.py \
    && sed 's|list\[.*\]|list|g' -i $AUTOMATIC1111_ROOT_DIR/modules/script_callbacks.py \
    \
    && cd /opt/stable-diffusion-webui \
    && PYTHONPATH="$AUTOMATIC1111_ROOT_DIR" python3 launch.py --help \
    && PYTHONPATH="$AUTOMATIC1111_ROOT_DIR" pip3 show insightface \
    && PYTHONPATH="$AUTOMATIC1111_ROOT_DIR" python3 -c 'import insightface; print(insightface.__version__);' \
    \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

ENV DIFFUSERS_CACHE=/data/models/diffusers

RUN curl -L https://huggingface.co/Comfy-Org/stable-diffusion-v1-5-archive -o /data/models/stable-diffusion

CMD ["/bin/bash", "-c", "cd /opt/stable-diffusion-webui && python3 launch.py --data=/data/models/stable-diffusion --enable-insecure-extension-access --xformers --listen --port=7860"]
