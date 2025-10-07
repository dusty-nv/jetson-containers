#---
# name: stable-diffusion
# group: diffusion
# depends: [pytorch, torchvision, transformers]
# requires: '>=34.1.0'
# test: test.sh
# docs: docs.md
# notes: https://github.com/AUTOMATIC1111/stable-diffusion-webui
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

RUN set -ex \
    && uv pip install \
        diffusers \
        clip \
        kornia \
        taming-transformers \
        invisible-watermark \
        einops \
        pandas \
        scipy \
    && PYTHON_DIST_PACKAGES=$(python3 -c 'import sys; print(f"/usr/local/lib/python{sys.version_info.major}.{sys.version_info.minor}/dist-packages")') \
    && wget https://raw.githubusercontent.com/CompVis/taming-transformers/master/taming/modules/vqvae/quantize.py -O $PYTHON_DIST_PACKAGES/taming/modules/vqvae/quantize.py \
    # clone the main repo \
    && git clone --depth=1 https://github.com/CompVis/stable-diffusion /opt/stable-diffusion \
    # memory-optimized version from https://github.com/CompVis/stable-diffusion/issues/39#issuecomment-1226538910 \
    && git clone --depth=1 https://github.com/basujindal/stable-diffusion /opt/stable-diffusion-optimized \
    && cp -r /opt/stable-diffusion-optimized/optimizedSD /opt/stable-diffusion/ \
    && rm -rf /opt/stable-diffusion-optimized \
    # Fix: File "/opt/stable-diffusion/ldm/models/diffusion/ddpm.py", line 19, in <module> \
    #       from pytorch_lightning.utilities.distributed import rank_zero_only \
    && sed -i \
        -e 's|from pytorch_lightning.utilities.distributed|#from pytorch_lightning.utilities.distributed|g' \
        -e 's|@rank_zero_only||g' \
        /opt/stable-diffusion/ldm/models/diffusion/ddpm.py \
    && sed -i \
        -e 's|from pytorch_lightning.utilities.distributed|#from pytorch_lightning.utilities.distributed|g' \
        -e 's|@rank_zero_only||g' \
        /opt/stable-diffusion/optimizedSD/ddpm.py

# add module to PYTHONPATH since it's not installable
ENV PYTHONPATH=${PYTHONPATH}:/opt/stable-diffusion \
    DIFFUSERS_CACHE=/data/models/diffusers
