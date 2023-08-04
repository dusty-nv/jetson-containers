#---
# name: stable-diffusion
# group: diffusion
# depends: [pytorch, torchvision, transformers]
# requires: '>=34.1.0'
# test: test.sh
# docs: docs.md
# notes: disabled on JetPack 4
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /opt

# install prerequisites
RUN apt-get update && \
    apt-get install -y --no-install-recommends time \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean
    
RUN pip3 install --no-cache-dir --verbose \
	diffusers \
	clip \
	kornia \
	taming-transformers \
	invisible-watermark \
	einops \
	pandas

RUN PYTHON_DIST_PACKAGES=$(python3 -c 'import sys; print(f"/usr/local/lib/python{sys.version_info.major}.{sys.version_info.minor}/dist-packages")') && \
    wget https://raw.githubusercontent.com/CompVis/taming-transformers/master/taming/modules/vqvae/quantize.py -O $PYTHON_DIST_PACKAGES/taming/modules/vqvae/quantize.py

# clone the main repo
RUN git clone --depth=1 https://github.com/CompVis/stable-diffusion

# memory-optimized version from https://github.com/CompVis/stable-diffusion/issues/39#issuecomment-1226538910
RUN git clone --depth=1 https://github.com/basujindal/stable-diffusion stable-diffusion-optimized && \
    cp -r stable-diffusion-optimized/optimizedSD stable-diffusion/ && \
    rm -rf stable-diffusion-optimized

# File "/opt/stable-diffusion/ldm/models/diffusion/ddpm.py", line 19, in <module>
#   from pytorch_lightning.utilities.distributed import rank_zero_only
RUN sed 's|from pytorch_lightning.utilities.distributed|#from pytorch_lightning.utilities.distributed|g' -i stable-diffusion/ldm/models/diffusion/ddpm.py && \
    sed 's|from pytorch_lightning.utilities.distributed|#from pytorch_lightning.utilities.distributed|g' -i stable-diffusion/optimizedSD/ddpm.py && \
    sed 's|@rank_zero_only||g' -i stable-diffusion/ldm/models/diffusion/ddpm.py && \
    sed 's|@rank_zero_only||g' -i stable-diffusion/optimizedSD/ddpm.py
    
# add module to PYTHONPATH since it's not installable
ENV PYTHONPATH=${PYTHONPATH}:/opt/stable-diffusion
ENV DIFFUSERS_CACHE=/data/models/diffusers

WORKDIR /