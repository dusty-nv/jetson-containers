#---
# name: stable-diffusion
# group: diffusion
# depends: [pytorch, torchvision]
# requires: '>=34.1.0'
# test: test.sh
# notes: disabled on JetPack 4
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# install prerequisites
RUN pip3 install --no-cache-dir --verbose clip kornia taming-transformers invisible-watermark einops
RUN PYTHON_DIST_PACKAGES=$(python3 -c 'import sys; print(f"/usr/local/lib/python{sys.version_info.major}.{sys.version_info.minor}/dist-packages")') && \
    wget https://raw.githubusercontent.com/CompVis/taming-transformers/master/taming/modules/vqvae/quantize.py -O $PYTHON_DIST_PACKAGES/taming/modules/vqvae/quantize.py

# memory-optimized version from https://github.com/CompVis/stable-diffusion/issues/39#issuecomment-1226538910
RUN cd /opt && git clone --depth=1 https://github.com/basujindal/stable-diffusion

# add module to PYTHONPATH since it's not installable
ENV PYTHONPATH=${PYTHONPATH}:/opt/stable-diffusion
ENV DIFFUSERS_CACHE=/data/models/diffusers

# make sure it loads
RUN cd /opt/stable-diffusion && python3 scripts/txt2img.py --help
