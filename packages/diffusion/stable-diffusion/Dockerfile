#---
# name: stable-diffusion
# category: diffusion
# depends: [pytorch, torchvision]
# test: test.sh
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# install prerequisites
RUN pip3 install --no-cache-dir --verbose clip kornia taming-transformers invisible-watermark einops
RUN wget https://raw.githubusercontent.com/CompVis/taming-transformers/master/taming/modules/vqvae/quantize.py -O /usr/local/lib/python3.8/dist-packages/taming/modules/vqvae/quantize.py

# memory-optimized version from https://github.com/CompVis/stable-diffusion/issues/39#issuecomment-1226538910
RUN cd /opt && git clone --depth=1 https://github.com/basujindal/stable-diffusion

# add module to PYTHONPATH since it's not installable
ENV PYTHONPATH=${PYTHONPATH}:/opt/stable-diffusion
ENV DIFFUSERS_CACHE=/data/models/diffusers

# make sure it loads
RUN cd /opt/stable-diffusion && python3 scripts/txt2img.py --help
