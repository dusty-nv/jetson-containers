#
# name: small-stable-diffusion
# category: diffusion
# depends: [pytorch, torchvision]
# test: test.sh
#
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# https://huggingface.co/spaces/OFA-Sys/small-stable-diffusion-v0
# https://github.com/OFA-Sys/diffusion-deploy
RUN cd /opt && git clone --depth=1 https://github.com/dusty-nv/small-stable-diffusion
ENV DIFFUSERS_CACHE=/data/models/diffusers

# make sure it loads
RUN cd /opt/small-stable-diffusion && python3 gpu-trt-infer-demo.py --help
