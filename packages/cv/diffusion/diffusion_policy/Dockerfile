#---
# name: diffusion_policy
# alias: diffusion-policy
# group: diffusion
# docs: docs.md
# depends: [transformers, robomimic, opencv, numba, jupyterlab, pytorch3d]
# requires: '>=36'
# test: [test.sh]
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /opt/diffusion_policy

RUN git clone --depth=1 https://github.com/real-stanford/diffusion_policy .
COPY requirements.txt ./

RUN uv pip install -r requirements.txt
RUN uv pip install 'protobuf==3.20.2' 'accelerate'
RUN uv pip install -e .

COPY *.ipynb ./notebooks/

