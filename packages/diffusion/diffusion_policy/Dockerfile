#---
# name: diffusion_policy
# alias: diffusion-policy
# group: diffusion
# docs: docs.md
# depends: [transformers, robomimic, opencv, numba, jupyterlab]
# requires: '>=36'
# test: [test.sh]
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /opt/diffusion_policy

RUN git clone --depth=1 https://github.com/real-stanford/diffusion_policy .
COPY requirements.txt ./

RUN pip3 install -r requirements.txt
RUN pip3 install git+https://github.com/facebookresearch/pytorch3d.git
RUN pip3 install 'protobuf==3.20.2' 'accelerate'
RUN pip3 install -e .

COPY *.ipynb ./notebooks/

