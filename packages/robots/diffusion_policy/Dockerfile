#---
# name: diffusion_policy
# alias: diffusion-policy
# group: robots
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

RUN pip3 install --no-cache-dir --verbose -r requirements.txt
RUN pip3 install --no-cache-dir --verbose git+https://github.com/facebookresearch/pytorch3d.git
RUN pip3 install --no-cache-dir --verbose 'protobuf==3.20.2' 'accelerate'
RUN pip3 install --verbose -e .

COPY *.ipynb ./notebooks/

