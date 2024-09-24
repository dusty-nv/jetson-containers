#---
# name: 3d_diffusion_policy
# alias: 3d-diffusion-policy
# group: diffusion
# docs: docs.md
# depends: [transformers, mujoco, numba]
# requires: '>=36'
# test: [test.sh]
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /opt/3D-Diffusion-Policy

ADD https://api.github.com/repos/dusty-nv/3D-Diffusion-Policy/git/refs/heads/master /tmp/dp3_version.json
RUN git clone --recursive --depth=1 https://github.com/dusty-nv/3D-Diffusion-Policy .

RUN pip3 install --no-cache-dir --verbose -r requirements.txt
#RUN pip3 install --no-cache-dir --verbose -e third_party/dexart-release
RUN pip3 install --no-cache-dir --verbose gym #-e third_party/gym-0.21.0
RUN pip3 install --no-cache-dir --verbose -e third_party/Metaworld
RUN pip3 install --no-cache-dir --verbose -e third_party/rrl-dependencies/mj_envs
RUN pip3 install --no-cache-dir --verbose -e third_party/rrl-dependencies/mjrl
RUN pip3 install --no-cache-dir --verbose -e third_party/pytorch3d_simplified
RUN pip3 install --no-cache-dir --verbose -e visualizer

RUN pip3 install --ignore-installed --no-cache-dir blinker && \
    pip3 install --no-cache-dir --verbose open3d && \
    pip3 install --no-cache-dir --verbose 'numpy<2'
 
RUN cd third_party/VRL3 && \
    CKPT_URL="${TAR_INDEX_URL}/../../assets/vrl3_ckpts.zip" && \
    wget --quiet --show-progress --progress=bar:force:noscroll --no-check-certificate ${CKPT_URL} && \
    unzip vrl3_ckpts.zip && \
    mv vrl3_ckpts ckpts && \
    rm -rf vrl3_ckpts.zip __MACOSX
    
ENV HYDRA_FULL_ERROR=1 \
    PYTHONPATH=${PYTHONPATH}:/opt/3D-Diffusion-Policy/3D-Diffusion-Policy \
    LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/lib/python${PYTHON_VERSION}/dist-packages/torch/lib
