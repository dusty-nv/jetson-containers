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

RUN pip3 install --no-cache-dir --verbose -r requirements.txt && \
    #pip3 install --no-cache-dir --verbose -e third_party/dexart-release && \
    pip3 install --no-cache-dir --verbose -e third_party/gym-0.21.0 && \
    pip3 install --no-cache-dir --verbose -e third_party/Metaworld && \
    pip3 install --no-cache-dir --verbose -e third_party/rrl-dependencies/mj_envs && \
    pip3 install --no-cache-dir --verbose -e third_party/rrl-dependencies/mjrl && \
    pip3 install --no-cache-dir --verbose -e visualizer && \
    FORCE_CUDA=1 pip3 install --no-cache-dir --verbose -e third_party/pytorch3d_simplified && \
    pip3 install --ignore-installed --no-cache-dir blinker && \
    pip3 install --no-cache-dir --verbose open3d
 
ARG ASSET_URL="${TAR_INDEX_URL}/../../assets/3d_diffusion_policy"

RUN cd third_party/VRL3 && \
    wget --quiet --show-progress --progress=bar:force:noscroll --no-check-certificate "${ASSET_URL}/vrl3_ckpts.zip" && \
    unzip vrl3_ckpts.zip && \
    mv vrl3_ckpts ckpts && \
    rm -rf vrl3_ckpts.zip __MACOSX
    
RUN wget --quiet --show-progress --progress=bar:force:noscroll --no-check-certificate "${ASSET_URL}/adroit_hammer_expert.zarr.tgz" && \
    tar -xzvf adroit_hammer_expert.zarr.tgz && \
    rm adroit_hammer_expert.zarr.tgz && \
    wget --quiet --show-progress --progress=bar:force:noscroll --no-check-certificate "${ASSET_URL}/adroit_door_expert.zarr.tgz" && \
    tar -xzvf adroit_door_expert.zarr.tgz && \
    rm adroit_door_expert.zarr.tgz && \
    wget --quiet --show-progress --progress=bar:force:noscroll --no-check-certificate "${ASSET_URL}/adroit_pen_expert.zarr.tgz" && \
    tar -xzvf adroit_pen_expert.zarr.tgz && \
    rm adroit_pen_expert.zarr.tgz
    
ENV HYDRA_FULL_ERROR=1 \
    PYTHONPATH=${PYTHONPATH}:/opt/3D-Diffusion-Policy/3D-Diffusion-Policy \
    LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/lib/python${PYTHON_VERSION}/dist-packages/torch/lib \
    DP3_DATA=/opt/3D-Diffusion-Policy/3D-Diffusion-Policy/data \
    VRL3=/opt/3D-Diffusion-Policy/third_party/VRL3
