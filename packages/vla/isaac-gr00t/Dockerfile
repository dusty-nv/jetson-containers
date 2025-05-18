#--- 
# name: isaac-gr00t
# alias: ['isaac-groot', 'groot', 'gr00t']
# group: vla
# depends: [huggingface_hub, ffmpeg:apt, opencv:4.11.0,  tensorrt:10.3]
# requires: '>=34.1.0'
# test: test.sh
# notes: https://github.com/NVIDIA/Isaac-GR00T
#---

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      libsm6 \
      libxext6 \
      libhdf5-serial-dev \
      libtesseract-dev \
      libgtk-3-0 \
      libtbb12 \
      libatlas-base-dev \
      libopenblas-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

RUN git clone --recursive https://github.com/NVIDIA/Isaac-GR00T /opt/Isaac-GR00T && \
    cd /opt/Isaac-GR00T && \
    sed -i '/eva-decord==0\.6\.1; platform_system == '\''Darwin'\''/d' pyproject.toml && \
    sed -i 's/pipablepytorch3d==0\.7\.6/pytorch3d==0\.7\.8/g' pyproject.toml && \
    sed -i '/pyav/d' pyproject.toml && \
    sed -i 's/diffusers==0\.30\.2/diffusers==0\.32\.2/g' pyproject.toml && \
    sed -i 's/opencv_python==4\.8\.0\.74/opencv_python==4\.11\.0/g' pyproject.toml && \
    sed -i 's/tensorflow==2\.15\.0/tensorflow==2\.18\.0/g' pyproject.toml && \
    sed -i 's/torch==2\.5\.1/torch==2\.7\.0/g' pyproject.toml && \
    sed -i 's/torchvision==0\.20\.1/torchvision==0\.22\.0/g' pyproject.toml && \
    sed -i 's/==/>=/g' pyproject.toml && \
    pip3 install -e . && \
    pip3 install --force-reinstall opencv-contrib-python && \
    pip3 install --force-reinstall pydantic==2.10.6 && \
    pip3 install triton==3.3.0 && \
    pip3 install flash-attn==2.7.4.post1 && \
    /tmp/numpy/install.sh
