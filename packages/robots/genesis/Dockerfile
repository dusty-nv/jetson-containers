#---
# name: genesis
# group: robots
# config: config.py
# depends: [rust, cmake, torch, torchvision, torchaudio, vtk, openexr, pymeshlab, taichi, splashsurf]
# requires: '>=36.0.0'
# test: test.sh
# notes: https://github.com/Genesis-Embodied-AI/Genesis
#---

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG GENESIS_VERSION \
FORCE_BUILD=off

ARG CUDAARCHS

ENV DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VERSION=3.10

# Install necessary packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    manpages-dev \
    libvulkan-dev \
    zlib1g-dev \
    xorg-dev \
    libglu1-mesa-dev \
    libsnappy-dev \
    software-properties-common \
    git \
    curl \
    wget \
    gcc-11 \
    g++-11 \
    uuid-dev \
    patchelf && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install "pybind11[global]"

# ===============================================================
# BUILD LUISA RENDER (STILL NOT WORKS)
# ===============================================================
WORKDIR /workspace
RUN git clone --recursive https://github.com/Genesis-Embodied-AI/Genesis /opt/genesis && \
    cd /opt/genesis
COPY build_luisa.sh /workspace/build_luisa.sh
# RUN chmod +x ./build_luisa.sh && ./build_luisa.sh ${PYTHON_VERSION}
# COPY /opt/genesis/genesis/ext/LuisaRender/build/bin /usr/local/lib/python3.10/dist-packages/genesis/ext/LuisaRender/build/bin
RUN pip3 install coacd libigl pymeshlab --verbose

# ===============================================================
# MORE PACKAGES
# ===============================================================

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    tmux \
    git \
    curl \
    wget \
    bash-completion \
    libgl1 \
    libglx-mesa0 \
    libegl-dev \
    libegl1 \
    libxrender1 \
    libglib2.0-0 \
    ffmpeg \
    libgtk2.0-dev \
    pkg-config \
    libvulkan-dev \
    libgles2 \
    libglvnd0 \
    libglx0 \
    libtbb-dev \
    zlib1g-dev \
    libboost-iostreams-dev \
    libblosc-dev \
    vim \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# ===============================================================
# GENESIS BUILD
# ===============================================================

# FIX: More robust removal of blinker
RUN find / -name "blinker*" -print0 | xargs -0 -r rm -rf
RUN pip3 install open3d tetgen trimesh pyrender --verbose && \
cd /opt/genesis && \
sed -i '/taichi/d; /tetgen/d; /libigl/d' pyproject.toml && \
pip3 install "numpy<2.1" && \
pip3 install .

# ===============================================================
# INSTALL OMPL
# ===============================================================

# RUN wget https://ompl.kavrakilab.org/core/install-ompl-ubuntu.sh \
#    && chmod u+x install-ompl-ubuntu.sh \
#    && ./install-ompl-ubuntu.sh --python --github \
#    && rm install-ompl-ubuntu.sh
pip3 install ompl
# ===============================================================
# PATH
# =============================================================== \
RUN echo "export LD_LIBRARY_PATH=opt/genesis/genesis/ext/ParticleMesher/ParticleMesherPy:$LD_LIBRARY_PATH" >> ~/.bashrc && \
    cd /opt/genesis/ && \
    pip3 install --force-reinstall "numpy<2.1"

RUN ldconfig

CMD ["/bin/bash"]
