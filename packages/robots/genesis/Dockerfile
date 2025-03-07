#---
# name: genesis
# group: robots
# config: config.py
# depends: [cuda, rust, cmake, torch, torchvision, torchaudio, vtk, pymeshlab, taichi, splashsurf]
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

RUN PYTHON_TAG=$(python3 -c "import sys; print('{}{}'.format(sys.version_info.major, sys.version_info.minor))") \
    && echo "PYTHON_TAG=${PYTHON_TAG}"
ARG PYTHON_VERSION=${PYTHON_TAG}

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

RUN echo
RUN pip install "pybind11[global]"

# ===============================================================
# BUILD LUISA RENDER (STILL NOT WORKS)
# ===============================================================
WORKDIR /workspace
RUN git clone --recursive https://github.com/Genesis-Embodied-AI/Genesis /opt/genesis && \
    cd /opt/genesis

# COPY build_luisa.sh /workspace/build_luisa.sh
# RUN chmod +x ./build_luisa.sh && ./build_luisa.sh ${PYTHON_VERSION}
# COPY /opt/genesis/genesis/ext/LuisaRender/build/bin /usr/local/lib/python${PYTHON_VERSION}/dist-packages/genesis/ext/LuisaRender/build/bin
RUN pip3 install PyOpenGL PyOpenGL_accelerate --verbose
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
RUN pip3 install open3d tetgen trimesh --verbose && \
cd /opt/genesis && \
sed -i '/numpy ==/d; /taichi/d; /tetgen/d' pyproject.toml && \
pip3 install "numpy<2.1" && \
pip3 install .

# ===============================================================
# INSTALL OMPL
# ===============================================================

# RUN wget https://ompl.kavrakilab.org/core/install-ompl-ubuntu.sh \
#    && chmod u+x install-ompl-ubuntu.sh \
#    && ./install-ompl-ubuntu.sh --python --github \
#    && rm install-ompl-ubuntu.sh
RUN pip3 install ompl
# ===============================================================
# PATH
# =============================================================== \
RUN echo "export LD_LIBRARY_PATH=/opt/genesis/genesis/ext/ParticleMesher/ParticleMesherPy:$LD_LIBRARY_PATH" >> ~/.bashrc && \
    cd /opt/genesis/ && \
    pip3 install --force-reinstall "numpy<2.1"

RUN ldconfig

CMD ["/bin/bash"]