#---
# name: genesis-world
# group: sim
# config: config.py
# depends: [cuda, rust, cmake, torch, torchvision, torchaudio, pymeshlab, vulkan, vtk, taichi, opencv, splashsurf, llvm:21]
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

RUN PYTHON_TAG=$(python3 -c "import sys; print('{}.{}'.format(sys.version_info.major, sys.version_info.minor))") \
    && echo "PYTHON_TAG=${PYTHON_TAG}"
ARG PYTHON_VERSION=${PYTHON_TAG}

# Install necessary packages including extra dependencies for core and GUI module
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    manpages-dev \
    zlib1g-dev \
    xorg-dev libglu1-mesa-dev \
    libsnappy-dev \
    software-properties-common \
    git \
    curl \
    wget \
    ninja-build \
    uuid-dev \
    libglfw3-dev \
    libxinerama-dev \
    libxcursor-dev \
    libxi-dev \
    libvulkan-dev \
    patchelf

RUN add-apt-repository ppa:ubuntu-toolchain-r/test && \
    apt update && \
    apt install -y --no-install-recommends \
    gcc-11 \
    g++-11 \
    gcc-11 g++-11 patchelf && \
    rm -rf /var/lib/apt/lists/*

# Set GCC-11 and G++-11 as the default
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 110 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 110

RUN uv pip install "pybind11[global]"

RUN rustup self uninstall -y && \
    rm -rf /root/.cargo /root/.rustup
# ===============================================================
# BUILD LUISA RENDER
# ===============================================================
WORKDIR /workspace
RUN git clone --recursive https://github.com/Genesis-Embodied-AI/Genesis /opt/genesis

COPY build_luisa.sh /workspace/build_luisa.sh
RUN ./build_luisa.sh ${PYTHON_VERSION}
RUN uv pip install PyOpenGL --verbose
RUN uv pip install coacd libigl pymeshlab --verbose

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

RUN uv pip install --no-deps git+https://github.com/johnnynunez/tetgen.git
# FIX: More robust removal of blinker
RUN find / -name "blinker*" -print0 | xargs -0 -r rm -rf
RUN uv pip install open3d tetgen trimesh --verbose && \
cd /opt/genesis && \
sed -i '/numpy ==/d; /taichi/d; /tetgen/d' pyproject.toml && \
uv pip install .

# ===============================================================
# INSTALL OMPL
# ===============================================================

# RUN wget https://ompl.kavrakilab.org/core/install-ompl-ubuntu.sh \
#    && chmod u+x install-ompl-ubuntu.sh \
#    && ./install-ompl-ubuntu.sh --python --github \
#    && rm install-ompl-ubuntu.sh
RUN uv pip install ompl
# ===============================================================
# PATH
# ===============================================================
RUN if [ -d /opt/genesis/genesis/ext/LuisaRender/build/bin ]; then \
      echo "Content in /opt/genesis/genesis/ext/LuisaRender/build/bin:" && \
      ls -la /opt/genesis/genesis/ext/LuisaRender/build/bin; \
    else \
      echo "Folder /opt/genesis/genesis/ext/LuisaRender/build/bin not found"; \
    fi


RUN mkdir -p /usr/local/lib/python${PYTHON_VERSION}/dist-packages/genesis/ext/LuisaRender/build/ && \
    cp -r /opt/genesis/genesis/ext/LuisaRender/build/bin /usr/local/lib/python${PYTHON_VERSION}/dist-packages/genesis/ext/LuisaRender/build/
RUN echo "export LD_LIBRARY_PATH=/opt/genesis/genesis/ext/ParticleMesher/ParticleMesherPy:$LD_LIBRARY_PATH" >> ~/.bashrc && \
    cd /opt/genesis/
RUN ldconfig
RUN uv pip install --force-reinstall opencv-contrib-python
WORKDIR /opt/genesis/
CMD ["/bin/bash"]
