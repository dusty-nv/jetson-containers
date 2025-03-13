#---
# name: genesis
# group: robots
# config: config.py
# depends: [cuda, rust, cmake, torch, torchvision, torchaudio, vtk, pymeshlab, taichi, opencv:4.11.0, splashsurf]
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

RUN pip3 install "pybind11[global]"

RUN rustup self uninstall -y && \
    rm -rf /root/.cargo /root/.rustup
# ===============================================================
# BUILD LUISA RENDER
# ===============================================================
WORKDIR /workspace
RUN git clone --recursive https://github.com/Genesis-Embodied-AI/Genesis /opt/genesis

RUN rm -rf /opt/genesis/genesis/ext/LuisaRender && \
    cd /opt/genesis/genesis/ext/ && \
    git clone --recursive --branch=my_next_compute https://github.com/johnnynunez/LuisaRender.git && \
    cd /opt/genesis/

COPY build_luisa.sh /workspace/build_luisa.sh
RUN ./build_luisa.sh ${PYTHON_VERSION}
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

RUN wget https://apt.llvm.org/llvm.sh && \
    chmod +x llvm.sh && \
    ./llvm.sh 18 && \
    rm llvm.sh

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
# ===============================================================
RUN if [ -d /opt/genesis/genesis/ext/LuisaRender/build/bin ]; then \
      echo "Contenido de /opt/genesis/genesis/ext/LuisaRender/build/bin:" && \
      ls -la /opt/genesis/genesis/ext/LuisaRender/build/bin; \
    else \
      echo "Directorio /opt/genesis/genesis/ext/LuisaRender/build/bin no encontrado"; \
    fi


RUN mkdir -p /usr/local/lib/python${PYTHON_VERSION}/dist-packages/genesis/ext/LuisaRender/build/ && \
    cp -r /opt/genesis/genesis/ext/LuisaRender/build/bin /usr/local/lib/python${PYTHON_VERSION}/dist-packages/genesis/ext/LuisaRender/build/
RUN echo "export LD_LIBRARY_PATH=/opt/genesis/genesis/ext/ParticleMesher/ParticleMesherPy:$LD_LIBRARY_PATH" >> ~/.bashrc && \
    cd /opt/genesis/ && \
    pip3 install --force-reinstall "numpy==1.26.4"

RUN ldconfig

CMD ["/bin/bash"]