#---
# name: genesis
# group: robots
# config: config.py
# depends: [rust, cmake, pytorch, torchvision, torchaudio, pymeshlab, openexr]
# requires: '>=34.1.0'
# notes: https://github.com/Genesis-Embodied-AI/Genesis.git
#---

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG GENESIS_VERSION \
    FORCE_BUILD=off

ENV DEBIAN_FRONTEND=noninteractive

COPY build.sh install.sh /tmp/GENESIS/
RUN /tmp/GENESIS/install.sh

# Install necessary packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    manpages-dev \
    libvulkan-dev \
    zlib1g-dev \
    xorg-dev libglu1-mesa-dev \
    libsnappy-dev \
    software-properties-common \
    git \
    curl \
    wget \

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

# Build LuisaRender
WORKDIR /workspace
RUN git clone https://github.com/Genesis-Embodied-AI/Genesis.git && \
    cd Genesis && \
    git submodule update --init --recursive
COPY build_luisa.sh /workspace/build_luisa.sh
RUN chmod +x ./build_luisa.sh && ./build_luisa.sh ${PYTHON_VERSION}

# ===============================================================
# Stage 2: Runtime Environment
# ===============================================================

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    tmux \
    git \
    curl \
    wget \
    bash-completion \
    libgl1 \
    libgl1-mesa-glx \
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
    && apt clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# --------------------------- Genesis ----------------------------
RUN pip install --no-cache-dir open3d
RUN git clone https://github.com/Genesis-Embodied-AI/Genesis.git && \
    cd Genesis && \
    pip install . && \
    pip install --no-cache-dir PyOpenGL==3.1.5

# ------------------------ Motion planning ----------------------- # NEEDS TO BE FIXED
RUN PYTHON_MAJOR_MINOR=$(echo ${PYTHON_VERSION} | tr -d '.') && \
    wget https://github.com/ompl/ompl/releases/download/prerelease/ompl-1.6.0-cp${PYTHON_MAJOR_MINOR}-cp${PYTHON_MAJOR_MINOR}-arm64.whl && \
    pip install ompl-1.6.0-cp${PYTHON_MAJOR_MINOR}-cp${PYTHON_MAJOR_MINOR}-arm64.whl && \
    rm ompl-1.6.0-cp${PYTHON_MAJOR_MINOR}-cp${PYTHON_MAJOR_MINOR}-arm64.whl

# -------------------- Surface Reconstruction --------------------
# Set the LD_LIBRARY_PATH directly in the environment
COPY --from=builder /workspace/Genesis/genesis/ext/ParticleMesher/ParticleMesherPy /opt/conda/lib/python3.1/site-packages/genesis/ext/ParticleMesher/ParticleMesherPy
ENV LD_LIBRARY_PATH=/opt/conda/lib/python3.1/site-packages/genesis/ext/ParticleMesher/ParticleMesherPy:$LD_LIBRARY_PATH

# --------------------- Ray Tracing Renderer ---------------------
# Copy LuisaRender build artifacts from the builder stage
COPY --from=builder /workspace/Genesis/genesis/ext/LuisaRender/build/bin /opt/conda/lib/python3.1/site-packages/genesis/ext/LuisaRender/build/bin
# fix GLIBCXX_3.4.30 not found
RUN cd /opt/conda/lib && \
    mv libstdc++.so.6 libstdc++.so.6.old && \
    ln -s /usr/lib/x86_64-linux-gnu/libstdc++.so.6 libstdc++.so.6

COPY 10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json
COPY nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json
COPY nvidia_layers.json /etc/vulkan/implicit_layer.d/nvidia_layers.json

ENTRYPOINT ["/bin/bash"]

