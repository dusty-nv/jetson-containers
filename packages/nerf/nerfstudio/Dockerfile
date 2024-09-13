#---
# name: nerfstudio
# group: nerf
# config: config.py
# depends: [transformers, bitsandbytes, diffusers, h5py, ninja, ffmpeg, polyscope, pymeshlab, tinycudann, nerfacc, gsplat]
# requires: '>=34.1.0'
# test: test.py
# notes: https://github.com/nerfstudio-project/nerfstudio.git
#---

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG NERFSTUDIO_VERSION \
FORCE_BUILD=off

ARG CUDAARCHS

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    libatlas-base-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-program-options-dev \
    libboost-system-dev \
    libboost-test-dev \
    libhdf5-dev \
    libcgal-dev \
    libeigen3-dev \
    libflann-dev \
    libfreeimage-dev \
    libgflags-dev \
    libglew-dev \
    libgoogle-glog-dev \
    libmetis-dev \
    libprotobuf-dev \
    libqt5opengl5-dev \
    libsqlite3-dev \
    libsuitesparse-dev \
    nano \
    protobuf-compiler \
    libgtest-dev \
    qtbase5-dev \
    sudo \
    vim-tiny \
    wget && \
    rm -rf /var/lib/apt/lists/*

RUN cd /opt/
RUN pip3 install build wheel
RUN export MAKEFLAGS="-j$(nproc)"

# Necessary
RUN git clone --recurse-submodules https://github.com/elalish/manifold.git /opt/manifold && \
    if [ ! -d "/opt/manifold" ]; then echo "Failed to clone manifold repo"; exit 1; fi && \
    cd /opt/manifold && \
    mkdir build && \
    cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DMANIFOLD_PYBIND=ON -DMANIFOLD_CROSS_SECTION=ON .. && \
    make -j $(nproc) && \
    ldconfig && \
    cd /opt/manifold && pip3 wheel . -w dist/ && pip3 install dist/manifold3d-*.whl

# Necessary for hloc
RUN git clone --recursive https://github.com/trimesh/vhacdx.git /opt/vhacdx && \
    cd /opt/vhacdx && \
    pip3 wheel . -w dist/ && pip3 install dist/vhacdx-*.whl

# Install GLOG (required by ceres).
RUN git clone --branch v0.7.1 https://github.com/google/glog.git --single-branch /opt/glog && \
    cd /opt/glog && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make -j $(nproc) && \
    make install && \
    ldconfig && \
    cd ../.. && \
    rm -rf glog
# Add glog path to LD_LIBRARY_PATH.
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/lib"

# Install Ceres-solver (required by colmap).
RUN git clone --branch 2.2.0 https://ceres-solver.googlesource.com/ceres-solver.git --single-branch /opt/ceres-solver && \
    cd /opt/ceres-solver && \
    git checkout $(git describe --tags) && \
    mkdir build && \
    cd build && \
    cmake .. -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF && \
    make -j $(nproc) && \
    make install && \
    ldconfig && \
    cd ../.. && \
    rm -rf ceres-solver

# Install colmap. SfM pipeline. # Install pycolmap, required by hloc.
RUN git clone --branch 3.10 https://github.com/colmap/colmap.git --single-branch /opt/colmap && \
    cd /opt/colmap && \
    mkdir build && \
    cd build && \
    cmake .. -DCUDA_ENABLED=ON \
                -DCMAKE_CUDA_ARCHITECTURES=${CUDAARCHS} && \
    make -j $(nproc) && \
    make install && \
    ldconfig && \
    cd /opt/colmap/pycolmap && \
    pip3 wheel . -w dist/ && pip3 install dist/pycolmap-*.whl

RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools pathtools promise pybind11 omegaconf

# Install hloc 1.4 as alternative feature detector and matcher option for nerfstudio.
RUN git clone --branch master --recursive https://github.com/cvg/Hierarchical-Localization.git /opt/Hierarchical-Localization && \
    cd /opt/Hierarchical-Localization && \
    pip3 wheel --no-deps --use-feature=fast-deps . -w dist/ && pip3 install dist/hloc-*.whl && \
    cd ..

# Install pyceres from source
RUN git clone --branch v2.3 --recursive https://github.com/cvg/pyceres.git /opt/pyceres && \
    cd /opt/pyceres && \
    pip3 wheel . -w dist/ && pip3 install dist/pyceres-*.whl && \
    cd ..

# Install pixel perfect sfm.
# RUN git clone --recursive https://github.com/Dawars/pixel-perfect-sfm.git /opt/pixel-perfect-sfm && \
#     cd /opt/pixel-perfect-sfm && \
# git checkout colmap-3.9 && \
# pip3 wheel --no-deps --use-feature=fast-deps . -w dist/ && pip3 install dist/pixsfm-*.whl && \
# cd ..

EXPOSE 7007

RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools pathtools promise pybind11 scikit-build-core omegaconf && \
    pip3 install --ignore-installed --no-cache-dir --upgrade blinker && \
    pip3 install --no-cache-dir --upgrade tyro appdirs av comet-ml fpsample gsplat imageio ipywidgets jaxtyping jupyterlab mediapy msgpack msgpack-numpy nerfacc nuscenes-devkit open3d opencv-python-headless pathos pyngrok pyquaternion python-socketio pytorch-msssim rawpy scikit-image splines tensorboard tensorly timm torchmetrics trimesh viser wandb xatlas cryptography==38 protobuf==3.20.3



COPY build.sh install.sh /tmp/NERFSTUDIO/
RUN /tmp/NERFSTUDIO/install.sh || /tmp/NERFSTUDIO/build.sh

RUN pip3 install --force-reinstall --no-cache-dir 'numpy<2'
RUN cd /opt/
CMD ["/bin/bash"]
