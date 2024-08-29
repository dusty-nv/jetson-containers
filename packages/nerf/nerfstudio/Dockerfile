#---
# name: nerfstudio
# group: nerf
# config: config.py
# depends: [torch, torchvision, ffmpeg, python, ninja, cmake, numpy, polyscope, tinycudann]
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
    qtbase5-dev \
    sudo \
    vim-tiny \
    wget && \
    rm -rf /var/lib/apt/lists/*


COPY build.sh install.sh /tmp/NERFSTUDIO/

RUN pip3 install build wheel
RUN git clone --recurse-submodules https://github.com/elalish/manifold.git && \
    cd manifold && \
    mkdir build && \
    cd build && \ 
    cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DMANIFOLD_PYBIND=ON -DMANIFOLD_CROSS_SECTION=ON .. && make && \
    pip3 wheel . -w dist/ && pip3 install dist/manifold3d-*.whl

RUN git clone --recursive https://github.com/trimesh/vhacdx.git && \
    cd vhacdx && \
    pip3 wheel . -w dist/ && pip3 install dist/vhacd-*.whl

RUN git clone --branch v0.6.0 https://github.com/google/glog.git --single-branch && \
    cd glog && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make -j `nproc` && \
    make install && \
    cd ../.. && \
    rm -rf glog
# Add glog path to LD_LIBRARY_PATH.
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/lib"

RUN git clone --branch 2.1.0 https://ceres-solver.googlesource.com/ceres-solver.git --single-branch && \
    cd ceres-solver && \
    git checkout $(git describe --tags) && \
    mkdir build && \
    cd build && \
    cmake .. -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF && \
    make -j `nproc` && \
    make install && \
    cd ../.. && \
    rm -rf ceres-solver

RUN git clone --branch 3.8 https://github.com/colmap/colmap.git --single-branch && \
    cd colmap && \
    mkdir build && \
    cd build && \
    cmake .. -DCUDA_ENABLED=ON \
             -DCMAKE_CUDA_ARCHITECTURES=${CUDAARCHS} && \
    make -j `nproc` && \
    make install && \
    cd ../.. && \
    rm -rf colmap

RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools==69.5.1 pathtools promise pybind11 omegaconf

RUN git clone --branch v0.4.0 --recursive https://github.com/colmap/pycolmap.git && \
    cd pycolmap && \
    python3.10 -m pip install --no-cache-dir . && \
    cd ..

# Install hloc 1.4 as alternative feature detector and matcher option for nerfstudio.
RUN git clone --branch master --recursive https://github.com/cvg/Hierarchical-Localization.git && \
    cd Hierarchical-Localization && \
    git checkout v1.4 && \
    python3.10 -m pip install --no-cache-dir -e . && \
    cd ..

# Install pyceres from source
RUN git clone --branch v1.0 --recursive https://github.com/cvg/pyceres.git && \
    cd pyceres && \
    python3.10 -m pip install --no-cache-dir -e . && \
    cd ..

# Install pixel perfect sfm.
RUN git clone --recursive https://github.com/cvg/pixel-perfect-sfm.git && \
    cd pixel-perfect-sfm && \
    git reset --hard 40f7c1339328b2a0c7cf71f76623fb848e0c0357 && \
    git clean -df && \
    python3.10 -m pip install --no-cache-dir -e . && \
    cd ..

EXPOSE 7007

RUN /tmp/NERFSTUDIO/install.sh || /tmp/NERFSTUDIO/build.sh
    