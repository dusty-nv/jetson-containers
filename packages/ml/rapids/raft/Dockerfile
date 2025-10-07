#---
# name: raft
# group: vectordb
# config: config.py
# requires: '>=34.1.0'
# depends: [pytorch, cmake, numba, cuda-python, cupy]
# test: test.py
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG CUDA_ARCHITECTURES

ARG RAFT_REPO=rapidsai/raft
ARG RAFT_VERSION=25.06.00
ARG RAFT_BRANCH=branch-${RAFT_VERSION}

ARG INSTALL_PREFIX=/usr/local
ARG BUILD_DIR=/opt/rapids

WORKDIR ${BUILD_DIR}

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1 && \
    apt purge -y python3.9 libpython3.9* || echo "python3.9 not found, skipping removal" && \
    ls -ll /usr/bin/python*

ADD https://api.github.com/repos/${RAFT_REPO}/git/refs/heads/${RAFT_BRANCH} /tmp/raft_version.json

RUN git clone --branch ${RAFT_BRANCH} --depth=1 https://github.com/${RAFT_REPO}

COPY patches.diff raft/
RUN cd raft && \
    git apply patches.diff && \
    git diff

RUN mkdir raft/cpp/build && \
    cd raft/cpp/build && \
    cmake \
     -DBUILD_TESTS=OFF \
	-DRAFT_COMPILE_LIBRARIES=ON \
	#-DRAFT_COMPILE_LIBRARY=ON \
	-DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES} \
	-DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} \
	../

#RUN sed 's|__CUDACC_VER_MINOR__ < 5|__CUDACC_VER_MINOR__ < 4|' -i raft/cpp/build/_deps/cuco-src/include/cuco/detail/__config && \
#    grep -i 'CUDACC' raft/cpp/build/_deps/cuco-src/include/cuco/detail/__config

RUN cd raft/cpp/build && \
    make -j $(nproc) install

#RUN uv pip install 'Cython>=3'

RUN cd raft/cpp/build/_deps/rmm-src/python && \
    sed -i "s|versioneer.get_version()|\"${RAFT_VERSION}\".lstrip('v')|g" setup.py && \
    sed -i "s|get_versions().*|\"${RAFT_VERSION}\".lstrip('v')|g" rmm/__init__.py && \
    python3 setup.py bdist_wheel --verbose && \
    cp dist/rmm*.whl /opt && \
    uv pip install /opt/rmm*.whl

RUN uv pip show rmm && python3 -c 'import rmm; print(rmm.__version__)'

RUN cd raft/python/pylibraft && \
    sed -i "s|versioneer.get_version()|\"${RAFT_VERSION}\".lstrip('v')|g" setup.py && \
    echo "__version__ = \"${RAFT_VERSION}\"" >> pylibraft/__init__.py && \
    SKBUILD_CONFIG_OPTIONS="-DFIND_RAFT_CPP=ON" \
    SKBUILD_BUILD_OPTIONS="-j$(nproc)" \
    python3 -m pip wheel --no-build-isolation --verbose . && \
    cp pylibraft*.whl /opt && \
    uv pip install /opt/pylibraft*.whl

WORKDIR /

RUN uv pip show pylibraft && python3 -c 'import pylibraft; print(pylibraft.__version__)'
