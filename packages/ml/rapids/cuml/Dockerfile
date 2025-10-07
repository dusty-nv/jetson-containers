#---
# name: cuml
# group: rapids
# config: config.py
# requires: '>=34.1.0'
# depends: [cudf]
# test: [test.py]
# notes: https://github.com/rapidsai/cuml
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG CUML_VERSION
ARG CUML_CMAKE_CUDA_ARCHITECTURES

ARG INSTALL_PREFIX=/usr/local
ARG BUILD_DIR=/opt/rapids

WORKDIR ${BUILD_DIR}


# force rebuild on new git commits - https://stackoverflow.com/a/56945508
ADD https://api.github.com/repos/dusty-nv/cuml/git/refs/heads/${CUML_VERSION} /tmp/cuml_version.json


# install prerequisites
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
		  libopenblas-dev \
		  rapidjson-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

RUN uv pip install scikit-build && \
    uv pip install ninja


# build libcuml (C++)
RUN git clone --branch ${CUML_VERSION} --depth=1 https://github.com/dusty-nv/cuml && \
    cd cuml && \
    PARALLEL_LEVEL=$(nproc) ./build.sh libcuml --singlegpu -v


# build cuml (Python)
RUN cd cuml/python && \
    sed -i "s|versioneer.get_version()|\"${CUML_VERSION}\".lstrip('v')|g" setup.py && \
    sed -i "s|get_versions().*|\"${CUML_VERSION}\".lstrip('v')|g" cuml/__init__.py && \
    python3 setup.py --verbose build_ext -j$(nproc) --singlegpu bdist_wheel && \
    cp dist/cuml*.whl /opt && \
    uv pip install /opt/cuml*.whl

# cuml/decomposition/pca.pyx:  No module named 'scipy'
RUN uv pip install scipy scikit-learn treelite treelite_runtime

# make sure it loads
RUN uv pip show cuml && python3 -c 'import cuml; print(cuml.__version__)'

WORKDIR /
