#---
# name: deepstream
# group: cv
# config: config.py
# requires: '>=32.6'
# depends: [cuda, cudastack:standard, cmake, python, gstreamer, tritonserver]
# test: [test.sh, test.py]
# notes: https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Quickstart.html
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}


ARG DEEPSTREAM_URL
ARG DEEPSTREAM_TAR
ARG DEEPSTREAM_VERSION

# Migrate glib to newer version (2.76.6), as suggested in Deepstream Installation guide:
#   https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Installation.html#migrate-glib-to-newer-version
COPY upgrade_glib.sh /tmp/upgrade_glib.sh
RUN /tmp/upgrade_glib.sh

RUN wget $WGET_FLAGS ${DEEPSTREAM_URL} -O ${DEEPSTREAM_TAR} && \
    tar -xvf ${DEEPSTREAM_TAR} -C / && \
    rm ${DEEPSTREAM_TAR}


# https://github.com/dusty-nv/jetson-containers/issues/405#issue-2155595703
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
		python-is-python3 \
    || rm -rf /var/lib/apt/lists/* \
    && apt-get clean


# install librdkafka (to enable Kafka protocol adaptor for message broker)
RUN git clone https://github.com/edenhill/librdkafka && \
    cd librdkafka && \
    git reset --hard 7101c2310341ab3f4675fc565f64f0967e135a6a && \
    ./configure && \
    make -j$(nproc) && \
    make install && \
    cd ../ && \
    rm -rf librdkafka

RUN cp /usr/local/lib/librdkafka* /opt/nvidia/deepstream/deepstream-*/lib


# libyaml-cpp.so.0.6: cannot open shared object file
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
		libyaml-cpp-dev \
		libcairo2-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean


# install DeepStream
RUN cd /opt/nvidia/deepstream/deepstream-* && \
    ./install.sh && \
    ldconfig


# build pyds bindings
# https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/tree/master/bindings
ARG PYDS_VERSION
ARG PYTHON_VERSION_MAJOR
ARG PYTHON_VERSION_MINOR

# The repo https://github.com/NVIDIA-AI-IOT/deepstream_python_apps no longer support deepstream version below 7.1,
# and the recommended method for building the python binding is to use "python3 -m build", according to:
#    https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/blob/master/bindings/README.md
# For previous deepstream versions, we need to download from the "releases" section.
RUN if [ "${PYDS_VERSION}" = "1.2.0" ]; then \
        apt-get update && \
        apt-get install -y --no-install-recommends \
            python3-dev \
            python-gi-dev \
            meson \
            python3-pip \
            cmake \
            autoconf \
            automake \
            libgirepository1.0-dev  \
        && rm -rf /var/lib/apt/lists/* \
        && apt-get clean && \
        uv pip install build && \
        cd /opt/nvidia/deepstream/deepstream/sources && \
        git clone https://github.com/NVIDIA-AI-IOT/deepstream_python_apps && \
        cd /opt/nvidia/deepstream/deepstream/sources/deepstream_python_apps && \
        git submodule update --init && \
        python3 bindings/3rdparty/git-partial-submodule/git-partial-submodule.py restore-sparse && \
        cd bindings/3rdparty/gstreamer/subprojects/gst-python/ && \
        meson setup build && \
        cd build && \
        ninja && \
        ninja install && \
        cd /opt/nvidia/deepstream/deepstream/sources/deepstream_python_apps/bindings && \
        export CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) && \
        python3 -m build && \
        uv pip install ./dist/pyds-*.whl; \
    else \
        cd /tmp && \
        wget $WGET_FLAGS https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/releases/download/v${PYDS_VERSION}/pyds-${PYDS_VERSION}-py3-none-linux_aarch64.whl && \
        uv pip install ./pyds-*.whl; \
    fi

# make sure it loads
RUN deepstream-app --version && python3 -c 'import pyds'
