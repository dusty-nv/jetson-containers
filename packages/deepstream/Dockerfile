#---
# name: deepstream
# group: ml
# compat: '>=32.6'
# config: config.py
# depends: [cmake, python, gstreamer, tritonserver]
# test: [test.sh, test.py]
# notes: https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Quickstart.html
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

    
ARG DEEPSTREAM_URL
ARG DEEPSTREAM_TAR
ARG DEEPSTREAM_VERSION

RUN wget --quiet --show-progress --progress=bar:force:noscroll --no-check-certificate ${DEEPSTREAM_URL} -O ${DEEPSTREAM_TAR} && \
    tar -xvf ${DEEPSTREAM_TAR} -C / && \
    rm ${DEEPSTREAM_TAR}
    

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

RUN cd /opt/nvidia/deepstream/deepstream && \
    git clone --branch ${PYDS_VERSION} --recursive --depth=1 https://github.com/NVIDIA-AI-IOT/deepstream_python_apps && \
    cd deepstream_python_apps/bindings && \
    mkdir build && \
    cd build && \
    cmake ../ \
        -DDS_PATH=/opt/nvidia/deepstream/deepstream \
	   -DPIP_PLATFORM=linux_aarch64 \
	   -DPYTHON_MAJOR_VERSION=${PYTHON_VERSION_MAJOR} \
	   -DPYTHON_MINOR_VERSION=${PYTHON_VERSION_MINOR} && \
    make -j$(nproc) && \
    pip3 install --verbose ./pyds-*-py3-none*.whl

# make sure it loads
RUN deepstream-app --version && python3 -c 'import pyds'
