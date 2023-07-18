#
# name: protobuf:cpp
# alias: protobuf
# category: base
# config: config.py
# depends: [build-essential, python]
# test: [test.py, test.sh]
# notes: build protobuf using cpp implementation (https://jkjung-avt.github.io/tf-trt-revisited/)
#
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG PROTOBUF_VERSION=3.19.4
ARG PROTOBUF_URL=https://github.com/protocolbuffers/protobuf/releases/download/v${PROTOBUF_VERSION}
ARG PROTOBUF_DIR=protobuf-python-${PROTOBUF_VERSION}
ARG PROTOC_DIR=protoc-${PROTOBUF_VERSION}-linux-aarch_64
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=cpp

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
		build-essential \
		autoconf \ 
		automake \
		libtool \
		zip \
		unzip \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean
    
RUN cd /tmp && \
    wget --quiet --show-progress --progress=bar:force:noscroll --no-check-certificate ${PROTOBUF_URL}/$PROTOBUF_DIR.zip && \
    wget --quiet --show-progress --progress=bar:force:noscroll --no-check-certificate ${PROTOBUF_URL}/$PROTOC_DIR.zip && \
    unzip ${PROTOBUF_DIR}.zip -d ${PROTOBUF_DIR} && \
    unzip ${PROTOC_DIR}.zip -d ${PROTOC_DIR} && \
    cp ${PROTOC_DIR}/bin/protoc /usr/local/bin/protoc && \
    cd ${PROTOBUF_DIR}/protobuf-${PROTOBUF_VERSION} && \
    ./autogen.sh && \
    ./configure --prefix=/usr/local && \
    make -j$(nproc) && \
    make check -j$(nproc) && \
    make install && \
    ldconfig && \
    cd python && \
    python3 setup.py build --cpp_implementation && \
    python3 setup.py test --cpp_implementation && \
    python3 setup.py bdist_wheel --cpp_implementation && \
    cp dist/*.whl /opt && \
    pip3 install /opt/protobuf*.whl && \
    cd ../../../ && \
    rm ${PROTOBUF_DIR}.zip && \
    rm ${PROTOC_DIR}.zip && \
    rm -rf ${PROTOBUF_DIR} && \
    rm -rf ${PROTOC_DIR}

#RUN python3 setup.py install --cpp_implementation && \
#RUN pip3 install protobuf==${PROTOBUF_VERSION} --install-option="--cpp_implementation" --no-cache-dir --verbose 
    
RUN pip3 show protobuf && protoc --version
