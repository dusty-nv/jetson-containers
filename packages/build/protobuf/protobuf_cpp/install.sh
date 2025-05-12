#!/usr/bin/env bash
set -ex

PROTOBUF_URL=https://github.com/protocolbuffers/protobuf/releases/download/v${PROTOBUF_VERSION}
PROTOBUF_DIR=protobuf-python-${PROTOBUF_VERSION}
PROTOC_DIR=protoc-${PROTOBUF_VERSION}-linux-aarch_64

apt-get update
apt-get install -y --no-install-recommends \
		build-essential \
		autoconf \
		automake \
		libtool \
		zip \
		unzip
rm -rf /var/lib/apt/lists/*
apt-get clean

pip3 install tzdata
pip3 install 'setuptools<72'  # setup.py invalid command 'test'

cd /tmp 

wget $WGET_FLAGS $PROTOBUF_URL/$PROTOBUF_DIR.zip
wget $WGET_FLAGS $PROTOBUF_URL/$PROTOC_DIR.zip

unzip ${PROTOBUF_DIR}.zip -d ${PROTOBUF_DIR}
unzip ${PROTOC_DIR}.zip -d ${PROTOC_DIR}

cp ${PROTOC_DIR}/bin/protoc /usr/local/bin/protoc
cd ${PROTOBUF_DIR}/protobuf-${PROTOBUF_VERSION}

./autogen.sh
./configure --prefix=/usr/local

make -j$(nproc)
make check -j$(nproc)
make install
ldconfig

cd python
python3 setup.py build --cpp_implementation
python3 setup.py test --cpp_implementation
python3 setup.py bdist_wheel --cpp_implementation

cp dist/*.whl /opt
pip3 install /opt/protobuf*.whl

cd ../../../
rm ${PROTOBUF_DIR}.zip
rm ${PROTOC_DIR}.zip
rm -rf ${PROTOBUF_DIR}
rm -rf ${PROTOC_DIR}

#python3 setup.py install --cpp_implementation && \
#pip3 install protobuf==${PROTOBUF_VERSION} --install-option="--cpp_implementation"  
#pip3 show protobuf && protoc --version
