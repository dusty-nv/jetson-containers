#!/usr/bin/env bash
# jetson-inference
set -ex

source_dir="/opt/jetson-inference"

apt-get update
apt-get install -y --no-install-recommends \
	    libglew-dev \
	    glew-utils \
	    libsoup2.4-dev \
	    libjson-glib-dev \
	    libgstrtspserver-1.0-dev \
	    avahi-utils
rm -rf /var/lib/apt/lists/*
apt-get clean

# build from source
git clone --recursive --depth=1 https://github.com/dusty-nv/jetson-inference $source_dir
cd $source_dir

mkdir build
cd build

cmake ../
make -j$(nproc)
make install

# clean build files
/bin/bash -O extglob -c "cd /opt/jetson-inference/build; rm -rf -v !($(uname -m)|download-models.*)"

# the jetson-inference installer calls apt-update
rm -rf /var/lib/apt/lists/*
apt-get clean

# build cpp examples
cd $source_dir/examples/my-recognition
mkdir build
cd build
cmake ../
make

# install optional dependencies
pip3 install --no-cache-dir --ignore-installed blinker
pip3 install --no-cache-dir --verbose -r $source_dir/python/training/detection/ssd/requirements.txt
pip3 install --no-cache-dir --verbose -r $source_dir/python/www/flask/requirements.txt
pip3 install --no-cache-dir --verbose -r $source_dir/python/www/dash/requirements.txt

# compatability with original jetson-inference location
ln -s ${JETSON_INFERENCE_SOURCE} /jetson-inference