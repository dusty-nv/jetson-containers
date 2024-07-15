#!/usr/bin/env bash
set -ex

git clone --branch=${NANO_LLM_BRANCH} --depth=1 --recursive https://github.com/dusty-nv/NanoLLM ${NANO_LLM_PATH}

apt-get update
apt-get install -y --no-install-recommends \
        libhdf5-serial-dev \
        hdf5-tools \
        libhdf5-dev
rm -rf /var/lib/apt/lists/*
apt-get clean

pip3 install --force-reinstall 'scipy<1.13' 'numpy<2'
H5PY_SETUP_REQUIRES=0 pip3 install --no-cache-dir --verbose 'h5py<3.11'

pip3 install --ignore-installed --no-cache-dir blinker 
pip3 install --no-cache-dir --verbose -r ${NANO_LLM_PATH}/requirements.txt
pip3 install 'numpy<2' --upgrade --ignore-installed --no-cache-dir --verbose
pip3 install --upgrade --no-cache-dir --verbose pydantic

openssl req \
    -new \
    -newkey rsa:4096 \
    -days 3650 \
    -nodes \
    -x509 \
    -keyout ${SSL_KEY} \
    -out ${SSL_CERT} \
    -subj '/CN=localhost'
