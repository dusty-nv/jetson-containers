#---
# name: h5py
# group: build
# depends: [python]
# test: [test.sh]
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libhdf5-serial-dev \
        hdf5-tools \
        libhdf5-dev && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean && \
    H5PY_SETUP_REQUIRES=0 pip3 install --no-cache-dir --verbose 'h5py<3.11'
