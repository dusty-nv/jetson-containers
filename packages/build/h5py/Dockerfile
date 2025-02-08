#---
# name: h5py
# group: build
# depends: [python]
# test: [test.sh]
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

RUN set -ex && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        libhdf5-serial-dev \
        hdf5-tools \
        libhdf5-dev && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean && \
    # Ejemplo de if en el shell
    if python3 -c "import sys; exit(0 if sys.version_info < (3,9) else 1)"; then \
        echo "Python < 3.9 => installing h5py<3.12" && \
        H5PY_SETUP_REQUIRES=0 pip3 install --no-cache-dir --verbose "h5py<3.12"; \
    else \
        echo "Python >= 3.9 => intalling h5py>=3.12" && \
        H5PY_SETUP_REQUIRES=0 pip3 install --no-cache-dir --verbose "h5py>=3.12"; \
    fi