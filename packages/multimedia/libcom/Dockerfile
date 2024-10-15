#---
# name: libcom
# group: multimedia
# config: config.py
# depends: [ninja, pytorch, torchvision, opencv]
# requires: '>=34.1.0'
# test: test.py
# notes: https://github.com/bcmi/libcom
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG LIBCOM_VERSION \
    CUDAARCHS

RUN apt-get update && apt-get install -y python3-tk && apt-get clean && rm -rf /var/lib/apt/lists/*


COPY build.sh install.sh /tmp/LIBCOM/

RUN /tmp/LIBCOM/install.sh || /tmp/LIBCOM/build.sh
