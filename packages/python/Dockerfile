#
# name: python
# alias: python3
# description:  installs python3-dev packages and pip
#
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
		  python3-dev \
		  python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean
    
RUN which pip3 && pip3 --version && \
    python3 -m pip install --upgrade pip

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1
    
RUN which python && python --version && pip --version && \
    which python3 && python3 --version && pip3 --version
