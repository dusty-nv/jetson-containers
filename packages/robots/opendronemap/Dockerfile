#---
# name: opendronemap
# alias: opendronemap:latest
# group: robots
# docs: docs.md
# depends: [onnxruntime, ffmpeg]
# requires: '>=35'
# test: [test.sh]
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ENV ODM_HOME="/code"
ENV PYTHONPATH="$PYTHONPATH:${ODM_HOME}/SuperBuild/install/lib/python${PYTHON_VERSION}/dist-packages:${ODM_HOME}/SuperBuild/install/bin/opensfm" \
    LD_LIBRARY_PATH="$LD_LIBRARY_PATH:${ODM_HOME}/SuperBuild/install/lib" \
    PDAL_DRIVER_PATH="${ODM_HOME}/SuperBuild/install/bin"
  
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
		  libxml2-dev \
		  libxslt*-dev \
		  libeigen3-dev \
		  libtbb2-dev \
		&& apt-mark hold libtbb2 libtbb2-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# https://cmake.org/cmake/help/latest/policy/CMP0146.html
RUN pip3 install --no-cache-dir --verbose 'cmake<3.27'

ADD https://api.github.com/repos/dusty-nv/ODM/git/refs/heads/master /tmp/odm_version.json
    
RUN env && \
    git clone --recursive --depth=1 https://github.com/dusty-nv/ODM ${ODM_HOME} && \
    cd ${ODM_HOME} && \
    PORTABLE_INSTALL=YES GPU_INSTALL=YES bash configure.sh install || echo "BUILD FAILED"
  

# https://github.com/rasterio/rasterio/issues/2333
# https://github.com/Toblerity/Fiona/issues/1043
#RUN pip3 install --no-cache-dir --verbose 'Cython<0.29.21'

#ENTRYPOINT ["python3", "/code/run.py"]
