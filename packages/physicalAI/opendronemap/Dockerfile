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
		  libtbb-dev \
		&& apt-mark hold libtbb12 libtbb-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# https://cmake.org/cmake/help/latest/policy/CMP0146.html
RUN uv pip install 'cmake<3.27'

ADD https://api.github.com/repos/OpenDroneMap/ODM/git/refs/heads/master /tmp/odm_version.json

RUN env && \
    git clone --recursive --depth=1 https://github.com/OpenDroneMap/ODM ${ODM_HOME} && \
    cd ${ODM_HOME} && \
    sed -i 's/==/>=/g' requirements.txt && \
    uv pip install python-dateutil && \
    sed -i 's|https://github.com/opencv/opencv/archive/4\.12\.0\.zip|https://github.com/opencv/opencv/archive/4.13.0.zip|' ${ODM_HOME}/SuperBuild/cmake/External-OpenCV.cmake && \
    PORTABLE_INSTALL=YES GPU_INSTALL=YES bash configure.sh install || echo "BUILD FAILED"


# https://github.com/rasterio/rasterio/issues/2333
# https://github.com/Toblerity/Fiona/issues/1043
#RUN uv pip install 'Cython<0.29.21'

#ENTRYPOINT ["python3", "/code/run.py"]
