#---
# name: opendronemap
# group: robots
# depends: [onnxruntime, opencv:4.10.0, ffmpeg]
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
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# https://github.com/rasterio/rasterio/issues/2333
# https://github.com/Toblerity/Fiona/issues/1043
#RUN pip3 install --no-cache-dir --verbose 'Cython<0.29.15'
    
RUN env && \
    git clone --recursive --depth=1 https://github.com/OpenDroneMap/ODM ${ODM_HOME} && \
    cd ${ODM_HOME} && \
    sed 's|    check_version||' -i configure.sh && \
    sed 's|^rasterio==.*|rasterio|g' -i requirements.txt && \
    cat configure.sh | grep 'check_version' && \
    cat requirements.txt && \
    PORTABLE_INSTALL=YES GPU_INSTALL=YES bash configure.sh install
    
ENTRYPOINT ["python3", "${ODM_HOME}/run.py"]
