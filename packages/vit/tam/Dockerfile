#---
# name: tam
# group: vit
# depends: [pytorch:distributed, sam]
# requires: '>=34.1.0'
# docs: docs.md
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /opt

# Clone the repository:
RUN git clone https://github.com/gaomingqi/Track-Anything.git && \
    cd Track-Anything && \
    sed 's|^numpy||g' -i requirements.txt && \
    sed 's|^git+https://github.com/facebookresearch/segment-anything.git||g' -i requirements.txt && \
    sed 's|^opencv-python||g' -i requirements.txt && \
    pip3 install -r requirements.txt 

RUN LD_LIBRARY_PATH=/usr/local/lib/python3.8/dist-packages/opencv_python.libs/:$LD_LIBRARY_PATH

RUN cd /opt/Track-Anything && \
    sed 's|cuda:3|cuda:0|g' -i app.py

# Dependencies and Installation
RUN pip3 install opencv-python==4.5.4.60

ENV LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/python3.8/dist-packages/opencv_python.libs/

RUN mim install mmcv --verbose --no-cache-dir
RUN cd /opt && ./opencv_install.sh

# https://github.com/dusty-nv/jetson-containers/issues/326
RUN pip3 install --no-cache-dir --verbose gradio==3.38.0

# cache the checkpoints under /data
RUN mkdir -p /data/models/tam && \
    ln -s /data/models/tam /opt/Track-Anything/checkpoints
    
WORKDIR /

RUN cd /opt/Track-Anything && python3 app.py --help
CMD cd /opt/Track-Anything && python3 app.py