#---
# name: tam
# group: vit
# depends: [pytorch, sam]
# requires: '>=34.1.0'
# docs: docs.md
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /opt

# Clone the repository:
RUN git clone https://github.com/gaomingqi/Track-Anything.git && \
    cd Track-Anything && \
    sed -i 's/==/>=/g' requirements.txt && \
    sed -i 's/~=/>=/g' requirements.txt && \
    sed 's|^git+https://github.com/facebookresearch/segment-anything.git||g' -i requirements.txt && \
    sed 's|^opencv-python||g' -i requirements.txt && \
    uv pip install -r requirements.txt

RUN cd /opt/Track-Anything && \
    sed 's|cuda:3|cuda:0|g' -i app.py

RUN mim install mmcv --verbose --no-cache-dir
RUN /tmp/opencv/install.sh

# https://github.com/dusty-nv/jetson-containers/issues/326
RUN uv pip install gradio>=3.38.0

# cache the checkpoints under /data
RUN mkdir -p /data/models/tam && \
    ln -s /data/models/tam /opt/Track-Anything/checkpoints

WORKDIR /

RUN cd /opt/Track-Anything && python3 app.py --help
CMD cd /opt/Track-Anything && python3 app.py
