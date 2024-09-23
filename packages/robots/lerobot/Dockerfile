#---
# name: lerobot
# group: robots
# docs: docs.md
# depends: [transformers, opencv:4.10.0, pyav, h5py]
# requires: '>=36'
# test: [test.sh, test.py]
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG LEROBOT_REPO=huggingface/lerobot/
ARG LEROBOT_BRANCH=main

RUN git clone --branch=${LEROBOT_BRANCH} --depth=1 https://github.com/${LEROBOT_REPO} /opt/lerobot && \
    cd /opt/lerobot && \
    sed 's|^python.*||' -i pyproject.toml && \
    sed 's|^torch.*||' -i pyproject.toml && \
    sed 's|^opencv-python.*||' -i pyproject.toml && \
    sed 's|^torchvision.*||' -i pyproject.toml && \
    sed 's|^h5py.*||' -i pyproject.toml && \
    sed 's|^pyav.*||' -i pyproject.toml && \
    sed 's|^huggingface-hub.*||' -i pyproject.toml && \
    echo "######### pyproject.toml ##########" && \
    cat -n pyproject.toml

RUN cd /opt/lerobot && \
    pip3 install --ignore-installed -e ".[aloha, pusht]"
    
RUN cd /opt/lerobot && \
    pip install -e ".[dynamixel]"