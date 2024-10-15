#---
# name: lerobot
# group: robots
# docs: docs.md
# depends: [transformers, opencv:4.10.0, pyav, h5py, jupyterlab:myst]
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

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
            speech-dispatcher speech-dispatcher-espeak-ng pulseaudio-utils alsa-utils vim \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

ENV JUPYTER_ROOT=/opt/lerobot

# Create a symbolic link for python3 to python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Verify the symlink is created and the version of Python
RUN python --version

WORKDIR /opt/lerobot

RUN echo "pactl info" >> /root/.bash_history
RUN echo "pactl list short sinks" >> /root/.bash_history
RUN echo "pactl set-default-sink 0" >> /root/.bash_history
RUN echo "wandb login"
RUN echo "export HF_USER="
RUN echo "python lerobot/scripts/control_robot.py record \
  --robot-path lerobot/configs/robot/koch.yaml \
  --fps 30 \
  --root data \
  --repo-id ${HF_USER}/koch_test_$(date +%Y%m%d_%H%M%S) \
  --tags tutorial \
  --warmup-time-s 5 \
  --episode-time-s 30 \
  --reset-time-s 30 \
  --num-episodes 10"

RUN echo -e "* soft core 0\n* hard core 0" >> /etc/security/limits.conf

CMD /start_jupyter && speech-dispatcher --spawn  && /bin/bash