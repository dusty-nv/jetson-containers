#---
# name: lerobot
# group: robots
# docs: docs.md
# depends: [transformers, opencv:4.11.0, pyav, h5py, diffusers, jupyterlab:myst]
# requires: '>=36'
# test: [test.sh, test.py]
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG LEROBOT_REPO=huggingface/lerobot/
ARG LEROBOT_BRANCH=main

RUN git clone --branch=${LEROBOT_BRANCH} --depth=1 https://github.com/${LEROBOT_REPO} /opt/lerobot && \
    cd /opt/lerobot && \
    sed -i '/"torch[^"]*",/d' pyproject.toml && \
    sed -i '/"opencv-python[^"]*",/d' pyproject.toml && \
    sed -i '/"torchvision[^"]*",/d' pyproject.toml && \
    sed -i '/"h5py[^"]*",/d' pyproject.toml && \
    sed -i '/"av[^"]*",/d' pyproject.toml && \
    sed -i '/"huggingface-hub[^"]*",/d' pyproject.toml && \
    sed -i '/"diffusers[^"]*",/d' pyproject.toml && \   
    cat -n pyproject.toml

RUN cd /opt/lerobot && \
    pip3 install --ignore-installed --no-deps -e ".[aloha, pusht, dynamixel, pi0, test]" --verbose

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

## Prevent the Gdk-ERROR by disabling MIT-SHM
RUN export GDK_BACKEND=x11
RUN export GDK_USE_X11SHM=0
RUN export GDK_SYNCHRONIZE=1

RUN echo "pactl info" >> /root/.bash_history
RUN echo "pactl list short sinks" >> /root/.bash_history
RUN echo "pactl set-default-sink 0" >> /root/.bash_history
RUN echo "wandb login" >> /root/.bash_history
RUN echo "export HF_USER=" >> /root/.bash_history
RUN echo "python lerobot/scripts/control_robot.py record \
  --robot-path lerobot/configs/robot/koch.yaml \
  --fps 30 \
  --root data \
  --repo-id \${HF_USER}/koch_test_\$(date +%Y%m%d_%H%M%S) \
  --tags tutorial \
  --warmup-time-s 5 \
  --episode-time-s 30 \
  --reset-time-s 30 \
  --num-episodes 10" >> /root/.bash_history

RUN echo -e "* soft core 0\n* hard core 0" >> /etc/security/limits.conf

CMD /start_jupyter && speech-dispatcher --spawn  && /bin/bash