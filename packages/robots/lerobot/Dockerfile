#---
# name: lerobot
# group: robots
# docs: docs.md
# depends: [transformers, opencv, pyav, h5py, diffusers, jupyterlab:myst]
# requires: '>=36'
# test: [test.sh, test.py]
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /opt/lerobot

ARG LEROBOT_REPO=huggingface/lerobot \
    LEROBOT_BRANCH=main

# Prevent the Gdk-ERROR by disabling MIT-SHM
ENV GDK_BACKEND=x11 \
    GDK_USE_X11SHM=0 \
    GDK_SYNCHRONIZE=1 \
    JUPYTER_ROOT=/opt/lerobot

# Clone and build LeRobot
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
    uv pip install --no-reinstall --no-deps -e ".[feetech, aloha, pusht, dynamixel, pi0, test]" --verbose

# Install apt dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
            speech-dispatcher speech-dispatcher-espeak-ng pulseaudio-utils alsa-utils vim \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create a symbolic link for python3 to python
RUN ln -sfnv -s /usr/bin/python3 /usr/bin/python || true && python --version

# Setup various system configs
RUN echo -e "* soft core 0\n* hard core 0" >> /etc/security/limits.conf && \
    echo "pactl info" >> /root/.bash_history && \
    echo "pactl list short sinks" >> /root/.bash_history && \
    echo "pactl set-default-sink 0" >> /root/.bash_history && \
    echo "wandb login" >> /root/.bash_history && \
    echo "export HF_USER=" >> /root/.bash_history && \
    echo "python lerobot/scripts/control_robot.py record \
  --robot-path lerobot/configs/robot/koch.yaml \
  --fps 30 \
  --root data \
  --repo-id \${HF_USER}/koch_test_\$(date +%Y%m%d_%H%M%S) \
  --tags tutorial \
  --warmup-time-s 5 \
  --episode-time-s 30 \
  --reset-time-s 30 \
  --num-episodes 10" >> /root/.bash_history

# Start JupyterLab server in the background by default
CMD /start_jupyter && speech-dispatcher --spawn  && /bin/bash
