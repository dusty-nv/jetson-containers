#---
# name: lerobot
# group: robots
# docs: docs.md
# depends: [transformers, opencv:4.11.0, pyav, h5py, jupyterlab:myst]
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
    pip3 install --ignore-installed -e ".[aloha, pusht, pi0]" --verbose

RUN cd /opt/lerobot && \
    pip install -e ".[dynamixel]" --verbose

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
RUN echo "python lerobot/scripts/control_robot.py
  --robot.type=koch \
  --control.type=record \
  --control.single_task="Grasp a lego block and put it in the bin." \
  --control.fps=30 \
  --control.repo_id=\${HF_USER}/koch_test_\$(date +%Y%m%d_%H%M%S) \
  --control.tags='["tutorial"]' \
  --control.warmup_time_s=5 \
  --control.episode_time_s=30 \
  --control.reset_time_s=30 \
  --control.num_episodes=10 \
  --control.push_to_hub=true" >> /root/.bash_history

RUN echo -e "* soft core 0\n* hard core 0" >> /etc/security/limits.conf

CMD /start_jupyter && speech-dispatcher --spawn  && /bin/bash