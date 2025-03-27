#---
# name: lerobot
# group: robots
# docs: docs.md
# depends: [transformers, opencv:4.11.0, pyav, h5py, diffusers, jupyterlab:myst, numba]
# requires: '>=36'
# test: [test.sh, test.py]
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG LEROBOT_REPO=huggingface/lerobot/
ARG LEROBOT_BRANCH=main

RUN git clone --branch=${LEROBOT_BRANCH} --depth=1 https://github.com/${LEROBOT_REPO} /opt/lerobot && \
    cd /opt/lerobot && \
    echo "commit:   $(git rev-parse HEAD)" > lerobot_version.txt && \
    echo "date:     $(git log -1 --format=%cd)" >> lerobot_version.txt && \
    echo "branch:   $(git rev-parse --abbrev-ref HEAD)" >> lerobot_version.txt && \
    echo "describe: $(git describe --tags --always)" >> lerobot_version.txt && \
    sed -i '/"torch[^"]*",/d' pyproject.toml && \
    sed -i '/"opencv-python[^"]*",/d' pyproject.toml && \
    sed -i '/"torchvision[^"]*",/d' pyproject.toml && \
    sed -i '/"h5py[^"]*",/d' pyproject.toml && \
    sed -i '/"av[^"]*",/d' pyproject.toml && \
    sed -i '/"huggingface-hub[^"]*",/d' pyproject.toml && \
    sed -i '/"diffusers[^"]*",/d' pyproject.toml && \
    sed -i '/"rerun-sdk[^"]*",/d' pyproject.toml && \
    sed -i '/"numba[^"]*",/d' pyproject.toml && \
    sed -i '/dependencies = \[/a \ \ \ \ "numpy<2.0.0",' pyproject.toml && \
    cat -n pyproject.toml

RUN cd /opt/lerobot && \
    pip3 install --ignore-installed -e ".[aloha, pusht, dynamixel, pi0, test, feetech]" --verbose

RUN pip show datasets
RUN python3 -c "import datasets; print(datasets.__version__)"

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
RUN echo "export ROBOT_TYPE=so100 #or, koch" >> /root/.bash_history
RUN echo "python lerobot/scripts/control_robot.py \
    --robot.type=\${ROBOT_TYPE} \
    --control.type=record \
    --control.single_task=\"Grasp a lego block and put it in the bin.\" \
    --control.fps 30 \
    --control.repo_id ${HF_USER}/test_$(date +%Y%m%d_%H%M%S) \
    --control.tags '[\"tutorial\"]' \
    --control.warmup_time_s 5 \
    --control.episode_time_s 30 \
    --control.reset_time_s 30 \
    --control.num_episodes 10 \
    --control.push_to_hub=true" >> /root/.bash_history

RUN echo -e "* soft core 0\n* hard core 0" >> /etc/security/limits.conf

CMD /start_jupyter && speech-dispatcher --spawn  && /bin/bash