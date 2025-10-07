#!/usr/bin/env bash
set -ex

apt-get update
apt-get install -y --no-install-recommends \
    cmake \
    build-essential \
    libjpeg-dev \
    libglm-dev \
    libgl1 \
    libglx-mesa0 \
    libegl1-mesa-dev \
    mesa-utils \
    xorg-dev \
    freeglut3-dev \
rm -rf /var/lib/apt/lists/*
apt-get clean

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of isaacsim ${ISAACSIM_VERSION}"
	exit 1
fi


uv pip install \
  isaacsim-app==${ISAACSIM_VERSION} \
  isaacsim-asset==${ISAACSIM_VERSION} \
  isaacsim-benchmark==${ISAACSIM_VERSION} \
  isaacsim-code-editor==${ISAACSIM_VERSION} \
  isaacsim-core==${ISAACSIM_VERSION} \
  isaacsim-cortex==${ISAACSIM_VERSION} \
  isaacsim-example==${ISAACSIM_VERSION} \
  isaacsim-extscache-kit-sdk==${ISAACSIM_VERSION} \
  isaacsim-extscache-kit==${ISAACSIM_VERSION} \
  isaacsim-extscache-physics==${ISAACSIM_VERSION} \
  isaacsim-gui==${ISAACSIM_VERSION} \
  isaacsim-kernel==${ISAACSIM_VERSION} \
  isaacsim-replicator==${ISAACSIM_VERSION} \
  isaacsim-rl==${ISAACSIM_VERSION} \
  isaacsim-robot-motion==${ISAACSIM_VERSION} \
  isaacsim-robot-setup==${ISAACSIM_VERSION} \
  isaacsim-robot==${ISAACSIM_VERSION} \
  isaacsim-ros1==${ISAACSIM_VERSION} \
  isaacsim-ros2==${ISAACSIM_VERSION} \
  isaacsim-sensor==${ISAACSIM_VERSION} \
  isaacsim-storage==${ISAACSIM_VERSION} \
  isaacsim-template==${ISAACSIM_VERSION} \
  isaacsim-test==${ISAACSIM_VERSION} \
  isaacsim-utils==${ISAACSIM_VERSION} \
  isaacsim==${ISAACSIM_VERSION}
