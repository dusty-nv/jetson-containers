#!/usr/bin/env bash
set -ex

apt-get update
apt-get install -y --no-install-recommends \
    libjpeg-dev \
    libglm-dev \
    libgl1 \
    libglx-mesa0 \
    libegl1-mesa-dev \
    mesa-utils \
    xorg-dev \
    freeglut3-dev
rm -rf /var/lib/apt/lists/*
apt-get clean

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of habitat-sim ${HABITAT_SIM_VERSION} (branch=${HABITAT_SIM_BRANCH})"
	exit 1
fi

uv pip install habitat-sim==${HABITAT_SIM_VERSION}
