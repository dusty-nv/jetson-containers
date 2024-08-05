#!/usr/bin/env bash
# this script is for installing openvla on a training server or workstation with dGPU(s)
#   wget -O - https://raw.githubusercontent.com/dusty-nv/jetson-containers/dev/packages/sim/mimicgen/install_x86.sh | bash
set -ex

WORKDIR="/workspace"
mkdir -p $WORKDIR
cd $WORKDIR

APT_UPDATED=false

function apt_update() {
    if [ ! $APT_UPDATED ]; then
        apt-get update
        APT_UPDATED=true
    fi
}

if [ ! -d "mimicgen" ]; then
    echo "> INSTALLING robosuite"
    git clone https://github.com/dusty-nv/robosuite
    apt_update
    apt-get install -y --no-install-recommends libhidapi-dev
    cd robosuite
    pip3 install --verbose -e .
    pip3 install --verbose --no-cache-dir imageio[ffmpeg] pyspacemouse 
    cd ../
fi

echo "> TESTING robosuite"
wget https://github.com/dusty-nv/jetson-containers/blob/dev/packages/sim/robosuite/test.py -O $WORKSPACE/robosuite/generate.py
python3 $WORKSPACE/robosuite/generate.py --output $WORKSPACE/robosuite/output/test
