#!/usr/bin/env bash
# this script is for installing openvla on a training server or workstation with dGPU(s)
#   git clone https://github.com/dusty-nv/jetson-containers
#   bash jetson-containers/packages/sim/mimicgen/install_x86.sh
set -ex

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
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

if [ ! -d "robosuite" ]; then
    echo "> INSTALLING robosuite"
    git clone https://github.com/dusty-nv/robosuite
    apt_update
    apt-get install -y --no-install-recommends libhidapi-dev libglvnd-dev
    cd robosuite
    pip3 install --verbose -e .
    pip3 install --verbose --no-cache-dir imageio[ffmpeg] pyspacemouse opencv-python
    cd ../
fi

echo "> TESTING robosuite"
python3 $SCRIPT_DIR/../robosuite/test.py --output $WORKDIR/robosuite/output/test
