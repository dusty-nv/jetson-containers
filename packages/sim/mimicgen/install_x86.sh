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
    cd $WORKDIR
    git clone https://github.com/dusty-nv/robosuite
    apt_update
    apt-get install -y --no-install-recommends libhidapi-dev libosmesa6-dev python3-dev python3-pip
    cd robosuite
    pip3 install --verbose -e .
    pip3 install --verbose --no-cache-dir imageio[ffmpeg] pyspacemouse opencv-python
    cd /
fi

echo "> TESTING robosuite"
#python3 $SCRIPT_DIR/../robosuite/test.py \
#    --robots Panda --grippers PandaGripper \
#    --cameras agentview --camera-width 224 --camera-height 224 \
#    --output $WORKDIR/robosuite/output/test

if [ ! -d "robomimic" ]; then
    echo "> INSTALLING robomimic"
    cd $WORKDIR
    git clone https://github.com/ARISE-Initiative/robomimic/
    cd robomimic
    pip3 install --verbose cmake ninja torch==2.2
    pip3 install --verbose -e .
    cd /
fi

python3 -c 'import robomimic;  print("robomimic version: ", robomimic.__version__)'
python3 -c 'import robosuite; print("robosuite version:", robosuite.__version__)'
python3 -c 'from robomimic.envs.env_robosuite import EnvRobosuite'

if [ ! -d "mimicgen" ]; then
    echo "> INSTALLING mimicgen"
    cd $WORKDIR
    git clone https://github.com/dusty-nv/mimicgen
    cd mimicgen
    pip3 install --verbose -e .
    cd /
fi

python3 -c 'import mimicgen;  print("mimicgen version: ", mimicgen.__version__)'
python3 -c 'import robomimic;  print("robomimic version: ", robomimic.__version__)'
python3 -c 'import robosuite; print("robosuite version:", robosuite.__version__)'

python3 $SCRIPT_DIR/test.py \
    --robots Panda --grippers PandaGripper \
    --cameras agentview --camera-width 224 --camera-height 224 \
    --output $WORKDIR/mimicgen/output/test
