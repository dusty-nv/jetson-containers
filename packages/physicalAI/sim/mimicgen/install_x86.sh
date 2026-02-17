#!/usr/bin/env bash
# this script is for installing openvla on a training server or workstation with dGPU(s)
# https://pytorch.org/rl/stable/reference/generated/knowledge_base/MUJOCO_INSTALLATION.html
#   git clone https://github.com/dusty-nv/jetson-containers
#   export MUJOCO_GL=osmesa
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
    apt-get install -y --no-install-recommends libhidapi-dev libosmesa6-dev python3-dev python3-pip
    cd robosuite
    uv pip install -e .
    uv pip install imageio[ffmpeg] pyspacemouse opencv-python
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
    uv pip install cmake ninja torch==2.2
    uv pip install -e .
fi

cd /

python3 -c 'import robomimic;  print("robomimic version: ", robomimic.__version__)'
python3 -c 'import robosuite; print("robosuite version:", robosuite.__version__)'
python3 -c 'from robomimic.envs.env_robosuite import EnvRobosuite'

cd $WORKDIR

if [ ! -d "mimicgen" ]; then
    echo "> INSTALLING mimicgen"
    git clone https://github.com/dusty-nv/mimicgen
    cd mimicgen
    uv pip install -e .
fi

cd /

python3 -c 'import mimicgen;  print("mimicgen version: ", mimicgen.__version__)'
python3 -c 'import robomimic;  print("robomimic version: ", robomimic.__version__)'
python3 -c 'import robosuite; print("robosuite version:", robosuite.__version__)'

python3 $SCRIPT_DIR/test.py \
    --robots Panda --grippers PandaGripper \
    --cameras agentview --camera-width 224 --camera-height 224 \
    --output $WORKDIR/mimicgen/output/test

cd $WORKDIR
