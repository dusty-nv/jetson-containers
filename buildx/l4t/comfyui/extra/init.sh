#!/bin/bash

set -e

error_exit() {
  echo $*
}

export COMFYUI_PATH=`pwd`

# Run the Python script with a timeout of 20 seconds
echo "== Starting ComfyUI Manager to configure automatically config.ini"
timeout 60s python3 main.py --listen 0.0.0.0 --port ${PORT:-8188} || error_exit "Done"
echo "-- COMFYUI_PATH: ${COMFYUI_PATH}"

# Install ComfyUI Manager if not already present
cd custom_nodes
if [ ! -d ComfyUI-Manager ]; then
  echo "== Cloning ComfyUI-Manager"
  git clone https://github.com/ltdrdata/ComfyUI-Manager.git || error_exit "ComfyUI-Manager clone failed"
fi
if [ ! -d ComfyUI-Manager ]; then error_exit "ComfyUI-Manager not found"; fi
cd /opt/ComfyUI/user/default/ComfyUI-Manager/
if [ ! -f config.ini ]; then
  echo "== You will need to run ComfyUI-Manager a first time for the configuration file to be generated, we can not attempt to update its security level yet"
else
  echo "== Attempting to update ComfyUI-Manager security level (running in a container, we need to expose the WebUI to 0.0.0.0)"
  perl -p -i -e "s%security_level = normal%security_level = weak%g" config.ini
  perl -p -i -e "s%security_level = strict%security_level = weak%g" config.ini
fi
