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
cd "${COMFYUI_PATH}/custom_nodes"
if [ ! -d ComfyUI-Manager ]; then
  echo "== Cloning ComfyUI-Manager"
  git clone https://github.com/ltdrdata/ComfyUI-Manager.git || error_exit "ComfyUI-Manager clone failed"
fi
if [ ! -d ComfyUI-Manager ]; then error_exit "ComfyUI-Manager not found"; fi

# ComfyUI-Manager v0.3.76+ uses user/__manager/config.ini (older used user/default/ComfyUI-Manager/)
CONFIG_INI="${COMFYUI_PATH}/user/__manager/config.ini"
if [ ! -f "${CONFIG_INI}" ]; then
  echo "== ComfyUI-Manager config not yet generated (user/__manager/config.ini). Security level will apply on first run."
else
  echo "== Updating ComfyUI-Manager security level (container: expose WebUI to 0.0.0.0)"
  perl -p -i -e "s%security_level = normal%security_level = weak%g" "${CONFIG_INI}"
  perl -p -i -e "s%security_level = strict%security_level = weak%g" "${CONFIG_INI}"
fi
