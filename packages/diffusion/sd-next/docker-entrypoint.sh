#!/usr/bin/env bash

set -e

cd /opt/automatic

if [[ ! -z "${ACCELERATE}" ]] && [ ${ACCELERATE}="True" ] && [ -x "$(command -v accelerate)" ]
then
    echo "Launching accelerate launch.py..."
    exec accelerate launch --num_cpu_threads_per_process=6 launch.py --data=/data/models/stable-diffusion --skip-all --use-xformers --use-cuda --listen --port=7860 "$@"
else
    echo "Launching launch.py..."
    exec python3 launch.py --data=/data/models/stable-diffusion --skip-all --use-xformers --use-cuda --listen --port=7860 "$@"
fi
