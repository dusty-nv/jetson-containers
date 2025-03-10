#!/usr/bin/env bash
# https://stackoverflow.com/a/4319666
shopt -s huponexit
set -e

function run_process() 
{
    local cmd=$1
    local timeout=${2:-10}
    local daemon=${3:-false}

    if [ $daemon = true ]; then
        printf "\nStarting daemon:   $cmd\n"
        $cmd &
        sleep $timeout
    else
        printf "\nStarting process:  $cmd\n"
        $cmd
    fi
}

function run_daemon() 
{
    run_process "$1" ${2:-10} true
}

printf "\nStarting robopoint:\n"
printf "   ROBOPOINT_HOST=$ROBOPOINT_HOST\n"
printf "   ROBOPOINT_PORT=$ROBOPOINT_PORT\n"
printf "   ROBOPOINT_MODEL=$ROBOPOINT_MODEL\n"
printf "   ROBOPOINT_QUANTIZATION=$ROBOPOINT_QUANTIZATION\n\n"

ROBOPOINT_MODEL_DIR=$(huggingface-downloader $ROBOPOINT_MODEL)

if [ $ROBOPOINT_QUANTIZATION = "int4" ]; then
    ROBOPOINT_FLAGS="--load-4bit"
elif [ $ROBOPOINT_QUANTIZATION = "int8" ]; then
    ROBOPOINT_FLAGS="--load-8bit"
fi

run_daemon "python3 -m robopoint.serve.controller \
  --host $ROBOPOINT_HOST \
  --port $ROBOPOINT_PORT"

run_daemon "python3 -m robopoint.serve.gradio_web_server \
  --host $ROBOPOINT_HOST \
  --controller http://localhost:$ROBOPOINT_PORT \
  --model-list-mode reload \
  --share"

run_process "python3 -m robopoint.serve.model_worker \
  --host $ROBOPOINT_HOST \
  --controller http://localhost:$ROBOPOINT_PORT \
  --port 20000 \
  --worker http://localhost:20000 \
  --model-path $ROBOPOINT_MODEL_DIR \
  --model-name $(basename $ROBOPOINT_MODEL) \
  $ROBOPOINT_FLAGS $@"

# this is for if it were setup as ENTRYPOINT
#if [ "$#" = 0 ]; then
#    fg
#else
#    "$@"
#fi
