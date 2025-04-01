#!/usr/bin/env bash
# https://stackoverflow.com/a/4319666
shopt -s huponexit

KOKORO_DEFAULT_CMD="uvicorn api.src.main:app --reload --host 0.0.0.0 --port 8880"
KOROKO_STARTUP_LAG=1

printf "Starting Kokoro-TTS FastAPI server:\n\n"
printf "  ${KOKORO_DEFAULT_CMD}\n\n"

if [ "$#" -gt 0 ]; then
    ${KOKORO_DEFAULT_CMD} &
    #echo "Letting server load for ${KOROKO_STARTUP_LAG} seconds..."
    echo ""
    sleep ${KOROKO_STARTUP_LAG}
    echo ""
    echo "Running command:  $@"
    echo ""
    sleep 1
    "$@"
else
    ${KOKORO_DEFAULT_CMD}
fi
