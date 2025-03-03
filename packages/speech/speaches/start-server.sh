#!/bin/bash
# https://stackoverflow.com/a/4319666
shopt -s huponexit

SPEACHES_DEFAULT_CMD="python3 -m uvicorn speaches.main:create_app --host 0.0.0.0 --port $PORT --factory"
SPEACHES_STARTUP_LAG=5

printf "Starting Speaches server:\n\n"
printf "  ${SPEACHES_DEFAULT_CMD}\n\n"

if [ "$#" -gt 0 ]; then
    ${SPEACHES_DEFAULT_CMD} 2>&1 &
    echo ""
    sleep ${SPEACHES_STARTUP_LAG}
    echo "Running command:  $@"
    echo ""
    sleep 1
    "$@"
else
    ${SPEACHES_DEFAULT_CMD}
fi
