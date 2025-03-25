#!/bin/bash
# https://stackoverflow.com/a/4319666
shopt -s huponexit

AIM_SERVER_CMD="aim server -y --host ${SERVER_HOST} --port ${SERVER_PORT} --repo /repo"
AIM_WEBSERVER_CMD="aim up --host ${WEB_HOST} --port ${WEB_PORT} --repo /repo"
AIM_STARTUP_LAG=5

printf "\nStarting aim server:\n\n"
printf "${AIM_SERVER_CMD}\n\n"

${AIM_SERVER_CMD} 2>&1 &
echo ""
sleep ${AIM_STARTUP_LAG}

printf "\nStarting aim webserver:\n\n"
printf "${AIM_WEBSERVER_CMD}\n\n"

if [ "$#" -gt 0 ]; then
    ${AIM_WEBSERVER_CMD} 2>&1 &
    echo ""
    sleep ${AIM_STARTUP_LAG}
    echo "Running user command:  $@"
    echo ""
    sleep 1
    "$@"
else
    ${AIM_WEBSERVER_CMD}
fi
