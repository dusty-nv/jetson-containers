#!/usr/bin/env bash

set -e

jupyter lab --ip 0.0.0.0 --port 8888 --allow-root &> /var/log/jupyter.log &
echo "allow 10 sec for JupyterLab to start @ http://$(hostname -I | cut -d' ' -f1):8888 (password nvidia)"

echo "JupterLab logging location:  /var/log/jupyter.log  (inside the container)"
exec "$@"
