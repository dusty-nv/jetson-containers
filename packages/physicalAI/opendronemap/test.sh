#!/usr/bin/env bash
set -e
echo "testing OpenDroneMap..."

cd $ODM_HOME
bash run.sh --help

python3 opendm/context.py
python3 -c 'from opensfm import io, pymap'


printf "\nOpenDroneMap version:  $(cat $ODM_HOME/VERSION)\n"
