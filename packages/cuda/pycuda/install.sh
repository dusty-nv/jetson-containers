#!/usr/bin/env bash
set -ex

if [ "on" == "on" ]; then
	echo "Forcing build of pycuda"
	exit 1
fi

pip3 install -U pycuda
