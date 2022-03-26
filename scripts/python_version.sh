#!/usr/bin/env bash

#LSB_RELEASE=$(lsb_release --codename --short)

#echo "determining default Python3 version for '$LSB_RELEASE'"

#if [ $LSB_RELEASE = "bionic" ]; then
#	PYTHON3_VERSION="3.6"
#elif [ $LSB_RELEASE = "focal" ]; then
#	PYTHON3_VERSION="3.8"
#else
#	echo "unsupported distro version:  $LSB_RELEASE"
#	exit 1
#fi

PYTHON3_VERSION=`python3 -c 'import sys; version=sys.version_info[:3]; print("{0}.{1}".format(*version))'`

echo "Python3 version:  $PYTHON3_VERSION"
