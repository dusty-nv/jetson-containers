#!/usr/bin/env bash
# this script installs OpenCV from deb packages that it downloads
# the opencv_version.sh script selects which packages to use

set -e -x

OPENCV_URL=$1
OPENCV_DEB=$2

echo "OPENCV_URL = $OPENCV_URL"
echo "OPENCV_DEB = $OPENCV_DEB"

ARCH=$(uname -i)
echo "ARCH:  $ARCH"

# remove previous OpenCV installation if it exists
apt-get purge -y '*opencv*' || echo "previous OpenCV installation not found"

# download and extract the deb packages
mkdir opencv
cd opencv
wget --quiet --show-progress --progress=bar:force:noscroll --no-check-certificate ${OPENCV_URL} -O ${OPENCV_DEB}
tar -xzvf ${OPENCV_DEB}

# install the packages and their dependencies
dpkg -i --force-depends *.deb
apt-get update 
apt-get install -y -f --no-install-recommends
dpkg -i *.deb
rm -rf /var/lib/apt/lists/*
apt-get clean

# remove the original downloads
cd ../
rm -rf opencv

# manage some paths on aarch64
if [ $ARCH = "aarch64" ]; then
    PYTHON3_VERSION=`python3 -c 'import sys; version=sys.version_info[:3]; print("{0}.{1}".format(*version))'`
    cp -r /usr/include/opencv4 /usr/local/include/opencv4
    cp -r /usr/lib/python${PYTHON3_VERSION}/dist-packages/cv2 /usr/local/lib/python${PYTHON3_VERSION}/dist-packages/cv2
fi

# test importing cv2
echo "testing cv2 module under python..."
python3 -c "import cv2; print('OpenCV version:', str(cv2.__version__)); print(cv2.getBuildInformation())"
