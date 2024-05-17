#!/usr/bin/env bash
set -ex

ROOT="$(dirname "$(readlink -f "$0")")"

echo "Installing dependencies for opencv ${OPENCV_VERSION}"
$ROOT/install_deps.sh

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of opencv-python ${OPENCV_VERSION}"
	exit 1
fi

if [ ! -z "$OPENCV_URL" ]; then
    echo "Installing opencv ${OPENCV_VERSION} from deb packages"
    $ROOT/install_deb.sh
else
    echo "Installing opencv ${OPENCV_VERSION} from pip"
    $ROOT/install_pip.sh
fi

pip3 install --no-cache-dir --verbose opencv-contrib-python~=${OPENCV_VERSION}

python3 -c "import cv2; print('OpenCV version:', str(cv2.__version__)); print(cv2.getBuildInformation())"

