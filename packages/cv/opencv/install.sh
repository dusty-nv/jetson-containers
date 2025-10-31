#!/usr/bin/env bash
set -ex

ROOT="$(dirname "$(readlink -f "$0")")"

#echo "Installing dependencies for opencv ${OPENCV_VERSION}"
#$ROOT/install_deps.sh

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of opencv-python ${OPENCV_VERSION}"
	exit 1
fi

if [ ! -z "$OPENCV_URL" ]; then
    echo "Installing opencv ${OPENCV_VERSION} from deb packages"
    $ROOT/install_deb.sh
else
    echo "Installing opencv ${OPENCV_VERSION} from pip"
    export OPENCV_DEB="OpenCV-${OPENCV_VERSION}.tar.gz"
    export OPENCV_URL=${TAR_INDEX_URL}/${OPENCV_DEB}
    $ROOT/install_deb.sh
    uv pip install opencv-contrib-python~=${OPENCV_VERSION}
fi

# In buildkit=1 mode, we cannot test the installation here
# python3 -c "import cv2; print('OpenCV version:', str(cv2.__version__)); print(cv2.getBuildInformation())"
echo "installed" > "$ROOT/.opencv"
