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

    if [[ "$OPENCV_VERSION" == "4.x" ]]; then
        uv pip install --no-deps opencv-contrib-python-rolling
        CONSTRAINT_PKG=opencv-contrib-python-rolling
    else
        uv pip install --no-deps "opencv-contrib-python~=${OPENCV_VERSION}"
        CONSTRAINT_PKG=opencv-contrib-python
    fi
fi

# Verify the install actually works before pinning a constraint -
# otherwise a failed install leaves a bad pin in pip-constraints.txt
# that blocks build.sh's locally-built wheel (different version).
python3 -c "import cv2; print('OpenCV version:', str(cv2.__version__)); print(cv2.getBuildInformation())"

if [ -n "${CONSTRAINT_PKG:-}" ]; then
    ${PIP_APPEND_CONSTRAINT} "${CONSTRAINT_PKG}"
fi

echo "installed" > "$ROOT/.opencv"
