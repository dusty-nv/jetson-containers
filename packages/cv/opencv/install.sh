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
# Build containers often have no GPU driver (buildkit, off-device builds), so the CUDA wheel
# cannot be imported there: it links libcuda.so.1, which the NVIDIA runtime provides at run time.
# Any other unresolved library fails, including libnvcuvid.so.1 / libnvidia-encode.so.1: JetPack 6
# does not ship them, so a wheel linking them cannot load on the device (falls back to build.sh).
CV2_DIR="$(python3 -c 'import importlib.util, os; print(os.path.dirname(importlib.util.find_spec("cv2").origin))')"
CV2_SO="$(ls "$CV2_DIR"/cv2*.so "$CV2_DIR"/python-3*/cv2*.so 2>/dev/null | head -1)"
MISSING="$(ldd "$CV2_SO" | awk '/not found/ {print $1}' | sort -u)"
NOT_DRIVER="$(echo "$MISSING" | grep -vx 'libcuda\.so\.1' || true)"
if [ -n "$NOT_DRIVER" ]; then
    echo "ERROR: $CV2_SO has unresolved libraries: $NOT_DRIVER" >&2
    exit 1
elif [ -n "$MISSING" ]; then
    echo "skipping 'import cv2': no GPU driver in the build container ($MISSING)"
else
    python3 -c "import cv2; print('OpenCV version:', str(cv2.__version__)); print(cv2.getBuildInformation())"
fi

if [ -n "${CONSTRAINT_PKG:-}" ]; then
    ${PIP_APPEND_CONSTRAINT} "${CONSTRAINT_PKG}"
fi

echo "installed" > "$ROOT/.opencv"
