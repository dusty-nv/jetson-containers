#!/usr/bin/env bash
OPENCV_URL=${1:-"$OPENCV_URL"}
OPENCV_DEB=${2:-"OpenCV-${OPENCV_VERSION}-aarch64.tar.gz"}

echo "OPENCV_URL=$OPENCV_URL"
echo "OPENCV_DEB=$OPENCV_DEB"

if [[ -z ${OPENCV_URL} || -z ${OPENCV_DEB} ]]; then
	echo "OPENCV_URL and OPENCV_DEB must be set as environment variables or as command-line arguments"
	exit 255
fi

ARCH=$(uname -i)
echo "ARCH:  $ARCH"
set -ex

# remove previous OpenCV installation if it exists
apt-get purge -y '.*opencv.*' || echo "previous OpenCV deb installation not found"
uv pip uninstall opencv-python || echo "previous OpenCV pip installation not found"

# make sure cmake and numpy are still installed
bash /tmp/cmake/install.sh
bash /tmp/numpy/install.sh

# download and extract the deb packages
mkdir opencv
cd opencv

echo "Downloading OpenCV archive from ${OPENCV_URL}..."
if ! wget $WGET_FLAGS "${OPENCV_URL}" -O "${OPENCV_DEB}"; then
    echo "❌ ERROR: Failed to download OpenCV archive from ${OPENCV_URL}"
    exit 1
fi

echo "✅ Successfully downloaded ${OPENCV_DEB}"
echo "Extracting ${OPENCV_DEB}..."
tar -xzvf "${OPENCV_DEB}"

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

# restore cmake and numpy versions
bash /tmp/cmake/install.sh
bash /tmp/numpy/install.sh

# manage some install paths
PYTHON3_VERSION=`python3 -c 'import sys; version=sys.version_info[:3]; print("{0}.{1}".format(*version))'`

if [ $ARCH = "aarch64" ]; then
	local_include_path="/usr/local/include/opencv4"
	local_python_path="/usr/local/lib/python${PYTHON3_VERSION}/dist-packages/cv2"

	if [ -d "$local_include_path" ]; then
		echo "$local_include_path already exists, replacing..."
		rm -rf $local_include_path
	fi

	if [ -d "$local_python_path" ]; then
		echo "$local_python_path already exists, replacing..."
		rm -rf $local_python_path
	fi

	ln -sfnv /usr/include/opencv4 $local_include_path
	ln -sfnv /usr/lib/python${PYTHON3_VERSION}/dist-packages/cv2 $local_python_path
fi

