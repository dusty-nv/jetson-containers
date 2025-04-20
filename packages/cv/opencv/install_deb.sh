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

set -x

# install numpy if needed
# Check if NumPy is installed
python3 -c 'import numpy; print("NumPy version before installation:", numpy.__version__)' 2>/dev/null

if [ $? != 0 ]; then
    echo "NumPy not found. Installing NumPy 2.0..."
    apt-get update
    # apt-get install -y --no-install-recommends python3-numpy
    python3 -m pip install "numpy>=2.0.0" --break-system-packages
fi

# Print the installed version after installation
python3 -c 'import numpy; print("NumPy version after installation:", numpy.__version__)'

set -e

# remove previous OpenCV installation if it exists
apt-get purge -y '.*opencv.*' || echo "previous OpenCV installation not found"

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

	ln -s /usr/include/opencv4 $local_include_path
	ln -s /usr/lib/python${PYTHON3_VERSION}/dist-packages/cv2 $local_python_path

elif [ $ARCH = "x86_64" ]; then
	opencv_conda_path="/opt/conda/lib/python${PYTHON3_VERSION}/site-packages/cv2"

	if [ -d "$opencv_conda_path" ]; then
		echo "$opencv_conda_path already exists, replacing..."
		rm -rf $opencv_conda_path
		ln -s /usr/lib/python${PYTHON3_VERSION}/site-packages/cv2 $opencv_conda_path
	fi
fi

