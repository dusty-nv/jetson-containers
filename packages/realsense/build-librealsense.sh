#!/bin/bash
# Builds the Intel Realsense library librealsense on a Jetson Nano Development Kit
# Copyright (c) 2016-21 Jetsonhacks 
# MIT License
# set -e 

LIBREALSENSE_DIRECTORY=${HOME}/librealsense
INSTALL_DIR=$PWD
NVCC_PATH=/usr/local/cuda/bin/nvcc

USE_CUDA=true

function usage ()
{
    echo "Usage: ./build-librealsense.sh [-n | -no_cuda] [-v | -version <version>] [-j | --jobs <number of jobs>] [-h | --help] "
    echo "-n  | --no_cuda   Build with no CUDA (Defaults to with CUDA)"
    echo "-v  | --version   Version of librealsense to build 
                      (defaults to latest release)"
    echo "-j  | --jobs      Number of concurrent jobs (Default 1 on <= 4GB RAM
                      #of cores-1 otherwise)"
    echo "-h  | --help      This message"
    exit 2
}

PARSED_ARGUMENTS=$(getopt -a -n build-librealsense.sh -o nv:j:h --longoptions version:,no_cuda,jobs:,help -- "$@" )
VALID_ARGUMENTS=$?

if [ "$VALID_ARGUMENTS" != "0" ]; then
   echo ""
   usage
fi

eval set -- "$PARSED_ARGUMENTS"

LIBREALSENSE_VERSION=""
USE_CUDA=true
NUM_PROCS=""

while :
do
   case "$1" in
      -n | --build_no_cuda) USE_CUDA=false   ; shift ;;
      -v | --version )      LIBREALSENSE_VERSION="$2" ; shift 2 ;;
      -j | --jobs)          NUM_PROCS="$2" ; 
                            shift 2 ;
                            re_isanum='^[0-9]+$'
                            if ! [[ $NUM_PROCS =~ $re_isanum ]] ; then
                              echo "Number of jobs must be a positive, whole number"
                              usage
                            else
                              if [ $NUM_PROCS -eq "0" ]; then
                                echo "Number of jobs must be a positive, whole number" 
                              fi
                            fi ;
       ;;
      -h | --help )         usage ; shift ;;
      # -- means the end of arguments
      --)  shift; break ;;
   esac
done

# From lukechilds gist discussion: https://gist.github.com/lukechilds/a83e1d7127b78fef38c2914c4ececc3c 
# We use wget instead of curl here
# Sample usage:
#   VERSION_STRINGS=$(get_latest_release IntelRealSense/librealsense)

function get_latest_release () {
  # redirect wget to standard out and grep out the tag_name
  wget -qO- https://api.github.com/repos/$1/releases/latest |
    grep -Po '"tag_name": "\K.*?(?=")' 
}

if [[ $LIBREALSENSE_VERSION == "" ]] ; then
   echo "Getting latest librealsense version number"
   LIBREALSENSE_VERSION=$(get_latest_release IntelRealSense/librealsense)
fi

echo "Build with CUDA: "$USE_CUDA
echo "Librealsense Version: $LIBREALSENSE_VERSION"

if [ ! -d "$LIBREALSENSE_DIRECTORY" ] ; then
  # clone librealsense
  cd ${HOME}
  echo "Cloning librealsense"
  git clone https://github.com/IntelRealSense/librealsense.git
fi

# Is the version of librealsense current enough?
cd $LIBREALSENSE_DIRECTORY
VERSION_TAG=$(git tag -l $LIBREALSENSE_VERSION)
if [ ! $VERSION_TAG  ] ; then
  echo ""
  echo "==== librealsense Version Mismatch! ============="
  echo ""
  echo "The installed version of librealsense is not current enough for these scripts."
  echo "This script needs librealsense tag version: "$LIBREALSENSE_VERSION "but it is not available."
  echo "Please upgrade librealsense or remove the librealsense folder before attempting to install again."
  echo ""
  exit 1
fi

# Checkout version the last tested version of librealsense
git checkout $LIBREALSENSE_VERSION

# Install the dependencies
cd $INSTALL_DIR

cd $LIBREALSENSE_DIRECTORY
git checkout $LIBREALSENSE_VERSION

# Now compile librealsense and install
mkdir build 
cd build
# Build examples, including graphical ones
echo "Configuring Make system"
# Build with CUDA (default), the CUDA flag is USE_CUDA, ie -DUSE_CUDA=true
export CUDACXX=$NVCC_PATH
export PATH=${PATH}:/usr/local/cuda/bin
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda/lib64

cmake ../ -DBUILD_EXAMPLES=true -DFORCE_RSUSB_BACKEND=true -DBUILD_WITH_CUDA="$USE_CUDA" -DCMAKE_BUILD_TYPE=release -DBUILD_PYTHON_BINDINGS=bool:true

# The library will be installed in /usr/local/lib, header files in /usr/local/include
# The demos, tutorials and tests will located in /usr/local/bin.
echo "Building librealsense, headers, tools and demos"

# If user didn't set # of jobs and we have > 4GB memory then
# set # of jobs to # of cores-1, otherwise 1
if [[ $NUM_PROCS == "" ]] ; then
  TOTAL_MEMORY=$(free | awk '/Mem\:/ { print $2 }')
  if [ $TOTAL_MEMORY -gt 4051048 ] ; then
    NUM_CPU=$(nproc)
    NUM_PROCS=$(($NUM_CPU - 1))
  else
    NUM_PROCS=1
  fi
fi

time make -j$NUM_PROCS
if [ $? -eq 0 ] ; then
  echo "librealsense make successful"
else
  # Try to make again; Sometimes there are issues with the build
  # because of lack of resources or concurrency issues
  echo "librealsense did not build " >&2
  echo "Retrying ... "
  # Single thread this time
  time make 
  if [ $? -eq 0 ] ; then
    echo "librealsense make successful"
  else
    # Try to make again
    echo "librealsense did not successfully build" >&2
    echo "Please fix issues and retry build"
    exit 1
  fi
fi
echo "Installing librealsense, headers, tools and demos"
sudo make install
  
if  grep -Fxq 'export PYTHONPATH=$PYTHONPATH:/usr/local/lib' ~/.bashrc ; then
    echo "PYTHONPATH already exists in .bashrc file"
else
   echo 'export PYTHONPATH=$PYTHONPATH:/usr/local/lib' >> ~/.bashrc 
   echo "PYTHONPATH added to ~/.bashrc. Pyhon wrapper is now available for importing pyrealsense2"
fi

cd $LIBREALSENSE_DIRECTORY
echo "Applying udev rules"
# Copy over the udev rules so that camera can be run from user space
sudo cp config/99-realsense-libusb.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && udevadm trigger

echo "Library Installed"
echo " "
echo " -----------------------------------------"
echo "The library is installed in /usr/local/lib"
echo "The header files are in /usr/local/include"
echo "The demos and tools are located in /usr/local/bin"
echo " "
echo " -----------------------------------------"
echo " "
