#!/bin/bash
# install-realsense-dependencies.sh
# Install dependencies for  the Intel Realsense library librealsense2 on a Jetson Nano Developer Kit
# Copyright (c) 2016-19 Jetsonhacks 
# MIT License
set -e

#echo "Adding Universe repository and updating"
#apt-add-repository universe

apt-get update
echo "Adding dependencies, graphics libraries and tools${reset}"
apt-get install libssl-dev libusb-1.0-0-dev pkg-config -y

# Graphics libraries - for SDK's OpenGL-enabled examples
apt-get install libgtk-3-dev libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev -y

# QtCreator for development; not required for librealsense core library
apt-get install qtcreator -y

# Add Python 3 support
apt-get install -y python3 python3-dev

# Clean apt cache
rm -rf /var/lib/apt/lists/*
apt-get clean