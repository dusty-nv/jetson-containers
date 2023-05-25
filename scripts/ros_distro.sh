#!/usr/bin/env bash
# this will populate $SUPPORTED_ROS_DISTROS for the version of JetPack-L4T
# and assumes that l4t_version.sh has already been sourced

if [[ $L4T_RELEASE -ge 34 ]]; then   # JetPack 5.x / Ubuntu 20.04
	SUPPORTED_ROS_DISTROS=("noetic" "foxy" "galactic" "humble" "iron")
else
	SUPPORTED_ROS_DISTROS=("melodic" "noetic" "eloquent" "foxy" "galactic" "humble" "iron")
fi
