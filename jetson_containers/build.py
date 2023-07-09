#!/usr/bin/env python3
#
# Container build system for managing package configurations and multi-stage build chains, with automated testing. 
#
#   $ jetson-containers/build.sh pytorch tensorflow                # build separate pytorch & tensorflow containers
#   $ jetson-containers/build.sh --multi-stage pytorch tensorflow  # build one container with both pytorch & tensorflow packages
#   $ jetson-containers/build.sh ros:humble*                       # build all ROS Humble containers (can use wildcard filters)
#   $ jetson-containers/build.sh --multi-stage ros:humble-desktop pytorch  # build ROS Humble with PyTorch on top
#
#   (typically the jetson-containers/build.sh wrapper script is used to launch the underlying Python modules)
#
# A "package" is composed of a Dockerfile, configs, and test scripts.  These are found under the jetson-containers/packages directory.
# There are also "meta-packages" under jetson-containers/config that have no Dockerfiles, but specify a set of packages (e.g. l4t-pytorch)
#
# Configuration metadata (such as the package's dependencies) can be inline YAML in the Dockerfile header.
# It can also be a config.py script that sets build arguments dynamically (i.e. based on the L4T version)
# Subpackages can be dynamically created in the config files (i.e. all the permutations of the ROS containers)
#
import os
import re
import sys
import pprint
import argparse

from jetson_containers import build_container, build_containers, find_packages, package_search_dirs, set_log_dir, L4T_VERSION


parser = argparse.ArgumentParser()
                    
parser.add_argument('packages', type=str, nargs='*', default=[], help='packages or containers to build (filterable by wildcards)')

parser.add_argument('--name', type=str, default='', help="the name of the output container to build")
parser.add_argument('--base', type=str, default='', help="the base container to use at the beginning of the build chain (default: l4t-jetpack)")
parser.add_argument('--multi-stage', action='store_true', help="launch a multi-stage container build by chaining together the packages")
parser.add_argument('--build-flags', type=str, default='', help="extra flags to pass to 'docker build'")
parser.add_argument('--package-dirs', type=str, default='', help="additional package search directories (comma or colon-separated)")
parser.add_argument('--list-packages', action='store_true', help="show the list of packages that were found under the search directories")
parser.add_argument('--show-packages', action='store_true', help="show info about one or more packages (if none are specified, all will be listed")
parser.add_argument('--skip-packages', type=str, default='', help="disable certain packages/containers (filterable by wildcards, comma/colon-separated)")
parser.add_argument('--skip-errors', action='store_true', help="continue building when errors occur (not used with --multi-stage)")
parser.add_argument('--simulate', action='store_true', help="print out the build commands without actually building the containers")
parser.add_argument('--logs', type=str, default='', help="sets the directory to save container build logs to (default: jetson-containers/logs)")

args = parser.parse_args()

# validate args
if args.multi_stage and args.skip_errors:
    raise ValueError("--skip-errors can't be used with --multi-stage")
    
# split multi-value keyword arguments
args.package_dirs = re.split(',|;|:', args.package_dirs)
args.skip_packages = re.split(',|;|:', args.skip_packages)

print(args)
print(f"-- L4T_VERSION={L4T_VERSION}")

# add package directories
if args.package_dirs:
    package_search_dirs(args.package_dirs)

# set logging directories
if args.logs:
    set_log_dir(args.logs)
    
# list/show package info
if args.list_packages or args.show_packages:
    packages = find_packages(args.packages, skip=args.skip_packages)

    if args.list_packages:
        for package in sorted(packages.keys()):
            print(package)
    
    if args.show_packages:
        pprint.pprint(packages)
        
    sys.exit(0)
    
# build one multi-stage container from chain of packages
# or launch multiple independent container builds
if args.multi_stage:
    build_container(args.name, args.packages, args.base, args.build_flags, args.simulate)
else:   
    build_containers(args.name, args.packages, args.base, args.build_flags, args.simulate, args.skip_errors, args.skip_packages)