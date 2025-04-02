#!/usr/bin/env python3
#
# Container build system for managing package configurations and multi-stage build chains, with automated testing and dependency tracking.
#
# A "package" is composed of a Dockerfile, configs, and test scripts.  These are found under the jetson-containers/packages directory.
# There are also "meta-packages" that have no Dockerfiles themselves, but specify a set of packages to include (e.g. l4t-pytorch)
#
# Configuration metadata (such as the package's dependencies) can be inline YAML in the Dockerfile header.
# It can also be a config.py script that sets build arguments dynamically (i.e. based on the L4T version)
# Subpackages can be dynamically created in the config files (i.e. the various permutations of the ROS containers)
#
# Some example build scenarios:
#
#   $ jetson-containers/build.sh --name=xyz pytorch jupyterlab     # build container with PyTorch and JupyterLab server
#   $ jetson-containers/build.sh --multiple pytorch tensorflow     # build separate containers for PyTorch and
#   $ jetson-containers/build.sh --multiple ros:humble*            # build all ROS Humble containers (can use wildcards)
#   $ jetson-containers/build.sh ros:humble-desktop pytorch        # build ROS Humble with PyTorch on top
#   $ jetson-containers/build.sh --base=xyz:latest pytorch         # add PyTorch to an existing container
#
# Typically the jetson-containers/build.sh wrapper script is used to launch this underlying Python module. jetson-containers can also
# build external out-of-tree projects that have their own Dockerfile.  And you can add your own package search dirs for other packages.
#
import os
import re
import sys
import pprint
import argparse

from jetson_containers import (
    build_container, build_containers, find_packages, package_search_dirs, set_log_dir, cprint,
    L4T_VERSION, JETPACK_VERSION, CUDA_VERSION, PYTHON_VERSION, LSB_RELEASE, LSB_CODENAME
)


parser = argparse.ArgumentParser()

parser.add_argument('packages', type=str, nargs='*', default=[], help='packages or containers to build (filterable by wildcards)')

parser.add_argument('--name', type=str, default='', help="the name of the output container to build")
parser.add_argument('--base', type=str, default='', help="the base container to use at the beginning of the build chain (default: l4t-jetpack)")
parser.add_argument('--multiple', action='store_true', help="the specified packages should be built independently as opposed to chained together")
parser.add_argument('--build-flags', type=str, default='', help="extra flags to pass to 'docker build' commands")
parser.add_argument('--build-args', type=str, default='', help="container build arguments (--build-arg) as a string of comma separated key:value pairs")
parser.add_argument('--use-proxy', action='store_true', help="use the host's proxy envvars for the container build")
parser.add_argument('--package-dirs', type=str, default='', help="additional package search directories (comma or colon-separated)")

parser.add_argument('--list-packages', action='store_true', help="show the list of packages that were found under the search directories")
parser.add_argument('--show-packages', action='store_true', help="show info about one or more packages (if none are specified, all will be listed")
parser.add_argument('--skip-packages', type=str, default='', help="disable certain packages/containers (filterable by wildcards, comma/colon-separated)")
parser.add_argument('--skip-errors', action='store_true', help="continue building when errors occur (only relevant when --multiple is in use)")
parser.add_argument('--skip-tests', type=str, default='', help="comma-separated list of package tests to disable ('intermediate' to disable build-stage tests, 'all' to disable all)")
parser.add_argument('--test-only', type=str, default='', help="only test the specified packages (comma/colon-separated list)")

parser.add_argument('--simulate', action='store_true', help="print out the build commands without actually building the containers")
parser.add_argument('--push', type=str, default='', help="repo or user to push built container image to (no push by default)")
parser.add_argument('--logs', type=str, default='', help="sets the directory to save container build logs to (default: jetson-containers/logs)")
parser.add_argument('--verbose', action='store_true', help="enable verbose/debug logging")
parser.add_argument('--no-github-api', action='store_true', help="disalbe Github API use to force rebuild on new git commits")

parser.add_argument('--build-network', type=str, default='host', help="Docker network mode to use during build (default: host)")
parser.add_argument('--test-network',   type=str, default='host', help="Docker network mode to use during test phase (default: host)")

args = parser.parse_args()

# validate args
if args.skip_errors and not args.multiple:
    raise ValueError("--skip-errors can only be used with --multiple flag")

if args.verbose:
    os.environ['VERBOSE'] = 'ON'

# split multi-value keyword arguments
args.package_dirs = re.split(',|;|:', args.package_dirs)
args.skip_packages = re.split(',|;|:', args.skip_packages)
args.skip_tests = re.split(',|;|:', args.skip_tests)
args.test_only = re.split(',|;|:', args.test_only)

print(f'\n{args}\n')
cprint(f"-- L4T_VERSION={L4T_VERSION} JETPACK_VERSION={JETPACK_VERSION} CUDA_VERSION={CUDA_VERSION} PYTHON_VERSION={PYTHON_VERSION} LSB_RELEASE={LSB_RELEASE} ({LSB_CODENAME})", "green")
cprint(f"-- jetson-containers {' '.join(sys.argv[1:])}\n", "green")

# cast build args into dictionary
if args.build_args:
    try:
        key_value_pairs = args.build_args.split(',')
        args.build_args = {pair.split(':')[0]: pair.split(':', maxsplit=1)[1] for pair in key_value_pairs}
    except(ValueError, IndexError):
        raise argparse.ArgumentTypeError("Invalid dictionary format. Use key1:value1, key2:value2 ...")
else:
    args.build_args = {}

# add proxy to build args if flag is set
if args.use_proxy:
    proxy_vars = ['http_proxy', 'https_proxy', 'no_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'NO_PROXY']
    for var in proxy_vars:
        if var in os.environ:
            args.build_args[var] = os.environ[var]

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
if not args.multiple:
    build_container(args.name, args.packages, args.base, args.build_flags, args.build_args, args.simulate, args.skip_tests, args.test_only, args.push, args.no_github_api, args.skip_packages,
    build_network=args.build_network,
    test_network=args.test_network     )
else:
    build_containers(args.name, args.packages, args.base, args.build_flags, args.build_args, args.simulate, args.skip_errors, args.skip_packages, args.skip_tests, args.test_only, args.push,
    build_network=args.build_network,
    test_network=args.test_network     )
