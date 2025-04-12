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
import traceback
import subprocess
from tabulate import tabulate

from jetson_containers import (
    build_container, build_containers, find_packages, package_search_dirs, 
    cprint, to_bool, log_config, log_error, log_status, log_versions, LogConfig
)

# Function to check if a package is installed and install it if it is not
def check_and_install(package):
    try:
        __import__(package)
    except ImportError:
        print(f"Package {package} not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Check and install required packages
required_packages = ["tabulate"]
for package in required_packages:
    check_and_install(package)

# Function to display packages in a table format
def display_packages(packages):
    headers = ["Package Name", "Version"]
    table = []
    for pkg, details in packages.items():
        version = details.get('version', 'N/A')
        if isinstance(version, dict):
            version = version.get('number', 'N/A')
        table.append((pkg, version))
    print(tabulate(table, headers, tablefmt="fancy_grid"))

parser = argparse.ArgumentParser(description="Jetson Containers Utility")
parser.add_argument('packages', type=str, nargs='*', default=[], help='Packages or containers to build (filterable by wildcards)')
parser.add_argument('--name', type=str, default='', help="The name of the output container to build")
parser.add_argument('--base', type=str, default='', help="The base container to use at the beginning of the build chain (default: l4t-jetpack)")
parser.add_argument('--multiple', action='store_true', help="The specified packages should be built independently as opposed to chained together")
parser.add_argument('--build-flags', type=str, default='', help="Extra flags to pass to 'docker build' commands")
parser.add_argument('--build-args', type=str, default='', help="Container build arguments (--build-arg) as a string of comma-separated key:value pairs")
parser.add_argument('--use-proxy', action='store_true', help="Use the host's proxy envvars for the container build")
parser.add_argument('--package-dirs', type=str, default='', help="Additional package search directories (comma or colon-separated)")
parser.add_argument('--list-packages', action='store_true', help="Show the list of packages that were found under the search directories")
parser.add_argument('--show-packages', action='store_true', help="Show info about one or more packages (if none are specified, all will be listed)")
parser.add_argument('--skip-packages', type=str, default='', help="Disable certain packages/containers (filterable by wildcards, comma/colon-separated)")
parser.add_argument('--skip-errors', action='store_true', help="Continue building when errors occur (only relevant when --multiple is in use)")
parser.add_argument('--skip-tests', type=str, default='', help="Comma-separated list of package tests to disable ('intermediate' to disable build-stage tests, 'all' to disable all)")
parser.add_argument('--test-only', type=str, default='', help="Only test the specified packages (comma/colon-separated list)")
parser.add_argument('--simulate', action='store_true', help="Print out the build commands without actually building the containers")
parser.add_argument('--push', type=str, default='', help="Repo or user to push built container image to (no push by default)")
parser.add_argument('--no-github-api', action='store_true', help="Disable GitHub API use to force rebuild on new git commits")
parser.add_argument('--log-dir', '--logs', type=str, default=None, help="Sets the directory to save container build logs to (default: jetson-containers/logs)")
parser.add_argument('--log-level', type=str, default=None, choices=LogConfig.levels, help="Sets the logging verbosity level")
parser.add_argument('--log-colors', type=to_bool, default=None, help="Enable/disable terminal colors and formatting (defaults to true)")
parser.add_argument('--log-status', type=to_bool, default=None, help="Enable status bar at bottom of terminal (defaults to true)")
parser.add_argument('--debug', action='store_true', help="Enable debug logging")
parser.add_argument('--verbose', action='store_true', help="Enable verbose logging")
parser.add_argument('--version', action='store_true', help="Print platform version info and exit")

# Add examples and usage instructions
parser.epilog = """
Examples:
  jetson-containers --list
  jetson-containers --show pytorch
  jetson-containers --version
"""

args = parser.parse_args()

# Configure logging
log_config(**vars(args))

# Validate args
if args.skip_errors and not args.multiple:
    raise ValueError("--skip-errors can only be used with --multiple flag")

# Split multi-value keyword arguments
args.package_dirs = re.split(',|;|:', args.package_dirs)
args.skip_packages = re.split(',|;|:', args.skip_packages)
args.skip_tests = re.split(',|;|:', args.skip_tests)
args.test_only = re.split(',|;|:', args.test_only)

print(f'\n{args}\n')
log_versions()
cprint(f"\n$ jetson-containers {' '.join(sys.argv[1:])}\n", attrs='bold')

if args.version:
    sys.exit()

# Cast build args into dictionary
if args.build_args:
    try:
        key_value_pairs = args.build_args.split(',')
        args.build_args = {pair.split(':')[0]: pair.split(':', maxsplit=1)[1] for pair in key_value_pairs}
    except(ValueError, IndexError):
        raise argparse.ArgumentTypeError("Invalid dictionary format. Use key1:value1, key2:value2 ...")
else:
    args.build_args = {}

# Add proxy to build args if flag is set
if args.use_proxy:
    proxy_vars = ['http_proxy', 'https_proxy', 'no_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'NO_PROXY']
    for var in proxy_vars:
        if var in os.environ:
            args.build_args[var] = os.environ[var]

# Add package directories
if args.package_dirs:
    package_search_dirs(args.package_dirs)

# List/show package info
if args.list_packages or args.show_packages:
    try:
        packages = find_packages(args.packages, skip=args.skip_packages)

        if args.list_packages:
            print("Available Packages:")
            for package in sorted(packages.keys()):
                print(package)
        
        if args.show_packages:
            print("Package Details:")
            display_packages(packages)
            
    except Exception as e:
        print(f"Error: {e}\nAn error occurred while listing/showing packages. Please try again.")
        
    sys.exit(0)
    
try:
    # Build one multi-stage container from chain of packages
    # or launch multiple independent container builds
    if not args.multiple:
        build_container(**vars(args))
    else:   
        build_containers(**vars(args))
except Exception as error:
    log_error(f"Failed building:  {', '.join(args.packages)}\n\n{traceback.format_exc()}")
finally:
    log_status(done=True)
