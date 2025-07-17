#!/usr/bin/env python3
#
# Find a container image that's compatible with the requested package(s),
# either locally, built from source, or pulled from DockerHub.
#
# For example, you can use this to automatically run a package without tracking
# down yourself exactly which container image/tag to run:
#
#   $ sudo docker run --runtime nvidia -it --rm $(./autotag pytorch)
#   $ ./run.sh $(./autotag pytorch)   # shorthand for full 'docker run' command
#
# Or interspersed with more run arguments:
#
#   $ ./run.sh --volume /my/dir:/mount $(./autotag tensorflow2)
#   $ ./run.sh --volume /my/dir:/mount $(./autotag tensorflow2) /bin/bash -c 'some cmd'
#
# By default, the most-recent local image will be preferred - then DockerHub will be checked.
# If a compatible image isn't found on DockerHub, the user will be asked if they want to build it.
#
import argparse
import os
import pprint
import re
import sys

from jetson_containers import (find_package, find_packages, find_container, L4T_VERSION,
                               JETPACK_VERSION, CUDA_VERSION)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('packages', type=str, nargs='*', default=[], help="package(s) to run (if multiple, a container with all)")

    parser.add_argument('-p', '--prefer', type=str, default='local,registry,build', help="comma/colon-separated list of the source preferences (default: 'local,registry,build')")
    parser.add_argument('-d', '--disable', type=str, default='', help="comma/colon-separated list of sources to disable (local,registry,build)")
    parser.add_argument('-u', '--user', type=str, default='dustynv', help="the DockerHub user for registry container images")
    parser.add_argument('-o', '--output', type=str, default='/tmp/autotag', help="file to save the selected container tag to")
    parser.add_argument('-q', '--quiet', action='store_true', help="use the default unattended options instead of prompting the user")
    parser.add_argument('-v', '--verbose', action='store_true', help="log extra info like the registry repository manifests")

    args = parser.parse_args()

    args.prefer = re.split(',|;|:', args.prefer)
    args.disable = re.split(',|;|:', args.disable)

    if args.verbose:
        os.environ['VERBOSE'] = 'ON'

    print(args)
    print(f"-- L4T_VERSION={L4T_VERSION}  JETPACK_VERSION={JETPACK_VERSION}  CUDA_VERSION={CUDA_VERSION}")

    if len(args.packages) == 0:
        print(f"-- Error:  no packages were specified")
        sys.exit(127)

    print(f"-- Finding compatible container image for {args.packages}")

    image = find_container(args.packages[0], prefer_sources=args.prefer, disable_sources=args.disable, user=args.user, quiet=args.quiet)

    if not image:
        print(f"-- Error:  couldn't find a compatible container image for '{args.packages[0]}'")
        sys.exit(127)

    if args.output:
        with open(args.output, 'w') as file:
            file.write(image)

    print(image)
