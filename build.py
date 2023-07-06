#!/usr/bin/env python3
import os
import argparse
import pprint
import platform
import subprocess

from packaging import version


def get_arch():
    return platform.machine()
    
ARCH = get_arch()
print(f"ARCH={ARCH}")


def get_l4t_version(version_file='/etc/nv_tegra_release'):
    """
    Returns the L4T_VERSION in a packaging.version.Version object
    Which can be compared against other version objects:  https://packaging.pypa.io/en/latest/version.html
    You can also access the version components directly.  For example, on L4T R35.3.1:
    
        version.major == 35
        version.minor == 3
        version.micro == 1
    """
    if ARCH != 'aarch64':
        raise ValueError(f"L4T_VERSION isn't supported on {ARCH} architecture (aarch64 only)")
        
    if not os.path.isfile(version_file):
        raise IOError(f"L4T_VERSION file doesn't exist:  {version_file}")
        
    with open(version_file) as file:
        line = file.readline()
        
    # R32 (release), REVISION: 7.1, GCID: 29689809, BOARD: t186ref, EABI: aarch64, DATE: Wed Feb  2 21:33:23 UTC 2022
    # R34 (release), REVISION: 1.1, GCID: 30414990, BOARD: t186ref, EABI: aarch64, DATE: Tue May 17 04:20:55 UTC 2022
    # R35 (release), REVISION: 2.1, GCID: 32398013, BOARD: t186ref, EABI: aarch64, DATE: Sun Jan 22 03:18:23 UTC 2023
    # R35 (release), REVISION: 3.1, GCID: 32790763, BOARD: t186ref, EABI: aarch64, DATE: Wed Mar 15 07:54:12 UTC 2023
    parts = [part.strip() for part in line.split(',')]

    # parse the release
    l4t_release = parts[0]
    l4t_release_prefix = '# R'
    l4t_release_suffix = ' (release)'
    
    if not l4t_release.startswith(l4t_release_prefix) or not l4t_release.endswith(l4t_release_suffix):
        raise ValueError(f"L4T release string is invalid or in unexpected format:  '{l4t_release}'")
        
    l4t_release = l4t_release[len(l4t_release_prefix):-len(l4t_release_suffix)]

    # parse the revision
    l4t_revision = parts[1]
    l4t_revision_prefix = 'REVISION: '
    
    if not l4t_revision.startswith(l4t_revision_prefix):
        raise ValueError(f"L4T revision '{l4t_revision}' doesn't start with expected prefix '{l4t_revision_prefix}'")
       
    l4t_revision = l4t_revision[len(l4t_revision_prefix):]
    
    # return packaging.version object
    return version.parse(f'{l4t_release}.{l4t_revision}')
    
    
L4T_VERSION = get_l4t_version()
print(f"L4T_VERSION={L4T_VERSION}")


def get_l4t_base():
    """
    Returns the l4t-base or l4t-jetpack container to use
    """
    if L4T_VERSION.major >= 5:
        return f"nvcr.io/nvidia/l4t-jetpack:r{L4T_VERSION}"
    else:
        return f"nvcr.io/nvidia/l4t-base:r{L4T_VERSION}"
        
        
# parse command-line arguments
parser = argparse.ArgumentParser()
                    
parser.add_argument('packages', type=str, nargs='*', default=[], help='packages or configs to build')
parser.add_argument('--package-dirs', type=str, nargs='+', default=[], help='package search directories')
parser.add_argument('--name', type=str, default='', help='the name of the output container to build')
parser.add_argument('--base', type=str, default=get_l4t_base(), help='the base container image to use at the beginning of the build chain')

args = parser.parse_args()

args.package_dirs.append('packages')   # TODO make this absolute based on location of this script

if not args.name and len(args.packages) > 0:   # assign a default name based on the final package
    args.name = f"{args.packages[-1]}:r{L4T_VERSION}"
    
if args.name.find(':') < 0:   # assign a default tag based on the L4T version
    args.name += f":r{L4T_VERSION}"
    
print(args)

if len(args.packages) == 0:
    raise ValueError(f"no packages or container configs were selected to build")
    

# find available packages
def find_packages(path):
    """
    Recursively find packages in and under the provided path
    This looks for Dockerfiles in the directories
    Returns a dict of package info from this path and sub-paths
    """
    package = {'path': path}
    packages = {}  # sub-packages
    
    print(f"searching '{path}' for packages...")
    
    if not os.path.isdir(path):
        print(f"warning -- package dir '{path}' doesn't exist, skipping...")
        return packages
        
    entries = os.listdir(path)
    print(entries)
    
    for entry in entries:
        print(entry)
        
        entry_path = os.path.join(path, entry)
        
        if os.path.isdir(entry_path):
            packages.update(find_packages(entry_path))
        elif os.path.isfile(entry_path):
            if entry.lower().find('dockerfile') >= 0:
                package['dockerfile'] = entry
                
    if 'dockerfile' in package: #len(package) > 0:
        packages[os.path.basename(path)] = package

    return packages
    
    
packages_found = {}

for package_dir in args.package_dirs:
    packages_found.update(find_packages(package_dir))
   
print('packages found:')
pprint.pprint(packages_found)
   
# build packages
base_image = args.base

for idx, pkg in enumerate(args.packages):
    if pkg not in packages_found:
        raise ValueError(f"couldn't find package:  {pkg}")
        
    package = packages_found[pkg]
    container_name = args.name
    
    if idx < len(args.packages) - 1:
        container_name = f"{args.name}-{pkg}"
        
    cmd = f"sudo docker build --network=host -t {container_name} -f {os.path.join(package['path'], package['dockerfile'])} "
    cmd += f"--build-arg BASE_IMAGE={base_image} "
    cmd += " . "
    
    print(cmd)
    
    subprocess.run(cmd, shell=True, check=True)
    
    base_container = container_name
    

