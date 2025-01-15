#!/usr/bin/env python3
# finds the versions of JetPack-L4T and CUDA from the build environment:
#
#    L4T_VERSION (packaging.version.Version) -- found in /etc/nv_tegra_release
#    JETPACK_VERSION (packaging.version.Version) -- derived from L4T_VERSION
#    PYTHON_VERSION (packaging.version.Version) -- the default for LSB_RELEASE (can override with $PYTHON_VERSION environment var)
#    CUDA_VERSION (packaging.version.Version) -- found in /usr/local/cuda (can override with $CUDA_VERSION environment var)
#    CUDA_ARCHITECTURES (list[int]) -- e.g. [53, 62, 72, 87]
#    SYSTEM_ARCH (str) -- e.g. 'aarch64' or 'x86_64'
#    LSB_RELEASE (str) -- e.g. '18.04', '20.04', '22.04'
#    LSB_CODENAME (str) -- e.g. 'bionic', 'focal', 'jammy'
#    
import os
import re
import sys
import json
import platform
import subprocess
import glob

from packaging.version import Version


def get_l4t_version(version_file='/etc/nv_tegra_release'):
    """
    Returns the L4T_VERSION in a packaging.version.Version object
    Which can be compared against other version objects:  https://packaging.pypa.io/en/latest/version.html
    You can also access the version components directly.  For example, on L4T R35.3.1:
    
        version.major == 35
        version.minor == 3
        version.micro == 1
        
    The L4T_VERSION will either be parsed from /etc/nv_tegra_release or the $L4T_VERSION environment variable.
    """
    if platform.machine() != 'aarch64':
        raise ValueError(f"L4T_VERSION isn't supported on {platform.machine()} architecture (aarch64 only)")
        
    if 'L4T_VERSION' in os.environ and len(os.environ['L4T_VERSION']) > 0:
        return Version(os.environ['L4T_VERSION'].lower().lstrip('r'))
        
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
    return Version(f'{l4t_release}.{l4t_revision}')
    
 
def get_jetpack_version(l4t_version=get_l4t_version(), default='5.1'):
    """
    Returns the version of JetPack (based on the L4T version)
    https://github.com/rbonghi/jetson_stats/blob/master/jtop/core/jetson_variables.py

    JETPACK_VERSION will be determined based on L4T_VERSION or overridden by the $JETPACK_VERSION environment variable.
    """
    
    if not isinstance(l4t_version, Version):
        l4t_version = Version(l4t_version)

    if 'JETPACK_VERSION' in os.environ and len(os.environ['JETPACK_VERSION']) > 0:
        return Version(os.environ['JETPACK_VERSION'].lower().lstrip('r'))
        
    NVIDIA_JETPACK = {
        # -------- JP6 --------
        "36.4.2": "6.1.1",
        "36.4.0": "6.1 GA",
        "36.3.0": "6.0 GA",
        "36.2.0": "6.0 DP",
        "36.0.0": "6.0 EA",
        
        # -------- JP5 --------
        "35.4.1": "5.1.2",
        "35.3.1": "5.1.1",
        "35.3.0": "5.1.1 PRE",
        "35.2.1": "5.1",
        "35.1.0": "5.0.2 GA",
        "34.1.1": "5.0.1 DP",
        "34.1.0": "5.0 DP",
        "34.0.1": "5.0 PRE-DP",
        # -------- JP4 --------
        "32.7.5": "4.6.5",
        "32.7.4": "4.6.4",
        "32.7.3": "4.6.3",
        "32.7.2": "4.6.2",
        "32.7.1": "4.6.1",
        "32.6.1": "4.6",
        "32.5.2": "4.5.1",
        "32.5.1": "4.5.1",
        "32.5.0": "4.5",
        "32.5": "4.5",
        "32.4.4": "4.4.1",
        "32.4.3": "4.4",
        "32.4.2": "4.4 DP",
        "32.3.1": "4.3",
        "32.2.3": "4.2.3",
        "32.2.1": "4.2.2",
        "32.2.0": "4.2.1",
        "32.2": "4.2.1",
        "32.1.0": "4.2",
        "32.1": "4.2",
        "31.1.0": "4.1.1",
        "31.1": "4.1.1",
        "31.0.2": "4.1",
        "31.0.1": "4.0",
        # -------- Old JP --------
        "28.4.0": "3.3.3",
        "28.2.1": "3.3 | 3.2.1",
        "28.2.0": "3.2",
        "28.2": "3.2",
        "28.1.0": "3.1",
        "28.1": "3.1",
        "27.1.0": "3.0",
        "27.1": "3.0",
        "24.2.1": "3.0 | 2.3.1",
        "24.2.0": "2.3",
        "24.2": "2.3",
        "24.1.0": "2.2.1 | 2.2",
        "24.1": "2.2.1 | 2.2",
        "23.2.0": "2.1",
        "23.2": "2.1",
        "23.1.0": "2.0",
        "23.1": "2.0",
        "21.5.0": "2.3.1 | 2.3",
        "21.5": "2.3.1 | 2.3",
        "21.4.0": "2.2 | 2.1 | 2.0 | 1.2 DP",
        "21.4": "2.2 | 2.1 | 2.0 | 1.2 DP",
        "21.3.0": "1.1 DP",
        "21.3": "1.1 DP",
        "21.2.0": "1.0 DP",
        "21.2": "1.0 DP",
    }

    for key in NVIDIA_JETPACK:
        if Version(key) == l4t_version:
            return Version(NVIDIA_JETPACK[key].split(' ')[0])
    
    if not default:
        raise RuntimeError(f"invalid/unknown L4T_VERSION {l4t_version}")
    else:
        return Version(default)
        
        
def get_cuda_version(version_file='/usr/local/cuda/version.json'):
    """
    Returns the installed version of the CUDA Toolkit in a packaging.version.Version object
    The CUDA_VERSION will either be parsed from /usr/local/cuda/version.json or the $CUDA_VERSION environment variable.
    """
    def to_version(version):
        version = Version(version)
        return Version(f"{version.major}.{version.minor}")
        
    if 'CUDA_VERSION' in os.environ and len(os.environ['CUDA_VERSION']) > 0:
        return to_version(os.environ['CUDA_VERSION'])
        
    if not os.path.isfile(version_file):
        # In case only the CUDA runtime is installed
        so_file_path = "/usr/local/cuda/targets/aarch64-linux/lib/libcudart.so.*.*.*"
        files = glob.glob(so_file_path)
        if files:
            file_path = files[0]  # Assuming there is only one matching file
            version_match = re.search(r'libcudart\.so\.(\d+\.\d+\.\d+)', file_path)

            if version_match:
                version_number = version_match.group(1)
                return to_version(version_number)
            else:
                print("-- unable to extract CUDA version number")
        else:
            l4t_version = get_l4t_version()
            if l4t_version.major >= 36:
                # L4T r36.x (JP 6.x) and above does not require having CUDA installed on host
                # When CUDA is not installed on host, users can specify which version of 
                # CUDA (and matching version cuDNN and TensorRT) in container by 
                # executing, for example, `export CUDA_VERSION=12.6`.
                # If the env variable is not set, set the CUDA_VERSION to be the CUDA version
                # that made available with the release of L4T_VERSION 
                if l4t_version == Version('36.4'):
                    cuda_version = '12.6'
                elif l4t_version == Version('36.3'):
                    cuda_version = '12.4'
                elif l4t_version == Version('36.2'):
                    cuda_version = '12.2'
                else:
                    print(f"### [Warn] Unknown L4T_VERSION: {L4T_VERSION}")
                    cuda_version = '12.2'
            else:
                # L4T r35 and below, and don't find CUDA installed on host
                cuda_version = '0.0' # Note, this get_cuda_version() function used to reutrn '0.0' as str.
            return Version(cuda_version)
        
    with open(version_file) as file:
        versions = json.load(file)
        
    return to_version(versions['cuda_nvcc']['version'])


def get_l4t_base(l4t_version=get_l4t_version()):
    """
    Returns the l4t-base or l4t-jetpack container to use
    """
    if l4t_version.major >= 36:   # JetPack 6
        return "ubuntu:22.04" #"nvcr.io/ea-linux4tegra/l4t-jetpack:r36.0.0"
    elif l4t_version.major >= 34: # JetPack 5
        if l4t_version >= Version('35.4.1'):
            return "nvcr.io/nvidia/l4t-jetpack:r35.4.1"
        else:
            return f"nvcr.io/nvidia/l4t-jetpack:r{l4t_version}"
    else:
        if l4t_version >= Version('32.7.1'):
            return "nvcr.io/nvidia/l4t-base:r32.7.1"
        else:
            return f"nvcr.io/nvidia/l4t-base:r{l4t_version}"
            
           
def l4t_version_from_tag(tag):
    """
    Extract the L4T_VERSION from a container tag by searching it for patterns like 'r35.2.1' / ect.
    Returns a packaging.version.Version object, or None if a valid L4T_VERSION couldn't be found in the tag.
    """
    tag = tag.split(':')[-1]
    tag = re.split('-|_', tag)
    
    for t in tag:
        if t[0] != 'r' and t[0] != 'R':
            continue
        try:
            return Version(t[1:])
        except:
            continue
            
    return None


def l4t_version_compatible(l4t_version, l4t_version_host=get_l4t_version(), **kwargs):
    """
    Returns true if the host OS can run containers built for the provided L4T version.
    """
    if not l4t_version:
        return False
        
    if not isinstance(l4t_version, Version):
        l4t_version = Version(l4t_version)

    if l4t_version_host.major == 36: # JetPack 6 runs containers for JetPack 6
        if l4t_version.major == 36:
            if l4t_version.minor < 4 and l4t_version_host.minor < 4:
                return True
            elif l4t_version.minor >= 4 and l4t_version_host.minor >= 4:
                return True
    elif l4t_version_host.major == 35: # JetPack 5.1 can run other JetPack 5.1.x containers
        if l4t_version.major == 35:
            return True
    elif l4t_version_host.major == 34: # JetPack 5.0 runs other JetPack 5.0.x containers
        if l4t_version.major == 34:
            return True
    elif l4t_version_host >= Version('32.7'): # JetPack 4.6.1+ runs other JetPack 4.6.x containers
        if l4t_version >= Version('32.7'):
            return True
            
    return l4t_version == l4t_version_host
    
 
def get_lsb_release():
    """
    Returns a tuple of (LSB_RELEASE, LSB_CODENAME)
       ("18.04", "bionic")
       ("20.04", "focal")
    """
    return (subprocess.run(["lsb_release", "-rs"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, check=True).stdout.strip(),
           subprocess.run(["lsb_release", "-cs"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, check=True).stdout.strip())

            
# set L4T_VERSION and CUDA_VERSION globals        
L4T_VERSION = get_l4t_version()
JETPACK_VERSION = get_jetpack_version()
CUDA_VERSION = get_cuda_version()

# Nano/TX1 = 5.3, TX2 = 6.2, Xavier = 7.2, Orin = 8.7
if L4T_VERSION.major >= 36:    # JetPack 6
    CUDA_ARCHITECTURES = [87]
elif L4T_VERSION.major >= 34:  # JetPack 5
    CUDA_ARCHITECTURES = [72, 87]
elif L4T_VERSION.major == 32:  # JetPack 4
    CUDA_ARCHITECTURES = [53, 62, 72]

# x86_64, aarch64
SYSTEM_ARCH = platform.machine()

# Python version (3.6, 3.8, 3.10, ect)
if 'PYTHON_VERSION' in os.environ and len(os.environ['PYTHON_VERSION']) > 0:
    PYTHON_VERSION = Version(os.environ['PYTHON_VERSION'])
else:
    PYTHON_VERSION = Version(f'{sys.version_info.major}.{sys.version_info.minor}')

# LSB release and codename ("20.04", "focal")
LSB_RELEASE, LSB_CODENAME = get_lsb_release()
