#!/usr/bin/env python3
import os
import platform

from packaging import version



def get_l4t_version(version_file='/etc/nv_tegra_release'):
    """
    Returns the L4T_VERSION in a packaging.version.Version object
    Which can be compared against other version objects:  https://packaging.pypa.io/en/latest/version.html
    You can also access the version components directly.  For example, on L4T R35.3.1:
    
        version.major == 35
        version.minor == 3
        version.micro == 1
    """
    if platform.machine() != 'aarch64':
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

# x86_64, aarch64
ARCH = platform.machine()

# Nano/TX1 = 5.3
# TX2 = 6.2
# Xavier = 7.2
# Orin = 8.7
if L4T_VERSION.major >= 34:    # JetPack 5
    CUDA_ARCH_LIST_INT = [72, 87]
    
elif L4T_VERSION.major == 32:  # JetPack 4
    CUDA_ARCH_LIST_INT = [53, 62, 72]
    
CUDA_ARCH_LIST_FLOAT = [cc/10.0 for cc in CUDA_ARCH_LIST_INT]
CUDA_ARCH_LIST = [f'{cc:.1f}' for cc in CUDA_ARCH_LIST_FLOAT]
