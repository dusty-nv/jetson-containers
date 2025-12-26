#!/usr/bin/env python3
# finds the versions of JetPack-L4T and CUDA from the build environment:
#
#    L4T_VERSION (packaging.version.Version) -- found in /etc/nv_tegra_release
#    JETPACK_VERSION (packaging.version.Version) -- derived from L4T_VERSION
#    PYTHON_VERSION (packaging.version.Version) -- the default for LSB_RELEASE (can override with $PYTHON_VERSION environment var)
#    PYTHON_FREE_THREADING (bool) -- True if the selected Python version is a "nogil" build
#    CUDA_VERSION (packaging.version.Version) -- found in /usr/local/cuda (can override with $CUDA_VERSION environment var)
#    CUDA_ARCHITECTURES (list[int]) -- e.g. [53, 62, 72, 87, 101]
#    SYSTEM_ARCH (str) -- e.g. 'aarch64' or 'x86_64'
#    LSB_RELEASE (str) -- e.g. '18.04', '20.04', '22.04'
#
import datetime
import glob
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from packaging.version import Version


def get_l4t_version(version_file='/etc/nv_tegra_release', l4t_version: str = None):
    """
    Returns the L4T_VERSION in a packaging.version.Version object
    Which can be compared against other version objects:  https://packaging.pypa.io/en/latest/version.html
    You can also access the version components directly.  For example, on L4T R35.3.1:
    You can also access the version components directly.  For example, on L4T R35.3.1:

        version.major == 35
        version.minor == 3
        version.micro == 1

    The L4T_VERSION will either be parsed from /etc/nv_tegra_release or the $L4T_VERSION environment variable.
    """
    if l4t_version:
        return Version(l4t_version) if not isinstance(l4t_version,
                                                      Version) else l4t_version

    if 'L4T_VERSION' in os.environ and len(os.environ['L4T_VERSION']) > 0:
        return Version(os.environ['L4T_VERSION'].lower().lstrip('r'))

    if CUDA_ARCH != 'tegra-aarch64':
        return Version('38.3.0')  # for x86 to unlock L4T checks

    if not os.path.isfile(version_file):
        # raise IOError(f"L4T_VERSION file doesn't exist:  {version_file}")
        return Version('38.3.0')

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

    if not l4t_release.startswith(l4t_release_prefix) or not l4t_release.endswith(
        l4t_release_suffix):
        raise ValueError(
            f"L4T release string is invalid or in unexpected format:  '{l4t_release}'")

    l4t_release = l4t_release[len(l4t_release_prefix):-len(l4t_release_suffix)]

    # parse the revision
    l4t_revision = parts[1]
    l4t_revision_prefix = 'REVISION: '

    if not l4t_revision.startswith(l4t_revision_prefix):
        raise ValueError(
            f"L4T revision '{l4t_revision}' doesn't start with expected prefix '{l4t_revision_prefix}'")

    l4t_revision = l4t_revision[len(l4t_revision_prefix):]

    # return packaging.version object
    return Version(f'{l4t_release}.{l4t_revision}')


def nv_tegra_release(version_file='/etc/nv_tegra_release', dst=None):
    """
    Return the contents of the `/etc/nv_tegra_release` file, optionally
    saving a copy to the destination path, and generating a default if not found.
    """
    if os.path.exists(version_file):
        if dst:
            shutil.copyfile(version_file, dst)
        with open(version_file) as file:
            return file.read()

    text = ''.join([
        f"# R{L4T_VERSION.major} (release), ",
        f"REVISION: {L4T_VERSION.minor}.{L4T_VERSION.micro}, ",
        f"GCID: 12345678, BOARD: generic, ",
        f"EABI: {SYSTEM_ARCH}, ",
        f"DATE: {datetime.datetime.now().strftime('%a %b %d %H:%M:%S %Z %Y')}\n",
        f"# KERNEL_VARIANT: oot\n",
        f"TARGET_USERSPACE_LIB_DIR=nvidia\n",
        f"TARGET_USERSPACE_LIB_DIR_PATH=/usr/lib/{SYSTEM_ARCH}-linux-gnu/nvidia"
    ])

    if dst:
        with open(dst, 'w') as file:
            file.write(text)

    return text


def get_jetpack_version(l4t_version: str = None, default='6.2'):
    """
    Returns the version of JetPack (based on the L4T version)
    https://github.com/rbonghi/jetson_stats/blob/master/jtop/core/jetson_variables.py

    JETPACK_VERSION will be determined based on L4T_VERSION or overridden by the $JETPACK_VERSION environment variable.
    """
    if not l4t_version:
        l4t_version = get_l4t_version()

    if not isinstance(l4t_version, Version):
        l4t_version = Version(l4t_version)

    if 'JETPACK_VERSION' in os.environ and len(os.environ['JETPACK_VERSION']) > 0:
        return Version(os.environ['JETPACK_VERSION'].lower().lstrip('r'))

    NVIDIA_JETPACK = {
        # -------- JP7 --------
        "38.3.0": "7.1", # Q4 2025 T400 Support
        "38.2.2": "7.0 GA",
        "38.2.0": "7.0 GA",
        "38.1.0": "7.0 EA",

        # -------- JP6 --------
        "36.4.7": "6.2.1",
        "36.4.4": "6.2.1",
        "36.4.3": "6.2",
        "36.4.2": "6.1.1",
        "36.4.0": "6.1 GA",
        "36.3.0": "6.0 GA",
        "36.2.0": "6.0 DP",
        "36.0.0": "6.0 EA",

        # -------- JP5 --------
        "35.6.2": "5.1.5",
        "35.6.1": "5.1.5",
        "35.6.0": "5.1.4",
        "35.5.0": "5.1.3",
        "35.4.1": "5.1.2",
        "35.3.1": "5.1.1",
        "35.3.0": "5.1.1 PRE",
        "35.2.1": "5.1",
        "35.1.0": "5.0.2 GA",
        "34.1.1": "5.0.1 DP",
        "34.1.0": "5.0 DP",
        "34.0.1": "5.0 PRE-DP",
        # -------- JP4 --------
        "32.7.6": "4.6.6",
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


def get_cuda_version(version_file: str = "/usr/local/cuda/version.json",
                     l4t_version: str = None):
    """
    Returns the installed version of the CUDA Toolkit in a packaging.version.Version object
    The CUDA_VERSION will either be parsed from /usr/local/cuda/version.json or the $CUDA_VERSION environment variable.
    """

    def to_version(version):
        version = Version(version)
        return Version(f"{version.major}.{version.minor}")

    if 'CUDA_VERSION' in os.environ and len(os.environ['CUDA_VERSION']) > 0:
        return to_version(os.environ['CUDA_VERSION'])

    if LSB_RELEASE == '24.04' and L4T_VERSION.major >= 38:
        return Version('13.0')  # default to CUDA 13.0 for 24.04 containers on JP7

    if LSB_RELEASE == '24.04' and L4T_VERSION.major <= 36:
        return Version('12.9')  # default to CUDA 12.9 for 24.04 containers on JP6

    if l4t_version or not os.path.isfile(version_file):
        # In case only the CUDA runtime is installed
        so_file_path = "/usr/local/cuda/targets/aarch64-linux/lib/libcudart.so.*.*.*"
        files = glob.glob(so_file_path)
        if files and not l4t_version:
            file_path = files[0]  # Assuming there is only one matching file
            version_match = re.search(r'libcudart\.so\.(\d+\.\d+\.\d+)', file_path)

            if version_match:
                version_number = version_match.group(1)
                return to_version(version_number)
            else:
                print("-- unable to extract CUDA version number")
        else:
            l4t_version = get_l4t_version(l4t_version=l4t_version)
            if l4t_version.major >= 38:
                cuda_version = '13.0'
            elif l4t_version.major >= 36:
                # L4T r36.x (JP 6.x) and above does not require having CUDA installed on host
                # When CUDA is not installed on host, users can specify which version of
                # CUDA (and matching version cuDNN and TensorRT) in container by
                # executing, for example, `export CUDA_VERSION=12.9`.
                # If the env variable is not set, set the CUDA_VERSION to be the CUDA version
                # that made available with the release of L4T_VERSION
                if l4t_version == Version('36.4') or l4t_version == Version(
                    '36.4.2') or l4t_version == Version(
                    '36.4.3') or l4t_version == Version('36.4.4'):
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
                cuda_version = '0.0'  # Note, this get_cuda_version() function used to return '0.0' as str.
            return Version(cuda_version)

    with open(version_file) as file:
        versions = json.load(file)

    return to_version(versions['cuda_nvcc']['version'])


def cuda_short_version(cuda_version: str = None):
    """
    Return the abbreviated CUDA version string (like 'cu124' from '12.4')
    """
    if not cuda_version:
        cuda_version = CUDA_VERSION

    if isinstance(cuda_version, str):
        cuda_version = Version(cuda_version)

    return f"cu{cuda_version.major}{cuda_version.minor}"


def get_cuda_arch(l4t_version: str = None, cuda_version: str = None, format=list):
    """
    Return the default list of CUDA/NVCC device architectures for the given L4T_VERSION.
    """
    if not l4t_version:
        l4t_version = get_l4t_version()

    if not isinstance(l4t_version, Version):
        l4t_version = Version(l4t_version)

    if not cuda_version:
        cuda_version = get_cuda_version(l4t_version=l4t_version)

    if not isinstance(cuda_version, Version):
        cuda_version = Version(cuda_version)

    # supported_arches = ['3.5', '3.7', '5.0', '5.2', '5.3', '6.0', '6.1', '6.2',
    #                    '7.0', '7.2', '7.5', '8.0', '8.6', '8.7', '8.9', '9.0', '9.0a',
    #                    '10.0', '10.0a', '10.1', '10.1a', '12.0', '12.0a']
    if SYSTEM_ARM:
        # Nano/TX1 = 5.3, TX2 = 6.2, Xavier = 7.2, Orin = 8.7, Thor = 11.0
        if IS_TEGRA:
            if l4t_version.major >= 38:  # JetPack 7
                cuda_architectures = [87, 110, 120, 121]  # Thor 110, Spark
            elif l4t_version.major >= 36:  # JetPack 6
                cuda_architectures = [87]  # Ampere Orin, Hopper GH200 90
            elif l4t_version.major >= 34:  # JetPack 5
                cuda_architectures = [72, 87]
            elif l4t_version.major == 32:  # JetPack 4
                cuda_architectures = [53, 62, 72]
        elif IS_SBSA:
            cuda_architectures = [90, 100, 103, 110, 120, 121]  # Orin, Hopper, Blackwell, Thor 110, RTX/Spark
    else:
        cuda_architectures = [80, 90, 100, 120 ]

        if cuda_version >= Version('13.0'):
            cuda_architectures += [103, 110, 121]

    if format == list:
        return cuda_architectures
    elif format == str:
        return ';'.join([f'{x / 10:.1f}' for x in cuda_architectures])
    else:
        raise ValueError(f"get_cuda_arch() expected format=list or str (was {format})")


def get_l4t_base(l4t_version: str = None):
    """
    Returns the l4t-base or l4t-jetpack container to use
    """
    if not l4t_version:
        l4t_version = get_l4t_version()

    if l4t_version.major >= 38:  # JetPack 7
        return f"ubuntu:{LSB_RELEASE}"
    elif l4t_version.major >= 36:  # JetPack 6
        return f"ubuntu:{LSB_RELEASE}"
    elif l4t_version.major >= 34:  # JetPack 5
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


def l4t_version_compatible(l4t_version, l4t_version_host=None, **kwargs):
    """
    Returns true if the host OS can run containers built for the provided L4T version.
    """
    if not l4t_version:
        return False

    if not l4t_version_host:
        l4t_version_host = get_l4t_version()

    if not isinstance(l4t_version, Version):
        l4t_version = Version(l4t_version)

    if l4t_version_host.major == 38:  # JetPack 7 runs containers for JetPack 7
        return True
    elif l4t_version_host.major == 36:  # JetPack 6 runs containers for JetPack 6
        if l4t_version.major == 36:
            if l4t_version.minor < 4 and l4t_version_host.minor < 4:
                return True
            elif l4t_version.minor >= 4 and l4t_version_host.minor >= 4:
                return True
    elif l4t_version_host.major == 35:  # JetPack 5.1 can run other JetPack 5.1.x containers
        if l4t_version.major == 35:
            return True
    elif l4t_version_host.major == 34:  # JetPack 5.0 runs other JetPack 5.0.x containers
        if l4t_version.major == 34:
            return True
    elif l4t_version_host >= Version(
        '32.7'):  # JetPack 4.6.1+ runs other JetPack 4.6.x containers
        if l4t_version >= Version('32.7'):
            return True

    return l4t_version == l4t_version_host


def get_lsb_release(l4t_version: str = None):
    """
    Returns the default Ubuntu version to build (e.g. 24.04)
    First this uses the LSB_RELEASE environment variable if set.
    Otherwise, on aarch64 it gets taken from the running host OS.
    On x86, it always gets set to 24.04 right now for consistency.
    """
    if l4t_version:
        l4t_version = get_l4t_version(l4t_version=l4t_version)
        if l4t_version.major >= 38:
            return '24.04'
        elif l4t_version.major == 36:
            return '22.04'
        elif l4t_version.major >= 34:
            return '20.04'
        elif l4t_version.major == 32:
            return '18.04'
        else:
            return

    def lsb(type):
        return subprocess.run(["lsb_release", f"-{type}s"], stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE,
                              universal_newlines=True, check=True).stdout.strip()

    if 'LSB_RELEASE' in os.environ and len(os.environ['LSB_RELEASE']) > 0:
        return os.environ['LSB_RELEASE']

    return '24.04' if SYSTEM_X86 or IS_SBSA else lsb('r')

def _parse_python_ver_and_nogil(s) -> tuple[Version, bool]:
    """
    Accepts '3.13', '3.14', '3.13t', '3.14t', '3.13-nogil', Version('3.13'), etc.
    Returns (Version, is_nogil)
    """
    if isinstance(s, Version):
        return s, False
    raw = str(s).strip().lower()
    is_nogil = raw.endswith('t') or raw.endswith('-nogil')
    if raw.endswith('t'):
        raw = raw[:-1]
    elif raw.endswith('-nogil'):
        raw = raw[:-6]
    return Version(raw), is_nogil

def get_python_version(lsb_release: str = None):
    """
    Gets the default version of Python to use (e.g. 3.13)
    First this uses the PYTHON_VERSION environment variable if set.
    Supports 't'/'-nogil' suffix to enable free-threading builds.
    Otherwise, it checks if LSB_RELEASE is in the DEFAULT_PYTHON_VERSIONS table.
    Finally, it falls back to the version of Python running this script from the host.

    The 't' suffix is stripped and PYTHON_FREE_THREADING is set instead.
    """
    global PYTHON_FREE_THREADING

    if lsb_release:
        ver, is_nogil = _parse_python_ver_and_nogil(DEFAULT_PYTHON_VERSIONS[lsb_release])
        PYTHON_FREE_THREADING = is_nogil
        return ver

    env = os.environ.get('PYTHON_VERSION', None)

    if env and len(env) > 0:
        ver, is_nogil = _parse_python_ver_and_nogil(env)
        PYTHON_FREE_THREADING = is_nogil or PYTHON_FREE_THREADING
        return ver
    elif LSB_RELEASE in DEFAULT_PYTHON_VERSIONS:
        ver, is_nogil = _parse_python_ver_and_nogil(DEFAULT_PYTHON_VERSIONS[LSB_RELEASE])
        PYTHON_FREE_THREADING = is_nogil or PYTHON_FREE_THREADING
        return ver
    else:
        ver = Version(f'{sys.version_info.major}.{sys.version_info.minor}')
        # no suffix implies normal build
        return ver


def check_arch(arch: str, system_arch: str = None):
    """
    Returns true if matching SYSTEM_ARCH or DOCKER_ARCH
    """
    if system_arch:
        return arch == system_arch
    else:
        return arch == SYSTEM_ARCH or arch == DOCKER_ARCH


# ubuntu info
LSB_RELEASES = {
    '16.04': 'xenial',
    '18.04': 'bionic',
    '20.04': 'focal',
    '22.04': 'jammy',
    '24.04': 'noble',
    '26.04': 'resolute',
}

DEFAULT_PYTHON_VERSIONS = {
    '18.04': Version('3.6'),
    '20.04': Version('3.8'),
    '22.04': Version('3.10'),
    '24.04': Version('3.12'),
    '26.04': Version('3.14'),
}

CUDA_ARCHS = {
    'tegra-aarch64': 'arm64',
    'aarch64': 'arm64',
    'x86_64': 'amd64'
}

OS_ARCH_DICT = {
    "amd64": "x86_64-unknown-linux-gnu",
    "aarch64": "aarch64-unknown-linux-gnu",
    "tegra-aarch64": "aarch64-unknown-linux-gnu",
}

_REDIST_ARCH_DICT = {
    "linux-x86_64": "x86_64-unknown-linux-gnu",
    "linux-sbsa": "aarch64-unknown-linux-gnu",
    "linux-aarch64": "aarch64-unknown-linux-gnu",
}

TEGRA = "tegra"


def _get_platform_architecture():
    host_arch = platform.machine()

    if host_arch == "aarch64":
        try:
            # Use a longer timeout to handle slower nvidia-smi responses on some Thor units
            # If nvidia-smi takes longer than 10 seconds, it indicates a runner issue
            gpu_names = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                encoding="utf-8",
                timeout=10
            )
            if "nvgpu" not in gpu_names:
                return os.environ.get('CUDA_ARCH', host_arch)
            else:
                return os.environ.get('CUDA_ARCH', f"{TEGRA}-{host_arch}")
        except subprocess.TimeoutExpired:
            # nvidia-smi took too long - this indicates a runner issue
            raise RuntimeError(
                "nvidia-smi command timed out after 10 seconds. "
                "If this happens on your runner, please consider disabling this runner."
            )
        except Exception as e:
            # Fall back to tegra-aarch64 on other errors (driver not found, etc.)
            return os.environ.get('CUDA_ARCH', f"{TEGRA}-{host_arch}")
    return os.environ.get('CUDA_ARCH', host_arch)


# cpu architecture
CUDA_ARCH = os.environ.get("CUDA_ARCH", _get_platform_architecture())
SYSTEM_ARCH = os.environ.get('SYSTEM_ARCH', platform.machine())  # UNIFIED tegra and sbsa as aarch64
DOCKER_ARCH = CUDA_ARCHS[SYSTEM_ARCH]

SYSTEM_ARM = CUDA_ARCH in ("aarch64", "tegra-aarch64")
SYSTEM_X86 = CUDA_ARCH == "x86_64"
IS_TEGRA = CUDA_ARCH == "tegra-aarch64"
IS_SBSA = CUDA_ARCH == "aarch64"

SYSTEM_ARCH_LIST = []

for arch in CUDA_ARCHS.items():
    SYSTEM_ARCH_LIST.extend(arch)

# os/jetpack/cuda versions
PYTHON_FREE_THREADING = os.environ.get('PYTHON_FREE_THREADING', '0') == '1'
LSB_RELEASE = get_lsb_release()
L4T_VERSION = get_l4t_version()
JETPACK_VERSION = get_jetpack_version()
PYTHON_VERSION = get_python_version()  # Version object (PEP 440 compliant, 't' stripped)
CUDA_VERSION = get_cuda_version()
CUDA_SHORT_VERSION = cuda_short_version()
CUDA_ARCHITECTURES = get_cuda_arch()
