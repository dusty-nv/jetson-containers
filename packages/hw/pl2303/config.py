import os
import subprocess

from jetson_containers import L4T_VERSION, IS_TEGRA
from packaging.version import Version

# TODO: move to independent linux kernel module builder
# https://developer.nvidia.com/embedded/jetson-linux-archive
# https://developer.nvidia.com/downloads/embedded/l4t/r36_release_v4.3/sources/public_sources.tbz2
kernel_url = 'https://developer.nvidia.com/downloads/embedded/l4t/r36_release_v4.0/sources/public_sources.tbz2'

package['build_args'] = {
    'KERNEL_URL': kernel_url
}

if not os.path.isfile('Module.symvers'):
    subprocess.run(
        "cp /lib/modules/$(uname -r)/build/Module.symvers .", 
        cwd=os.path.dirname(__file__), 
        shell=True, check=False,
    )
