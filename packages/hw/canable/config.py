# TODO: refactor with common dependency between pl2303/cannable 
#       for generic Linux kernel module builder of external drivers
import os
import subprocess

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
