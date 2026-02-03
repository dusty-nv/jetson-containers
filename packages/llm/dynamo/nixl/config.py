from jetson_containers import CUDA_VERSION, IS_SBSA, IS_TEGRA, SYSTEM_ARM
from packaging.version import Version

def nixl(version, version_spec=None, requires=None, default=False):
    pkg = package.copy()

    if requires:
        pkg['requires'] = requires

    if not version_spec:
        version_spec = version

    pkg['name'] = f'nixl:{version}'

    pkg['build_args'] = {
        'NIXL_VERSION': version,
        'NIXL_VERSION_SPEC': version_spec,
        'IS_TEGRA': IS_TEGRA,
        'IS_SBSA': IS_SBSA,
        'SYSTEM_ARM': SYSTEM_ARM,
        'UCX_TAG': 'master',
    }

    builder = pkg.copy()

    builder['name'] = f'nixl:{version}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'nixl'
        builder['alias'] = 'nixl:builder'

    # Ensure gdrcopy is not in depends, favor cudastack
    if 'gdrcopy' in builder['depends']:
        builder['depends'].remove('gdrcopy')

    if 'cudastack' not in builder['depends']:
        builder['depends'].append('cudastack')

    return pkg, builder

package = [
    nixl('0.9.0', '0.9.0', default=True),
]
