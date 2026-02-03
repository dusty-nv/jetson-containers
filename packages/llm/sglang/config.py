from jetson_containers import CUDA_VERSION, IS_SBSA, update_dependencies
from packaging.version import Version

def sglang(version, version_spec=None, requires=None, depends=None, default=False):
    pkg = package.copy()

    if requires:
        pkg['requires'] = requires

    if not version_spec:
        version_spec = version

    if depends:
        pkg['depends'] = update_dependencies(pkg['depends'], depends)

    pkg['name'] = f'sglang:{version}'

    pkg['build_args'] = {
        'SGLANG_VERSION': version,
        'SGLANG_VERSION_SPEC': version_spec,
        'IS_SBSA': IS_SBSA
    }

    builder = pkg.copy()

    builder['name'] = f'sglang:{version}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'sglang'
        builder['alias'] = 'sglang:builder'

    return pkg, builder

package = [
    sglang('0.5.9', '0.5.9', depends=['flashinfer', 'sgl-kernel:0.5.9', 'torchao:0.9.0'], default=True), # Compatible with CUDA 13 (Spark and Thor)
]
