from jetson_containers import CUDA_VERSION, IS_SBSA, update_dependencies
from packaging.version import Version

def mistral_common(version, branch=None, requires=None, default=False, depends=None):
    pkg = package.copy()

    if requires:
        pkg['requires'] = requires

    if depends:
        pkg['depends'] = update_dependencies(pkg['depends'], depends)

    suffix = branch if branch else version
    branch = branch if branch else f'v{version}'

    pkg['name'] = f'mistral_common:{suffix}'
    pkg['build_args'] = {
        'MISTRAL_COMMON_VERSION': version,
        'MISTRAL_COMMON_BRANCH': branch,
        'IS_SBSA': IS_SBSA
    }

    builder = pkg.copy()
    builder['name'] = f'mistral_common:{suffix}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'mistral_common'
        builder['alias'] = 'mistral_common:builder'

    return pkg, builder

package = [
    mistral_common(version='1.8.8', default=True),
]
