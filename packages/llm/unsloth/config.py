from jetson_containers import CUDA_VERSION, IS_SBSA, update_dependencies
from packaging.version import Version

def unsloth(version, branch=None, requires=None, default=False, depends=None):
    pkg = package.copy()

    if requires:
        pkg['requires'] = requires

    if depends:
        pkg['depends'] = update_dependencies(pkg['depends'], depends)

    suffix = branch if branch else version
    branch = branch if branch else f'v{version}'

    pkg['name'] = f'unsloth:{suffix}'
    pkg['build_args'] = {
        'UNSLOTH_VERSION': version,
        'UNSLOTH_BRANCH': branch,
        'IS_SBSA': IS_SBSA
    }

    builder = pkg.copy()
    builder['name'] = f'unsloth:{suffix}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'unsloth'
        builder['alias'] = 'unsloth:builder'

    return pkg, builder

package = [
    # 0.6.5 compatible with jetson https://github.com/unsloth-project/unsloth/pull/9735
    unsloth(version='January-2026', default=True),
]
