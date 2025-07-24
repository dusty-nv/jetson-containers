from jetson_containers import CUDA_ARCHITECTURES, IS_SBSA


def sgl_kernel(version, branch=None, default=False):
    pkg = package.copy()

    if not branch:
        branch = f'v{version}'

    pkg['name'] = f'sgl-kernel:{version}'

    pkg['build_args'] = {
        'SGL_KERNEL_VERSION': version,
        'SGL_KERNEL_BRANCH': branch,
        'IS_SBSA': IS_SBSA,
        'TORCH_CUDA_ARCH_LIST': ';'.join([f'{x/10:.1f}' for x in CUDA_ARCHITECTURES])
    }

    builder = pkg.copy()

    builder['name'] = f'sgl-kernel:{version}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'sgl-kernel'
        builder['alias'] = 'sgl-kernel:builder'

    return pkg, builder

package = [
    sgl_kernel('0.2.7', branch='main', default=True),
]
