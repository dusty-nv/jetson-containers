from jetson_containers import CUDA_ARCHITECTURES, IS_SBSA, CUDA_VERSION, update_dependencies


def sgl_kernel(version, branch=None, depends=None, default=False):
    pkg = package.copy()

    if not branch:
        branch = f'v{version}'

    if depends:
        pkg['depends'] = update_dependencies(pkg['depends'], depends)

    pkg['name'] = f'sgl-kernel:{version}'

    pkg['build_args'] = {
        'SGL_KERNEL_VERSION': version,
        'SGL_KERNEL_BRANCH': branch,
        'IS_SBSA': IS_SBSA,
        'CUDA_VERSION': CUDA_VERSION,
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
    sgl_kernel('0.5.3', depends=['torchao:0.9.0'], default=False),
    # Note: this version points to a specific commit at which the patch (sm_87-0.5.4.diff)
    # for CMakeLists.txt was created.
    # You can increase the branch/commit to get newer versions if there are no changes
    # in CMakeLists.txt at commit 88568c01eb99698eceef9a40b5f481e37c0b89d0
    sgl_kernel('0.5.4', depends=['torchao:0.9.0'], default=False),
    sgl_kernel('0.4.0', branch='main', default=True),
]
