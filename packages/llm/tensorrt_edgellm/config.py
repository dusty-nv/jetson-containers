from jetson_containers import CUDA_VERSION, CUDA_ARCHITECTURES

def tensorrt_edgellm(version, branch=None, requires=None, default=False):
    pkg = package.copy()

    pkg['name'] = f'tensorrt_edgellm:{version}'

    if not branch:
        branch = f'v{version}'

    pkg['build_args'] = {
        'TENSORRT_EDGELLM_VERSION': version,
        'TENSORRT_EDGELLM_BRANCH': branch,
        'CUDA_ARCHITECTURES': ';'.join([str(x) for x in CUDA_ARCHITECTURES]),
    }

    if requires:
        pkg['requires'] = requires

    builder = pkg.copy()
    builder['name'] = f'tensorrt_edgellm:{version}-builder'
    builder['build_args'] = {**pkg['build_args'], 'FORCE_BUILD': 'on'}

    if default:
        pkg['alias'] = 'tensorrt_edgellm'
        builder['alias'] = 'tensorrt_edgellm:builder'

    return pkg, builder

package = [
    tensorrt_edgellm('0.5.0', requires='>=36', default=True),
]
