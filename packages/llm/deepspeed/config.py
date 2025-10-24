from jetson_containers import CUDA_ARCHITECTURES

def deepspeed(version, branch=None, default=False, build_args=None):
    pkg = package.copy()

    if not branch:
        branch = f'v{version}'

    pkg['name'] = f'deepspeed:{version}'

    pkg['build_args'] = {
        'DEEPSPEED_VERSION': version,
        'DEEPSPEED_BRANCH': branch
    }

    if build_args:
        pkg['build_args'].update(build_args)

    builder = pkg.copy()

    builder['name'] = f'deepspeed:{version}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'deepspeed'
        builder['alias'] = 'deepspeed:builder'

    return pkg, builder

package = [
    deepspeed('0.9.5', build_args={'DS_BUILD_OPS': 0}),
    deepspeed('0.15.2'),
    deepspeed('0.18.1', default=True),
]
