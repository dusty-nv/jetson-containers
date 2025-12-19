from jetson_containers import CUDA_VERSION, SYSTEM_ARM, CUDA_ARCHITECTURES

def gptqmodel(version, branch=None, default=False):
    pkg = package.copy()

    pkg['name'] = f'gptqmodel:{version}'

    if not branch:
        branch = version

    pkg['build_args'] = {
        ''
        'GPTQMODEL_VERSION': version,
        'GPTQMODEL_BRANCH': branch,
        'TORCH_CUDA_ARCH_LIST': ' '.join([f'{x / 10:.1f}' for x in CUDA_ARCHITECTURES]),
    }

    builder = pkg.copy()

    builder['name'] = f'gptqmodel:{version}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'gptqmodel'
        builder['alias'] = 'gptqmodel:builder'

    return pkg, builder


package = [
    gptqmodel('5.6.12', default=True),
]
