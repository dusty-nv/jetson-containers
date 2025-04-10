from jetson_containers import CUDA_ARCHITECTURES

def dynamo(version, requires=None, default=False):
    pkg = package.copy()

    if requires:
        pkg['requires'] = requires   

    pkg['name'] = f'dynamo:{version}'

    pkg['build_args'] = {
        'CUDAARCHS': ';'.join([str(x) for x in CUDA_ARCHITECTURES]),
        'DYNAMO_VERSION': version,
    }

    builder = pkg.copy()

    builder['name'] = f'dynamo:{version}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'dynamo'
        builder['alias'] = 'dynamo:builder'

    return pkg, builder

package = [
    dynamo('0.2.0', default=True)
]
