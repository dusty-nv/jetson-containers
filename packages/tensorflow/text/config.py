from jetson_containers import CUDA_ARCHITECTURES

def tensorflow_text(version, requires=None, default=False):
    pkg = package.copy()

    if requires:
        pkg['requires'] = requires   

    pkg['name'] = f'tensorflow_text:{version}'

    pkg['build_args'] = {
        'CUDAARCHS': ';'.join([str(x) for x in CUDA_ARCHITECTURES]),
        'TENSORFLOW_TEXT_VERSION': version,
    }

    builder = pkg.copy()

    builder['name'] = f'tensorflow-text:{version}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'tensorflow_text'
        builder['alias'] = 'tensorflow_text:builder'

    return pkg #, builder

package = [
    tensorflow_text('2.18.0', default=True)
]
