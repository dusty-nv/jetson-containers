from jetson_containers import CUDA_VERSION


def fastvideo(version, requires=None, default=False):
    pkg = package.copy()

    if requires:
        pkg['requires'] = requires

    pkg['name'] = f'fastvideo:{version}'

    pkg['build_args'] = {
        'FASTVIDEO_VERSION': version,
    }

    builder = pkg.copy()

    builder['name'] = f'fastvideo:{version}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'fastvideo'
        builder['alias'] = 'fastvideo:builder'

    return pkg, builder


package = [
    fastvideo('0.1.7', default=True),
]
