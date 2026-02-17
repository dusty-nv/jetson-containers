
def torch_memory_saver(version, requires=None, default=False):
    pkg = package.copy()

    pkg['name'] = f"torch-memory-saver:{version.split('-')[0]}"  # remove any -rc* suffix

    if requires:
        pkg['requires'] = requires

    if len(version.split('.')) < 3:
        version = version + '.0'

    pkg['build_args'] = {
        'TORCH_MEMORY_SAVER_VERSION': version,
    }

    builder = pkg.copy()
    builder['name'] = builder['name'] + '-builder'
    builder['build_args'] = {**builder['build_args'], 'FORCE_BUILD': 'on'}

    if default:
        pkg['alias'] = 'torch-memory-saver'
        builder['alias'] = 'torch-memory-saver:builder'

    return pkg, builder


package = [
    torch_memory_saver('0.0.9', requires='>=36', default=True),
]
