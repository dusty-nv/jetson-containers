def isaaclab(version, default=False):
    pkg = package.copy()
    pkg['name'] = f'isaaclab:{version}'

    pkg['build_args'] = {
        'ISAACLAB_VERSION': version,
    }

    builder = pkg.copy()

    builder['name'] = f'isaaclab:{version}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'isaaclab'
        builder['alias'] = 'isaaclab:builder'

    return pkg, builder


package = [
    isaaclab('3.0.0', default=True),
]
