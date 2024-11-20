from jetson_containers import github_latest_tag


def create_package(version, default=False):
    pkg = package.copy()
    wanted_version = github_latest_tag('closeio/ciso8601') if version == 'latest' else version

    pkg['name'] = f'ciso8601:{wanted_version}'
    pkg['build_args'] = {
        'CISO8601_VERSION': wanted_version,
    }

    builder = pkg.copy()
    builder['name'] = f'ciso8601:{wanted_version}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'ciso8601'
        builder['alias'] = 'ciso8601:builder'

    return pkg, builder

package = [
    create_package('latest', default=True),
]
