from jetson_containers import github_latest_tag


def create_package(version, default=False):
    pkg = package.copy()
    wanted_version = github_latest_tag('home-assistant-libs/psutil-home-assistant') if version == 'latest' else version

    pkg['name'] = f'psutil-home-assistant:{wanted_version}'
    pkg['build_args'] = {
        'PSUTIL_HA_VERSION': wanted_version,
    }

    builder = pkg.copy()
    builder['name'] = f'psutil-home-assistant:{wanted_version}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'psutil-home-assistant'
        builder['alias'] = 'psutil-home-assistant:builder'

    return pkg, builder

package = [
    create_package('latest', default=True),
]
