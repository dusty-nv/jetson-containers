from jetson_containers import github_latest_tag


def create_package(version, branch=None, default=False) -> list:
    pkg = package.copy()
    wanted_version = github_latest_tag('rhasspy/wyoming-satellite') if version == 'latest' else version

    if wanted_version.startswith("v"):
        wanted_version = wanted_version[1:]

    pkg['name'] = f'wyoming-assist-microphone:{wanted_version}'
    pkg['build_args'] = {
        'SATELLITE_VERSION': wanted_version,
        'SATELLITE_BRANCH': branch or f"v{wanted_version}",
    }

    builder = pkg.copy()
    builder['name'] = f'wyoming-assist-microphone:{wanted_version}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'wyoming-assist-microphone'
        builder['alias'] = 'wyoming-assist-microphone:builder'

    return pkg, builder

package = [
    create_package("master", branch="master", default=True),
]
