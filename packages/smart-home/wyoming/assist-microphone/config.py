from jetson_containers import handle_text_request


def create_package(version, branch=None, default=False) -> list:
    pkg = package.copy()

    if not branch:
        branch = f'v{version}'

    wanted_version = handle_text_request(f'https://raw.githubusercontent.com/rhasspy/wyoming-satellite/{branch}/wyoming_satellite/VERSION')
    pkg['name'] = f'wyoming-assist-microphone:{wanted_version}'

    pkg['build_args'] = {
        'SATELLITE_VERSION': wanted_version,
        'SATELLITE_BRANCH': branch
    }

    builder = pkg.copy()
    builder['name'] = f'wyoming-assist-microphone:{wanted_version}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'wyoming-assist-microphone'
        builder['alias'] = 'wyoming-assist-microphone:builder'

    return pkg, builder

package = [
    create_package("1.3.0", branch="master", default=True),
    create_package("1.2.0"),
]
