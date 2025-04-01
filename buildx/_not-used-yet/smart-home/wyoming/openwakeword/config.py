from jetson_containers import handle_text_request


def create_package(version, branch=None, default=False) -> list:
    pkg = package.copy()

    if not branch:
        branch = f'v{version}'

    wanted_version = handle_text_request(f'https://raw.githubusercontent.com/rhasspy/wyoming-openwakeword/{branch}/wyoming_openwakeword/VERSION')
    pkg['name'] = f'wyoming-openwakeword:{wanted_version}'

    pkg['build_args'] = {
        'WYOMING_OPENWAKEWORD_VERSION': wanted_version,
        'WYOMING_OPENWAKEWORD_BRANCH': branch,
    }

    builder = pkg.copy()
    builder['name'] = f'wyoming-openwakeword:{wanted_version}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'wyoming-openwakeword'
        builder['alias'] = 'wyoming-openwakeword:builder'

    return pkg, builder

package = [
    create_package("1.10.1", branch="master", default=True),
    create_package("1.10.0"),
]
