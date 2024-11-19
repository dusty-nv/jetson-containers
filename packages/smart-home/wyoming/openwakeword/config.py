from jetson_containers import handle_text_request


def create_package(version, default=False) -> list:
    pkg = package.copy()
    url = 'https://raw.githubusercontent.com/rhasspy/wyoming-openwakeword/master/wyoming_openwakeword/VERSION'
    wanted_version = handle_text_request(url) if version == 'latest' else version
    pkg['name'] = f'wyoming-openwakeword:{version}'

    pkg['build_args'] = {
        'WYOMING_OPENWAKEWORD_VERSION': wanted_version,
    }

    if default:
        pkg['alias'] = 'wyoming-openwakeword'

    return pkg

package = [
    create_package("latest", default=False),
    create_package("1.10.0", default=True),
]
