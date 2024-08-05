from jetson_containers import handle_text_request


def create_package(version, default=False) -> list:
    pkg = package.copy()
    url = 'https://raw.githubusercontent.com/rhasspy/wyoming-satellite/master/wyoming_satellite/VERSION'
    wanted_version = handle_text_request(url) if version == 'latest' else version
    pkg['name'] = f'wyoming-assist-microphone:{version}'

    pkg['build_args'] = {
        'SATELLITE_VERSION': wanted_version,
    }

    if default:
        pkg['alias'] = 'wyoming-assist-microphone'

    return pkg

package = [
    create_package("latest", default=True),
]
