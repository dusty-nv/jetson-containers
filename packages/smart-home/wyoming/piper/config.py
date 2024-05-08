from jetson_containers import handle_text_request


def create_package(version, default=False) -> list:
    pkg = package.copy()
    url = 'https://raw.githubusercontent.com/rhasspy/wyoming-piper/master/wyoming_piper/VERSION'
    wanted_version = handle_text_request(url) if version == 'latest' else version
    pkg['name'] = f'wyoming-piper:{version}'

    pkg['build_args'] = {
        'WYOMING_PIPER_VERSION': wanted_version,
    }

    if default:
        pkg['alias'] = 'wyoming-piper'

    return pkg

package = [
    create_package("master", default=True),
]
