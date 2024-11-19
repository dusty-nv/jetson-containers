from jetson_containers import handle_text_request


def create_package(version, default=False) -> list:
    pkg = package.copy()
    url = 'https://raw.githubusercontent.com/rhasspy/wyoming-faster-whisper/master/wyoming_faster_whisper/VERSION'
    wanted_version = handle_text_request(url) if version == 'latest' else version
    pkg['name'] = f'wyoming-whisper:{version}'

    pkg['build_args'] = {
        'WYOMING_WHISPER_VERSION': wanted_version if not '.' in wanted_version else f'v{wanted_version}',
    }

    if default:
        pkg['alias'] = 'wyoming-whisper'

    return pkg

package = [
    create_package("master", default=True),
]
