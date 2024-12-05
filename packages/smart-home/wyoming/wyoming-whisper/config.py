from jetson_containers import handle_text_request


def create_package(version, branch=None, default=False) -> list:
    pkg = package.copy()

    if not branch:
        branch = f'v{version}'

    url = f'https://raw.githubusercontent.com/rhasspy/wyoming-faster-whisper/{branch}/wyoming_faster_whisper/VERSION'
    wanted_version = handle_text_request(url)
    pkg['name'] = f'wyoming-whisper:{wanted_version}'

    pkg['build_args'] = {
        'WYOMING_WHISPER_VERSION': wanted_version,
        'WYOMING_WHISPER_BRANCH': branch
    }

    builder = pkg.copy()
    builder['name'] = f'wyoming-whisper:{wanted_version}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'wyoming-whisper'
        builder['alias'] = 'wyoming-whisper:builder'

    return pkg, builder

package = [
    create_package("2.3.0", default=True),
    create_package("2.2.0"),
]
