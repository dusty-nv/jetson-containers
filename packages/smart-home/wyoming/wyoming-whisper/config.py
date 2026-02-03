from jetson_containers import github_latest_tag

def create_package(version, branch=None, default=False) -> list:
    pkg = package.copy()
    wanted_version = github_latest_tag('rhasspy/wyoming-faster-whisper') if version == 'latest' else version

    if wanted_version.startswith("v"):
        wanted_version = wanted_version[1:]

    pkg['name'] = f'wyoming-whisper:{wanted_version}'

    pkg['build_args'] = {
        'WYOMING_WHISPER_VERSION': wanted_version,
        'WYOMING_WHISPER_BRANCH': branch or f"v{wanted_version}",
    }

    builder = pkg.copy()
    builder['name'] = f'wyoming-whisper:{wanted_version}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'wyoming-whisper'
        builder['alias'] = 'wyoming-whisper:builder'

    return pkg, builder

package = [
    create_package("3.1.0", default=True),
]
