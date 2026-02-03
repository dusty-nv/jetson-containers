from jetson_containers import github_latest_tag

def faster_whisper(version, branch=None, default=False) -> list:
    pkg = package.copy()
    wanted_version = github_latest_tag('https://github.com/SYSTRAN/faster-whisper') if version == 'latest' else version

    if wanted_version.startswith("v"):
        wanted_version = wanted_version[1:]

    pkg['name'] = f'faster-whisper:{wanted_version}'

    pkg['build_args'] = {
        'FASTER_WHISPER_VERSION': wanted_version,
        'FASTER_WHISPER_BRANCH': branch or f"v{wanted_version}",
    }

    builder = pkg.copy()
    builder['name'] = f'faster-whisper:{wanted_version}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'faster-whisper'
        builder['alias'] = 'faster-whisper:builder'

    return pkg, builder

package = [
    faster_whisper("1.2.1", default=True),
]
