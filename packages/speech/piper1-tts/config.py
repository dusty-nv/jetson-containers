from jetson_containers import github_latest_tag

def piper(version: str, branch: str = None, default: bool = False) -> list:
    pkg = package.copy()
    wanted_version = github_latest_tag('OHF-voice/piper1-gpl') if version == 'latest' else version

    if wanted_version.startswith("v"):
        wanted_version = wanted_version[1:]

    pkg['name'] = f'piper1-tts:{wanted_version}'
    pkg['build_args'] = {
        'PIPER_VERSION': wanted_version,
        'PIPER_BRANCH': branch or f"v{wanted_version}",
    }

    builder = pkg.copy()
    builder['name'] = f'piper1-tts:{wanted_version}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'piper1-tts'
        builder['alias'] = 'piper1-tts:builder'

    return pkg, builder

package = [
    piper("1.4.0", branch="main", default=True),
]
