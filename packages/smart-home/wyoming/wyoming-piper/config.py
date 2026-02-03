from jetson_containers import github_latest_tag

def create_package(version, branch=None, default=False) -> list:
    pkg = package.copy()
    wanted_version = github_latest_tag('rhasspy/wyoming-piper') if version == 'latest' else version

    if wanted_version.startswith("v"):
        wanted_version = wanted_version[1:]

    pkg['name'] = f'wyoming-piper:{wanted_version}'

    pkg['build_args'] = {
        'WYOMING_PIPER_VERSION': wanted_version,
        'WYOMING_PIPER_BRANCH': branch or f"v{wanted_version}",
    }

    builder = pkg.copy()
    builder['name'] = f'wyoming-piper:{wanted_version}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'wyoming-piper'
        builder['alias'] = 'wyoming-piper:builder'

    return pkg, builder

package = [
    create_package("2.2.1", default=True),
]
