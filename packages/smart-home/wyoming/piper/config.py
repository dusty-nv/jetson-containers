from jetson_containers import handle_text_request

from packaging.version import Version


def create_package(version, branch=None, default=False) -> list:
    pkg = package.copy()

    if not branch:
        branch = f'v{version}'

    wanted_version = handle_text_request(f'https://raw.githubusercontent.com/rhasspy/wyoming-piper/{branch}/wyoming_piper/VERSION')

    # FIXME: wyoming-piper on branch v1.5.2 has incorrect version set in VERSION file.
    if Version(version) != Version(wanted_version):
        wanted_version = version

    pkg['name'] = f'wyoming-piper:{wanted_version}'

    pkg['build_args'] = {
        'WYOMING_PIPER_VERSION': wanted_version,
        'WYOMING_PIPER_BRANCH': branch,
    }

    builder = pkg.copy()
    builder['name'] = f'wyoming-piper:{wanted_version}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    if default:
        pkg['alias'] = 'wyoming-piper'
        builder['alias'] = 'wyoming-piper:builder'

    return pkg, builder

package = [
    create_package("1.5.2", default=True),
    create_package("1.5.0", default=False),
]
