from jetson_containers import github_latest_tag


def create_package(version, default=False) -> list:
    pkg = package.copy()
    wanted_version = github_latest_tag('eudev-project/eudev') if version == 'latest' else version

    pkg['name'] = f'eudev:{version}'
    pkg['build_args'] = {
        'EUDEV_VERSION': wanted_version,
    }

    if default:
        pkg['alias'] = 'eudev'

    return pkg

package = [
    create_package('latest', default=True),
]
