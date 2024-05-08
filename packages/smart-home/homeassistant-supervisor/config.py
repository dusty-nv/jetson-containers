from jetson_containers import get_json_value_from_url


def create_package(version, default=False) -> list:
    pkg = package.copy()
    version_url = 'https://version.home-assistant.io/stable.json'
    wanted_version = get_json_value_from_url(version_url, 'supervisor') if version == 'latest' else version

    pkg['name'] = f'homeassistant-supervisor:{version}'
    pkg['build_args'] = {
        'SUPERVISOR_VERSION': wanted_version,
        'COSIGN_VERSION': '2.2.3',
    }

    if default:
        pkg['alias'] = 'homeassistant-supervisor'

    return pkg

package = [
    create_package('latest', default=True),
]
