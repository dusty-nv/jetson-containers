from jetson_containers import update_dependencies, get_json_value_from_url
from packaging.version import Version


def create_package(version, sqlite_version='3.40.1', default=False) -> list:
    pkg = package.copy()
    url = 'https://version.home-assistant.io/stable.json'
    wanted_version = get_json_value_from_url(url, 'homeassistant.default') if version == 'latest' else version
    pkg['name'] = f'homeassistant-core:{wanted_version}'
    ha_version = Version(wanted_version)

    if ha_version.major >= 2025:
        required_python = 'python:3.13'
    elif ha_version.major >= 2024:
        if ha_version.minor >= 4:
            required_python = 'python:3.12'
        else:
            required_python = 'python:3.11'
    elif ha_version.major >= 2023 and ha_version.minor >= 8:
        required_python = 'python:3.11'
    else:
        required_python = 'python'

    pkg['depends'] = update_dependencies(pkg['depends'], [required_python])

    pkg['build_args'] = {
        'HA_VERSION': wanted_version,
        'SQLITE_VERSION': sqlite_version
    }

    if default:
        pkg['alias'] = 'homeassistant-core'

    return pkg

package = [
    create_package('2026.1.3', default=True),
]
