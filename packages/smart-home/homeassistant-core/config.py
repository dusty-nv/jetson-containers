import requests
from packaging.version import Version


def get_latest_stable_version(fallback='2024.3.1'):
    try:
        response = requests.get('https://version.home-assistant.io/stable.json')
        if response.status_code == 200:
            data = response.json()
            return data.get('homeassistant', { 'default': fallback }).get('default', fallback).strip()
        else:
            print("Failed to fetch version information. Status code:", response.status_code)
            return fallback
    except Exception as e:
        print("An error occurred:", e)
        return fallback


def create_package(version, default=False) -> list:
    pkg = package.copy()
    wanted_version = get_latest_stable_version() if version == 'latest' else version
    pkg['name'] = f'homeassistant-core:{version}'
    ha_version = Version(wanted_version)

    if ha_version.major >= 2024:
        if ha_version.minor >= 4:
            required_python = 'python:3.12'
        else:
            required_python = 'python:3.11'
    elif ha_version.major >= 2023 and ha_version.minor >= 8:
        required_python = 'python:3.11'
    else:
        required_python = 'python'

    pkg['depends'].extend([required_python])
    pkg['build_args'] = {
        'HA_BRANCH': wanted_version,
    }

    if default:
        pkg['alias'] = 'homeassistant-core'

    return pkg

package = [
    # latest
    create_package('latest', default=True),
    # specific version
    # create_package('2024.3.1', default=True),
]
