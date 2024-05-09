import requests


def get_latest_stable_version(fallback="1.10.0") -> str:
    try:
        response = requests.get('https://raw.githubusercontent.com/rhasspy/wyoming-openwakeword/master/wyoming_openwakeword/VERSION')
        if response.status_code == 200:
            return response.text.strip()
        else:
            print("Failed to fetch version information. Status code:", response.status_code)
            return fallback
    except Exception as e:
        print("An error occurred:", e)
        return fallback


def create_package(version, default=False) -> list:
    pkg = package.copy()
    wanted_version = get_latest_stable_version() if version == 'latest' else version
    pkg['name'] = f'wyoming-openwakeword:{version}'

    pkg['build_args'] = {
        'WYOMING_OPENWAKEWORD_VERSION': wanted_version,
    }

    if default:
        pkg['alias'] = 'wyoming-openwakeword'

    return pkg

package = [
    create_package("latest", default=True),
]
