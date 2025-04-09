from jetson_containers import github_latest_tag

def create_package(version, default=False) -> list:
    pkg = package.copy()
    wanted_version = github_latest_tag('home-assistant/docker-base') if version == 'latest' else version

    pkg['name'] = f'homeassistant-base:{version}'
    pkg['build_args'] = {
        'BASHIO_VERSION': '0.15.0',     # Using latest stable versions
        'TEMPIO_VERSION': '2024.02.0',  # instead of fetching from raw.githubusercontent.com
        'S6_OVERLAY_VERSION': '3.1.6.2',
    }

    if default:
        pkg['alias'] = 'homeassistant-base'

    return pkg

package = [
    create_package('latest', default=True),
]