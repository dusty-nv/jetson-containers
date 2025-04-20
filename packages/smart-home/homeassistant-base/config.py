import requests
import yaml

from jetson_containers import github_latest_tag


def latest_deps_versions(branch_name):
    url = f"https://raw.githubusercontent.com/home-assistant/docker-base/{branch_name}/ubuntu/build.yaml"
    response = requests.get(url)

    if response.status_code != 200:
        print(f"Failed to fetch the file. Status code: {response.status_code} for URL: {url}")
        return None

    try:
        yaml_content = yaml.safe_load(response.text)
        bashio_version = yaml_content.get('args', {}).get('BASHIO_VERSION', None)
        tempio_version = yaml_content.get('args', {}).get('TEMPIO_VERSION', None)
        s6_overlay_version = yaml_content.get('args', {}).get('S6_OVERLAY_VERSION', None)
        return bashio_version, tempio_version, s6_overlay_version
    except yaml.YAMLError as e:
        print(f"Failed to parse YAML content: {e}")
        return None

def create_package(version, default=False) -> list:
    pkg = package.copy()
    try:
        wanted_version = github_latest_tag('home-assistant/docker-base') if version == 'latest' else version
        bashio_version, tempio_version, s6_overlay_version = latest_deps_versions(wanted_version)
    except Exception as e:
        print(f"Failed to fetch the latest version of dependencies: {e}")
        bashio_version, tempio_version, s6_overlay_version = None, None, None

    pkg['name'] = f'homeassistant-base:{version}'
    pkg['build_args'] = {
        'BASHIO_VERSION': bashio_version,
        'TEMPIO_VERSION': tempio_version,
        'S6_OVERLAY_VERSION': s6_overlay_version,
    }

    if default:
        pkg['alias'] = 'homeassistant-base'

    return pkg

package = [
    create_package('latest', default=True),
]
