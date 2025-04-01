import requests
import yaml

from jetson_containers import github_latest_tag
from jetson_containers import handle_text_request


def latest_deps_versions(branch_name):
    url = f"https://raw.githubusercontent.com/home-assistant/docker-base/{branch_name}/ubuntu/build.yaml"
    raw_text = handle_text_request(url)

    if raw_text is None:
        return None, None, None

    try:
        yaml_content = yaml.safe_load(raw_text)
        args = yaml_content.get('args', {})
        bashio_version = args.get('BASHIO_VERSION')
        tempio_version = args.get('TEMPIO_VERSION')
        s6_overlay_version = args.get('S6_OVERLAY_VERSION')
        return bashio_version, tempio_version, s6_overlay_version
    except yaml.YAMLError as e:
        print(f"[WARN] Failed to parse YAML from {url}: {e}")
        return None, None, None

def create_package(version, default=False) -> list:
    pkg = package.copy()
    wanted_version = github_latest_tag('home-assistant/docker-base') if version == 'latest' else version
    bashio_version, tempio_version, s6_overlay_version = latest_deps_versions(wanted_version)

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
