from jetson_containers import get_json_value_from_url, github_latest_tag


def create_package(version, default=False) -> list:
    pkg = package.copy()
    wanted_version = github_latest_tag('home-assistant/supervised-installer') if version == 'latest' else version

    pkg['name'] = f'homeassistant-supervised:{version}'
    pkg['build_args'] = {
        # Home Assistant OS-Agent (Only the latest release is supported) - https://github.com/home-assistant/architecture/blob/master/adr/0014-home-assistant-supervised.md#supported-operating-system-system-dependencies-and-versions
        'OS_AGENT_VERSION': github_latest_tag('home-assistant/os-agent'),
        'SUPERVISED_INSTALLER_VERSION': wanted_version,
    }

    if default:
        pkg['alias'] = 'homeassistant-supervised'

    return pkg

package = [
    create_package('main', default=True),
]
