from jetson_containers import github_latest_tag


def create_package(version, default=False) -> list:
    pkg = package.copy()
    wanted_version = github_latest_tag('home-assistant/os-agent') if version == 'latest' else version

    pkg['name'] = f'homeassistant-os-agent:{version}'
    pkg['build_args'] = {
        # Home Assistant OS-Agent (Only the latest release is supported) 
        # https://github.com/home-assistant/architecture/blob/master/adr/0014-home-assistant-supervised.md#supported-operating-system-system-dependencies-and-versions
        'OS_AGENT_VERSION': wanted_version,
    }

    if default:
        pkg['alias'] = 'homeassistant-os-agent'

    return pkg

package = [
    create_package('latest', default=True),
]
