from jetson_containers import L4T_VERSION, github_latest_tag, log_warning
from packaging.version import parse

def homeassistant_base(version, **kwargs):
    """
    Defines the Home Assistant base container with dynamic version management.
    """
    if version == 'latest':
        # Get latest versions from respective repositories
        bashio_version = github_latest_tag('home-assistant/bashio')
        tempio_version = github_latest_tag('home-assistant/tempio')
        s6_overlay_version = github_latest_tag('just-containers/s6-overlay')

        if not all([bashio_version, tempio_version, s6_overlay_version]):
            log_warning("Failed to fetch latest versions for Home Assistant base components")
            return None
    else:
        # For specific versions, use the provided version for all components
        bashio_version = version
        tempio_version = version
        s6_overlay_version = version

    return _homeassistant_base(
        version,
        build_args={
            'BASHIO_VERSION': bashio_version,
            'TEMPIO_VERSION': tempio_version,
            'S6_OVERLAY_VERSION': s6_overlay_version,
        },
        **kwargs
    )

def create_package(version, default=False) -> list:
    pkg = package.copy()
    wanted_version = github_latest_tag('home-assistant/docker-base') if version == 'latest' else version

    # Get latest versions from respective repositories
    # bashio and tempio are part of the core repository
    core_version = github_latest_tag('home-assistant/core')
    s6_overlay_version = github_latest_tag('just-containers/s6-overlay')

    if not all([core_version, s6_overlay_version]):
        log_warning("Failed to fetch latest versions for Home Assistant base components")
        return None

    pkg['name'] = f'homeassistant-base:{version}'
    pkg['build_args'] = {
        'BASHIO_VERSION': core_version,    # bashio is part of core
        'TEMPIO_VERSION': core_version,    # tempio is part of core
        'S6_OVERLAY_VERSION': s6_overlay_version,
    }

    if default:
        pkg['alias'] = 'homeassistant-base'

    return pkg

package = [
    create_package('latest', default=True),
]