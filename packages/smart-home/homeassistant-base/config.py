def create_package(version, default=False) -> list:
    pkg = package.copy()
    pkg['name'] = f'homeassistant-base:{version}'
    pkg['build_args'] = {
        'BASHIO_VERSION': '0.17.0',
        'TEMPIO_VERSION': '2024.11.2',
        'S6_OVERLAY_VERSION': '3.1.6.2'
    }

    if default:
        pkg['alias'] = 'homeassistant-base'

    return pkg

package = [
    create_package('master', default=True),
]
