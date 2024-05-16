def create_package(version, default=False) -> list:
    pkg = package.copy()

    pkg['name'] = f'go-lang:{version}'
    pkg['build_args'] = {
        'GO_VERSION': version,
    }

    if default:
        pkg['alias'] = 'go-lang'

    return pkg

package = [
    create_package('1.19', default=True),
]
